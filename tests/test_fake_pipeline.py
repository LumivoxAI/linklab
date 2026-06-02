import socket
import asyncio
from typing import Any, cast
from collections import deque
from dataclasses import field, dataclass
from collections.abc import Callable, Coroutine

import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def pcm(value: int, frames: int) -> bytes:
    return bytes((value, 0)) * frames


class EventJournal:
    def __init__(self) -> None:
        self.events: list[object] = []
        self._changed = asyncio.Condition()

    async def record(self, event: object) -> None:
        async with self._changed:
            self.events.append(event)
            self._changed.notify_all()

    async def wait_for[T](
        self,
        event_type: type[T],
        *,
        after: int = 0,
        matching: Callable[[T], bool] | None = None,
    ) -> T:
        async with asyncio.timeout(1):
            async with self._changed:
                while True:
                    for event in self.events[after:]:
                        if isinstance(event, event_type) and (matching is None or matching(event)):
                            return event
                    await self._changed.wait()

    def of_type[T](self, event_type: type[T]) -> list[T]:
        return [event for event in self.events if isinstance(event, event_type)]


class FakePlaybackCallbacks:
    def __init__(self, journal: EventJournal) -> None:
        self.journal = journal
        self.client: linklab.VoiceClient | None = None
        self.output_id: linklab.OutputId | None = None
        self.received_frames = 0
        self.audio = bytearray()

    async def _record(self, event: object) -> None:
        await self.journal.record(event)

    async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None:
        await self._record(event)

    async def on_conversation_state(self, event: linklab.StateEvent) -> None:
        await self._record(event)

    async def on_transcript_update(self, event: linklab.TranscriptUpdateEvent) -> None:
        await self._record(event)

    async def on_transcript_final(self, event: linklab.TranscriptFinalEvent) -> None:
        await self._record(event)

    async def on_response_started(self, event: linklab.ResponseStartedEvent) -> None:
        await self._record(event)

    async def on_response_text_delta(self, event: linklab.ResponseTextDeltaEvent) -> None:
        await self._record(event)

    async def on_response_text_final(self, event: linklab.ResponseTextFinalEvent) -> None:
        await self._record(event)

    async def on_response_ended(self, event: linklab.ResponseEndedEvent) -> None:
        await self._record(event)

    async def on_response_cancelled(self, event: linklab.ResponseCancelledEvent) -> None:
        await self._record(event)

    async def on_output_started(self, event: linklab.OutputStartedEvent) -> None:
        self.output_id = event.output_id
        self.received_frames = 0
        self.audio.clear()
        await self._record(event)

    async def on_output_audio(self, event: linklab.OutputAudioEvent) -> None:
        assert type(event.audio) is bytes
        assert event.output_id == self.output_id
        assert event.start_frame == self.received_frames
        self.received_frames += len(event.audio) // 2
        self.audio.extend(event.audio)
        await self._record(event)

    async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None:
        assert event.output_id == self.output_id
        assert event.total_frames == self.received_frames
        await self._record(event)
        assert self.client is not None
        assert self.client.playback_finished(event.output_id, event.total_frames)
        self.output_id = None

    async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None:
        await self._record(event)

    async def on_error(self, event: linklab.ErrorEvent) -> None:
        await self._record(event)

    def interrupt(self, reason: linklab.PlaybackInterruptReason) -> bool:
        assert self.client is not None
        assert self.output_id is not None
        return self.client.playback_interrupted(
            self.output_id,
            self.received_frames,
            linklab.PlaybackPosition.EXACT,
            reason,
        )


class FakeWakelab:
    def __init__(self, client: linklab.VoiceClient, playback: FakePlaybackCallbacks) -> None:
        self.client = client
        self.playback = playback
        self.generation = 0

    def submit(
        self,
        audio: bytes,
        *,
        speech: bool,
        activated: bool,
        wake_word: str | None = None,
    ) -> linklab.AudioSubmitResult:
        result = self.client.submit_annotated_audio(
            linklab.AnnotatedAudio(audio, self.generation, False, speech, activated, wake_word)
        )
        if result is linklab.AudioSubmitResult.ACCEPTED and speech and self.playback.output_id is not None:
            assert self.playback.interrupt(linklab.PlaybackInterruptReason.BARGE_IN)
        return result


@dataclass(slots=True)
class TurnPlan:
    endpoint_frames: int
    transcript: str
    close_reason: linklab.InputCloseReason = linklab.InputCloseReason.ENDPOINT
    transcript_updates: tuple[str, ...] = ()
    text_deltas: tuple[str, ...] = ()
    text_final: str | None = None
    output_chunks: tuple[bytes, ...] = ()
    end_conversation: bool = False
    output_gate: asyncio.Event | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    writer_closed: bool = False


@dataclass(slots=True)
class InputCapture:
    plan: TurnPlan
    audio: bytearray = field(default_factory=bytearray)
    spans: list[tuple[int, bool, bytes]] = field(default_factory=list)
    processing_started: bool = False


class FakePipeline:
    def __init__(self, plans: list[TurnPlan]) -> None:
        self._plans = deque(plans)
        self.journal = EventJournal()
        self.inputs: dict[linklab.InputId, InputCapture] = {}
        self.tasks: set[asyncio.Task[None]] = set()

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationStartedEvent,
    ) -> None:
        await self.journal.record(event)

    async def on_input_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.InputStartedEvent,
    ) -> None:
        assert self._plans
        self.inputs[event.input_id] = InputCapture(self._plans.popleft())
        await self.journal.record(event)

    async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
        capture = self.inputs[event.input_id]
        capture.audio.extend(event.audio)
        capture.spans.append((event.start_frame, event.speech, event.audio))
        await self.journal.record(event)
        frames = len(capture.audio) // 2
        if frames < capture.plan.endpoint_frames or capture.processing_started:
            return

        assert frames == capture.plan.endpoint_frames
        capture.processing_started = True
        for revision, text in enumerate(capture.plan.transcript_updates, start=1):
            await session.update_transcript(event.input_id, revision, text)
        await session.close_input(event.input_id, capture.plan.close_reason)
        await session.finalize_transcript(event.input_id, capture.plan.transcript)
        task = asyncio.create_task(self._produce(session, event.input_id, capture.plan))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _produce(
        self,
        session: linklab.ServerSession,
        input_id: linklab.InputId,
        plan: TurnPlan,
    ) -> None:
        try:
            response = await session.start_response(input_id, end_conversation=plan.end_conversation)
            for text in plan.text_deltas:
                await response.send_text_delta(text)
            if plan.text_final is not None:
                await response.finalize_text(plan.text_final)
            if plan.output_chunks:
                output = await response.start_output()
                for index, audio in enumerate(plan.output_chunks):
                    await output.send_audio(audio)
                    if index == 0 and plan.output_gate is not None:
                        await plan.output_gate.wait()
                await output.finish()
            await response.finish()
        except linklab.WriterClosed:
            plan.writer_closed = True
        finally:
            plan.done.set()

    async def on_input_aborted(
        self,
        _session: linklab.ServerSession,
        event: linklab.InputAbortedEvent,
    ) -> None:
        await self.journal.record(event)

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        await self.journal.record(event)

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationCancelledEvent,
    ) -> None:
        await self.journal.record(event)

    async def stop(self) -> None:
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@dataclass(slots=True)
class Harness:
    server: linklab.VoiceServer
    client: linklab.VoiceClient
    callbacks: FakePlaybackCallbacks
    pipeline: FakePipeline
    wakelab: FakeWakelab

    async def close(self) -> None:
        await self.pipeline.stop()
        await self.client.close()
        await self.server.close()


async def open_harness(plans: list[TurnPlan], *, waiting_pre_roll_frames: int = 8_000) -> Harness:
    port = unused_port()
    pipeline = FakePipeline(plans)
    server = linklab.VoiceServer(
        linklab.ServerConfig(port, (PCM_16K,)),
        lambda _session: pipeline,
        object(),
    )
    await server.serve()
    callbacks = FakePlaybackCallbacks(EventJournal())
    client = linklab.VoiceClient(
        linklab.ClientConfig(
            f"ws://127.0.0.1:{port}",
            (PCM_16K,),
            waiting_pre_roll_frames=waiting_pre_roll_frames,
        ),
        callbacks,
        object(),
    )
    callbacks.client = client
    await client.connect()
    return Harness(server, client, callbacks, pipeline, FakeWakelab(client, callbacks))


def test_fake_pipeline_activation_sequential_turns_streaming_and_end_intent() -> None:
    async def scenario() -> None:
        first = TurnPlan(
            6,
            "hello",
            transcript_updates=("hel", "hello"),
            text_deltas=("streamed ", "answer"),
            text_final="streamed answer",
            output_chunks=(pcm(0x41, 2), pcm(0x42, 3)),
        )
        second = TurnPlan(5, "again", text_deltas=("bye",), text_final="bye", end_conversation=True)
        harness = await open_harness([first, second], waiting_pre_roll_frames=3)
        try:
            assert harness.wakelab.submit(pcm(0x10, 2), speech=False, activated=False) is (
                linklab.AudioSubmitResult.IGNORED_INACTIVE
            )
            assert (
                harness.wakelab.submit(pcm(0x11, 2), speech=False, activated=True, wake_word="lumivox")
                is linklab.AudioSubmitResult.ACCEPTED
            )
            assert harness.wakelab.submit(pcm(0x12, 2), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            assert harness.wakelab.submit(pcm(0x13, 2), speech=False, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(first.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )

            conversation = await harness.pipeline.journal.wait_for(linklab.ConversationStartedEvent)
            first_start = await harness.pipeline.journal.wait_for(linklab.InputStartedEvent)
            first_capture = harness.pipeline.inputs[first_start.input_id]
            assert conversation.wake_word == "lumivox"
            assert first_start.reason is linklab.InputStartReason.ACTIVATION
            assert bytes(first_capture.audio) == pcm(0x11, 2) + pcm(0x12, 2) + pcm(0x13, 2)
            assert first_capture.spans == [
                (0, False, pcm(0x11, 2)),
                (2, True, pcm(0x12, 2)),
                (4, False, pcm(0x13, 2)),
            ]
            assert harness.callbacks.audio == pcm(0x41, 2) + pcm(0x42, 3)
            playback = await harness.pipeline.journal.wait_for(linklab.PlaybackFinishedEvent)
            assert playback.played_frames == 5

            assert harness.wakelab.submit(pcm(0x21, 2), speech=False, activated=True) is (
                linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
            )
            assert harness.wakelab.submit(pcm(0x22, 2), speech=False, activated=True) is (
                linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
            )
            assert harness.wakelab.submit(pcm(0x23, 2), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(second.done.wait(), 1)
            ended = await harness.callbacks.journal.wait_for(linklab.ConversationEndedEvent)

            starts = harness.pipeline.journal.of_type(linklab.InputStartedEvent)
            assert [event.reason for event in starts] == [
                linklab.InputStartReason.ACTIVATION,
                linklab.InputStartReason.SPEECH,
            ]
            second_capture = harness.pipeline.inputs[starts[1].input_id]
            assert bytes(second_capture.audio) == pcm(0x21, 1) + pcm(0x22, 2) + pcm(0x23, 2)
            assert ended.reason is linklab.ConversationEndReason.COMPLETED
            assert [event.text for event in harness.callbacks.journal.of_type(linklab.ResponseTextDeltaEvent)] == [
                "streamed ",
                "answer",
                "bye",
            ]
        finally:
            await harness.close()

    run(scenario())


def test_fake_pipeline_no_speech_text_or_output_returns_to_waiting() -> None:
    async def scenario() -> None:
        plan = TurnPlan(3, "", close_reason=linklab.InputCloseReason.NO_SPEECH)
        harness = await open_harness([plan])
        try:
            assert harness.wakelab.submit(pcm(0x30, 3), speech=False, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(plan.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )

            assert [event.text for event in harness.callbacks.journal.of_type(linklab.TranscriptFinalEvent)] == [""]
            assert not harness.callbacks.journal.of_type(linklab.ResponseTextDeltaEvent)
            assert not harness.callbacks.journal.of_type(linklab.ResponseTextFinalEvent)
            assert not harness.callbacks.journal.of_type(linklab.OutputStartedEvent)
            assert [event.state for event in harness.callbacks.journal.of_type(linklab.StateEvent)] == [
                linklab.CoarseState.LISTENING,
                linklab.CoarseState.PROCESSING,
                linklab.CoarseState.RESPONDING,
                linklab.CoarseState.WAITING,
            ]
        finally:
            await harness.close()

    run(scenario())


def test_fake_pipeline_barge_in_interrupts_playback_and_rejects_stale_tts() -> None:
    async def scenario() -> None:
        output_gate = asyncio.Event()
        interrupted = TurnPlan(
            1,
            "first",
            text_deltas=("old",),
            output_chunks=(pcm(0x40, 2), pcm(0x41, 2)),
            end_conversation=True,
            output_gate=output_gate,
        )
        replacement = TurnPlan(3, "replacement", text_deltas=("new",), text_final="new")
        harness = await open_harness([interrupted, replacement], waiting_pre_roll_frames=2)
        try:
            assert harness.wakelab.submit(pcm(0x31, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await harness.callbacks.journal.wait_for(linklab.OutputAudioEvent)
            assert harness.wakelab.submit(pcm(0x50, 2), speech=False, activated=True) is (
                linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
            )
            assert harness.wakelab.submit(pcm(0x51, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )

            barge_start = await harness.pipeline.journal.wait_for(
                linklab.InputStartedEvent,
                matching=lambda event: event.reason is linklab.InputStartReason.BARGE_IN,
            )
            cancelled = await harness.callbacks.journal.wait_for(linklab.ResponseCancelledEvent)
            playback = await harness.pipeline.journal.wait_for(linklab.PlaybackInterruptedEvent)
            await asyncio.wait_for(replacement.done.wait(), 1)
            output_gate.set()
            await asyncio.wait_for(interrupted.done.wait(), 1)

            assert barge_start.interrupts_response_id == cancelled.response_id
            assert cancelled.reason is linklab.ResponseCancelReason.BARGE_IN
            assert playback.reason is linklab.PlaybackInterruptReason.BARGE_IN
            assert playback.played_frames == 2
            assert interrupted.writer_closed
            capture = harness.pipeline.inputs[barge_start.input_id]
            assert bytes(capture.audio) == pcm(0x50, 2) + pcm(0x51, 1)
            assert not harness.callbacks.journal.of_type(linklab.ConversationEndedEvent)
            assert harness.callbacks.journal.of_type(linklab.ResponseTextDeltaEvent)[-1].text == "new"
        finally:
            output_gate.set()
            await harness.close()

    run(scenario())


def test_fake_pipeline_user_cancel_orders_terminals_and_accepts_late_accounting() -> None:
    async def scenario() -> None:
        output_gate = asyncio.Event()
        plan = TurnPlan(1, "cancel", output_chunks=(pcm(0x60, 3), pcm(0x61, 1)), output_gate=output_gate)
        harness = await open_harness([plan])
        try:
            assert harness.wakelab.submit(pcm(0x30, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await harness.callbacks.journal.wait_for(linklab.OutputAudioEvent)
            assert harness.client.cancel_conversation(linklab.ConversationCancelReason.USER)
            assert harness.callbacks.interrupt(linklab.PlaybackInterruptReason.LOCAL_CANCEL)

            peer_cancel = await harness.pipeline.journal.wait_for(linklab.ConversationCancelledEvent)
            playback = await harness.pipeline.journal.wait_for(linklab.PlaybackInterruptedEvent)
            ended = await harness.callbacks.journal.wait_for(linklab.ConversationEndedEvent)
            output_gate.set()
            await asyncio.wait_for(plan.done.wait(), 1)

            terminals = [
                event
                for event in harness.callbacks.journal.events
                if isinstance(event, (linklab.ResponseCancelledEvent, linklab.ConversationEndedEvent))
            ]
            assert [type(event) for event in terminals] == [
                linklab.ResponseCancelledEvent,
                linklab.ConversationEndedEvent,
            ]
            assert peer_cancel.reason is linklab.ConversationCancelReason.USER
            assert playback.reason is linklab.PlaybackInterruptReason.LOCAL_CANCEL
            assert ended.reason is linklab.ConversationEndReason.CANCELLED
            assert plan.writer_closed
        finally:
            output_gate.set()
            await harness.close()

    run(scenario())
