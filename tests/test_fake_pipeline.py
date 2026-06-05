import socket
import asyncio
from typing import Any, Literal, cast
from collections import deque
from dataclasses import field, dataclass
from collections.abc import Callable, Coroutine

import pytest

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
        self.output_gate: asyncio.Event | None = None
        self.output_entered = asyncio.Event()
        self.fail_output = False

    async def _record(self, event: object) -> None:
        await self.journal.record(event)

    async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None:
        if event.state is linklab.ConnectionState.DISCONNECTED:
            self.output_id = None
            self.received_frames = 0
            self.audio.clear()
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
        self.output_entered.set()
        if self.output_gate is not None:
            await self.output_gate.wait()
        if self.fail_output:
            self.fail_output = False
            raise RuntimeError("fake playback failed")
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
    response_gate: asyncio.Event | None = None
    input_gate: asyncio.Event | None = None
    failure_stage: Literal["stt", "generation", "tts", "playback"] | None = None
    close_input: bool = True
    finalize_transcript: bool = True
    done: asyncio.Event = field(default_factory=asyncio.Event)
    writer_closed: bool = False
    failure_observed: bool = False


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
        self.responses: dict[linklab.ResponseId, TurnPlan] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.sessions: list[linklab.ServerSession] = []

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationStartedEvent,
    ) -> None:
        self.sessions.append(_session)
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
        if capture.plan.input_gate is not None:
            await capture.plan.input_gate.wait()
        if capture.plan.failure_stage == "stt":
            capture.plan.failure_observed = True
            capture.plan.done.set()
            raise RuntimeError("fake STT failed")
        frames = len(capture.audio) // 2
        if frames < capture.plan.endpoint_frames or capture.processing_started:
            return

        assert frames == capture.plan.endpoint_frames
        capture.processing_started = True
        if not capture.plan.close_input:
            capture.plan.done.set()
            return
        for revision, text in enumerate(capture.plan.transcript_updates, start=1):
            await session.update_transcript(event.input_id, revision, text)
        await session.close_input(event.input_id, capture.plan.close_reason)
        if not capture.plan.finalize_transcript:
            capture.plan.done.set()
            return
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
            self.responses[response.response_id] = plan
            if plan.failure_stage == "generation":
                try:
                    async with response:
                        for text in plan.text_deltas:
                            await response.send_text_delta(text)
                        raise RuntimeError("fake LLM failed")
                except RuntimeError:
                    plan.failure_observed = True
                return
            for text in plan.text_deltas:
                await response.send_text_delta(text)
            if plan.text_final is not None:
                await response.finalize_text(plan.text_final)
            if plan.output_chunks:
                output = await response.start_output()
                if plan.failure_stage == "tts":
                    try:
                        async with output:
                            await output.send_audio(plan.output_chunks[0])
                            raise RuntimeError("fake TTS failed")
                    except RuntimeError:
                        plan.failure_observed = True
                    return
                for index, audio in enumerate(plan.output_chunks):
                    await output.send_audio(audio)
                    if index == 0 and plan.output_gate is not None:
                        await plan.output_gate.wait()
                await output.finish()
                if plan.response_gate is not None:
                    await plan.response_gate.wait()
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
        plan = self.responses.get(event.response_id)
        if plan is not None and plan.failure_stage == "playback":
            plan.failure_observed = True
            raise RuntimeError("fake playback accounting failed")

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


async def open_harness(
    plans: list[TurnPlan],
    *,
    waiting_pre_roll_frames: int = 8_000,
    server_input_queue_frames: int = 32_000,
    client_playback_queue_ms: int = 2_000,
    reconnect: bool = False,
) -> Harness:
    port = unused_port()
    pipeline = FakePipeline(plans)
    server = linklab.VoiceServer(
        linklab.ServerConfig(port, (PCM_16K,), input_queue_frames=server_input_queue_frames),
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
            playback_queue_ms=client_playback_queue_ms,
            reconnect=reconnect,
        ),
        callbacks,
        object(),
    )
    if reconnect:

        async def reconnect_immediately(_delay: float) -> bool:
            return True

        setattr(client, "_wait_reconnect_delay", reconnect_immediately)
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


def test_fake_pipeline_simultaneous_client_and_server_close_leaves_no_live_work() -> None:
    async def scenario() -> None:
        output_gate = asyncio.Event()
        plan = TurnPlan(1, "closing", output_chunks=(pcm(0x70, 2), pcm(0x71, 1)), output_gate=output_gate)
        harness = await open_harness([plan])
        try:
            assert harness.wakelab.submit(pcm(0x30, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await harness.callbacks.journal.wait_for(linklab.OutputAudioEvent)

            await asyncio.gather(harness.client.close(), harness.server.close())
            output_gate.set()
            await asyncio.wait_for(plan.done.wait(), 1)

            assert harness.client.connection_state is linklab.ConnectionState.DISCONNECTED
            assert plan.writer_closed
            live_names = {
                task.get_name()
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task() and not task.done()
            }
            assert not {name for name in live_names if name.startswith("linklab-")}
        finally:
            output_gate.set()
            await harness.close()

    run(scenario())


@pytest.mark.parametrize(
    ("stage", "code", "reason"),
    [
        ("generation", linklab.ErrorCode.GENERATION_FAILED, linklab.ResponseCancelReason.GENERATION_FAILED),
        ("tts", linklab.ErrorCode.TTS_FAILED, linklab.ResponseCancelReason.TTS_FAILED),
    ],
)
def test_fake_pipeline_model_failure_is_scoped_and_next_turn_succeeds(
    stage: Literal["generation", "tts"],
    code: linklab.ErrorCode,
    reason: linklab.ResponseCancelReason,
) -> None:
    async def scenario() -> None:
        failed = TurnPlan(
            1,
            "failed turn",
            text_deltas=("partial",),
            output_chunks=(pcm(0x80, 1),) if stage == "tts" else (),
            failure_stage=stage,
        )
        recovered = TurnPlan(1, "recovered", text_deltas=("fresh",))
        harness = await open_harness([failed, recovered])
        try:
            assert harness.wakelab.submit(pcm(0x30, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(failed.done.wait(), 1)
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is code,
            )
            cancelled = await harness.callbacks.journal.wait_for(
                linklab.ResponseCancelledEvent,
                matching=lambda event: event.reason is reason,
            )
            waiting = await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            scoped = [event for event in harness.callbacks.journal.events if event in (error, cancelled, waiting)]
            assert scoped == [error, cancelled, waiting]
            assert failed.failure_observed

            recovery_start = len(harness.callbacks.journal.events)
            assert harness.wakelab.submit(pcm(0x31, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(recovered.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                after=recovery_start,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            assert [event.text for event in harness.callbacks.journal.of_type(linklab.ResponseTextDeltaEvent)] == [
                "fresh"
            ]
            assert len(harness.pipeline.sessions) == 1
        finally:
            await harness.close()

    run(scenario())


def test_fake_pipeline_stt_failure_closes_input_and_recovers_on_same_connection() -> None:
    async def scenario() -> None:
        failed = TurnPlan(1, "", failure_stage="stt")
        recovered = TurnPlan(1, "recovered")
        harness = await open_harness([failed, recovered])
        try:
            assert harness.wakelab.submit(pcm(0x40, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(failed.done.wait(), 1)
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.STT_FAILED,
            )
            final = await harness.callbacks.journal.wait_for(linklab.TranscriptFinalEvent)
            waiting = await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            terminals = [event for event in harness.callbacks.journal.events if event in (error, final, waiting)]
            assert terminals == [error, final, waiting]
            assert final.text == ""
            input_ = harness.pipeline.sessions[0]._core.validator._data.inputs[-1]
            assert isinstance(input_.terminal, linklab.InputClosedEvent)
            assert input_.terminal.reason is linklab.InputCloseReason.FAILED
            assert input_.terminal.accepted_end_frame == 1

            start = len(harness.callbacks.journal.events)
            assert harness.wakelab.submit(pcm(0x41, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(recovered.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                after=start,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            assert len(harness.callbacks.journal.of_type(linklab.ErrorEvent)) == 1
        finally:
            await harness.close()

    run(scenario())


@pytest.mark.parametrize("failure_owner", ["client", "server"])
def test_fake_pipeline_playback_failure_ends_only_its_conversation(failure_owner: str) -> None:
    async def scenario() -> None:
        producer_gate = asyncio.Event()
        response_gate = asyncio.Event()
        plan = TurnPlan(
            1,
            "playback",
            output_chunks=(pcm(0x50, 2),),
            output_gate=producer_gate if failure_owner == "client" else None,
            response_gate=response_gate if failure_owner == "server" else None,
            failure_stage="playback" if failure_owner == "server" else None,
        )
        harness = await open_harness([plan])
        if failure_owner == "client":
            harness.callbacks.fail_output = True
        try:
            assert harness.wakelab.submit(pcm(0x42, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            if failure_owner == "client":
                await asyncio.wait_for(harness.callbacks.output_entered.wait(), 1)
            else:
                await harness.callbacks.journal.wait_for(linklab.OutputEndedEvent)
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.PLAYBACK_FAILED,
            )
            cancelled = await harness.callbacks.journal.wait_for(linklab.ResponseCancelledEvent)
            ended = await harness.callbacks.journal.wait_for(linklab.ConversationEndedEvent)
            assert [
                type(event) for event in harness.callbacks.journal.events if event in (error, cancelled, ended)
            ] == [linklab.ErrorEvent, linklab.ResponseCancelledEvent, linklab.ConversationEndedEvent]
            assert cancelled.reason is linklab.ResponseCancelReason.PLAYBACK_FAILED
            assert ended.reason is linklab.ConversationEndReason.PLAYBACK_FAILED
            if failure_owner == "server":
                assert plan.failure_observed
            else:
                playback = await harness.pipeline.journal.wait_for(linklab.PlaybackInterruptedEvent)
                assert playback.reason is linklab.PlaybackInterruptReason.PLAYBACK_FAILED
        finally:
            producer_gate.set()
            response_gate.set()
            await harness.close()

    run(scenario())


def test_fake_pipeline_input_and_processing_timeouts_emit_complete_sequences() -> None:
    async def scenario() -> None:
        plan = TurnPlan(1, "", close_input=False)
        harness = await open_harness([plan])
        try:
            assert harness.wakelab.submit(pcm(0x60, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(plan.done.wait(), 1)
            session = harness.pipeline.sessions[0]
            input_deadline, _ = session._timeouts.deadlines
            assert input_deadline is not None
            session._timeouts.run_due(input_deadline)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.PROCESSING,
            )
            input_ = session._core.validator._data.inputs[-1]
            assert isinstance(input_.terminal, linklab.InputClosedEvent)
            assert input_.terminal.reason is linklab.InputCloseReason.MAX_DURATION
            assert input_.terminal.accepted_end_frame == 1

            processing_deadline, _ = session._timeouts.deadlines
            assert processing_deadline is not None
            session._timeouts.run_due(processing_deadline)
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.PROCESSING_TIMEOUT,
            )
            final = await harness.callbacks.journal.wait_for(linklab.TranscriptFinalEvent)
            waiting = await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            assert final.text == ""
            assert harness.callbacks.journal.events.index(error) < harness.callbacks.journal.events.index(final)
            assert harness.callbacks.journal.events.index(final) < harness.callbacks.journal.events.index(waiting)
        finally:
            await harness.close()

    run(scenario())


def test_fake_pipeline_waiting_timeout_ends_conversation_once() -> None:
    async def scenario() -> None:
        plan = TurnPlan(1, "waiting")
        harness = await open_harness([plan])
        try:
            assert harness.wakelab.submit(pcm(0x61, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(plan.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            session = harness.pipeline.sessions[0]
            deadline, _ = session._timeouts.deadlines
            assert deadline is not None
            session._timeouts.run_due(deadline)

            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.IDLE_TIMEOUT,
            )
            ended = await harness.callbacks.journal.wait_for(linklab.ConversationEndedEvent)
            assert harness.callbacks.journal.events.index(error) < harness.callbacks.journal.events.index(ended)
            assert ended.reason is linklab.ConversationEndReason.IDLE_TIMEOUT
            session._timeouts.run_due(deadline + 100)
            assert len(harness.callbacks.journal.of_type(linklab.ConversationEndedEvent)) == 1
        finally:
            await harness.close()

    run(scenario())


def test_fake_pipeline_server_input_overflow_rejects_queued_pcm_and_recovers() -> None:
    async def scenario() -> None:
        gate = asyncio.Event()
        saturated = TurnPlan(99, "", input_gate=gate)
        recovered = TurnPlan(1, "recovered")
        harness = await open_harness([saturated, recovered], server_input_queue_frames=1)
        try:
            assert harness.wakelab.submit(pcm(0x70, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await harness.pipeline.journal.wait_for(linklab.InputAudioEvent)
            assert harness.wakelab.submit(pcm(0x71, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            session = harness.pipeline.sessions[0]
            async with asyncio.timeout(1):
                while session._events.snapshots()[0].occupancy != 1:
                    await asyncio.sleep(0)
            assert harness.wakelab.submit(pcm(0x72, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.INPUT_OVERFLOW,
            )
            waiting = await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            assert harness.callbacks.journal.events.index(error) < harness.callbacks.journal.events.index(waiting)
            gate.set()

            start = len(harness.callbacks.journal.events)
            assert harness.wakelab.submit(pcm(0x73, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(recovered.done.wait(), 1)
            await harness.callbacks.journal.wait_for(
                linklab.StateEvent,
                after=start,
                matching=lambda event: event.state is linklab.CoarseState.WAITING,
            )
            first_input = next(iter(harness.pipeline.inputs))
            assert bytes(harness.pipeline.inputs[first_input].audio) == pcm(0x70, 1)
        finally:
            gate.set()
            await harness.close()

    run(scenario())


def test_fake_pipeline_playback_queue_overflow_maps_to_playback_failure() -> None:
    async def scenario() -> None:
        callback_gate = asyncio.Event()
        producer_gate = asyncio.Event()
        response_gate = asyncio.Event()
        plan = TurnPlan(
            1,
            "overflow",
            output_chunks=(pcm(0x80, 1), pcm(0x81, 10), pcm(0x82, 10)),
            output_gate=producer_gate,
            response_gate=response_gate,
        )
        harness = await open_harness([plan], client_playback_queue_ms=1)
        harness.callbacks.output_gate = callback_gate
        try:
            assert harness.wakelab.submit(pcm(0x74, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(harness.callbacks.output_entered.wait(), 1)
            producer_gate.set()
            interrupted = await harness.pipeline.journal.wait_for(
                linklab.PlaybackInterruptedEvent,
                matching=lambda event: event.reason is linklab.PlaybackInterruptReason.OVERFLOW,
            )
            assert interrupted.played_frames == 0
            callback_gate.set()
            error = await harness.callbacks.journal.wait_for(
                linklab.ErrorEvent,
                matching=lambda event: event.code is linklab.ErrorCode.PLAYBACK_FAILED,
            )
            cancelled = await harness.callbacks.journal.wait_for(linklab.ResponseCancelledEvent)
            ended = await harness.callbacks.journal.wait_for(linklab.ConversationEndedEvent)
            assert harness.callbacks.journal.events.index(error) < harness.callbacks.journal.events.index(cancelled)
            assert harness.callbacks.journal.events.index(cancelled) < harness.callbacks.journal.events.index(ended)
            assert ended.reason is linklab.ConversationEndReason.PLAYBACK_FAILED
        finally:
            producer_gate.set()
            response_gate.set()
            callback_gate.set()
            await harness.close()

    run(scenario())


def test_fake_pipeline_reconnect_discards_old_work_and_completes_fresh_conversation() -> None:
    async def scenario() -> None:
        output_gate = asyncio.Event()
        stale = TurnPlan(1, "old", output_chunks=(pcm(0x90, 1), pcm(0x91, 1)), output_gate=output_gate)
        fresh = TurnPlan(1, "fresh", text_deltas=("new connection",), end_conversation=True)
        harness = await open_harness([stale, fresh], reconnect=True)
        try:
            assert harness.wakelab.submit(pcm(0x75, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await harness.callbacks.journal.wait_for(linklab.OutputAudioEvent)
            first_session = harness.pipeline.sessions[0]
            before_loss = len(harness.callbacks.journal.events)
            first_session._core._request_close(1001, drain=False)
            await harness.callbacks.journal.wait_for(
                linklab.ConnectionStateEvent,
                after=before_loss,
                matching=lambda event: event.state is linklab.ConnectionState.DISCONNECTED,
            )
            await harness.callbacks.journal.wait_for(
                linklab.ConnectionStateEvent,
                after=before_loss,
                matching=lambda event: event.state is linklab.ConnectionState.READY,
            )
            assert harness.client.connection_state is linklab.ConnectionState.READY
            assert harness.wakelab.submit(pcm(0x76, 1), speech=True, activated=True) is (
                linklab.AudioSubmitResult.ACCEPTED
            )
            await asyncio.wait_for(fresh.done.wait(), 1)
            ended = await harness.callbacks.journal.wait_for(
                linklab.ConversationEndedEvent,
                after=before_loss,
            )
            output_gate.set()
            await asyncio.wait_for(stale.done.wait(), 1)

            assert ended.reason is linklab.ConversationEndReason.COMPLETED
            assert stale.writer_closed
            assert len(harness.pipeline.sessions) == 2
            assert [event.text for event in harness.callbacks.journal.of_type(linklab.ResponseTextDeltaEvent)] == [
                "new connection"
            ]
        finally:
            output_gate.set()
            await harness.close()

    run(scenario())


@pytest.mark.parametrize("close_code", [1002, 1008])
def test_fake_pipeline_does_not_reconnect_after_protocol_or_policy_close(close_code: int) -> None:
    async def scenario() -> None:
        harness = await open_harness([], reconnect=True)
        try:
            core = harness.client._core
            assert core is not None
            core._request_close(close_code, drain=False)
            await asyncio.wait_for(harness.client.wait_closed(), 1)
            assert harness.client.connection_state is linklab.ConnectionState.DISCONNECTED
            assert not harness.pipeline.sessions
        finally:
            await harness.close()

    run(scenario())
