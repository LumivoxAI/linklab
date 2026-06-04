import socket
import asyncio
from typing import Any, cast
from dataclasses import dataclass
from collections.abc import Coroutine

import pytest

import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


class Callbacks:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def _record(self, event: object) -> None:
        self.events.append(event)

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
        await self._record(event)

    async def on_output_audio(self, event: linklab.OutputAudioEvent) -> None:
        await self._record(event)

    async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None:
        await self._record(event)

    async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None:
        await self._record(event)

    async def on_error(self, event: linklab.ErrorEvent) -> None:
        await self._record(event)


class Handler:
    def __init__(self) -> None:
        self.playback: list[linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent] = []
        self.input_ready = asyncio.Event()
        self.later_input_started = asyncio.Event()

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        _event: linklab.ConversationStartedEvent,
    ) -> None:
        pass

    async def on_input_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.InputStartedEvent,
    ) -> None:
        if event.input_id != linklab.InputId(1):
            self.later_input_started.set()

    async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
        await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
        await session.finalize_transcript(event.input_id, "speech")
        self.input_ready.set()

    async def on_input_aborted(
        self,
        _session: linklab.ServerSession,
        _event: linklab.InputAbortedEvent,
    ) -> None:
        pass

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        self.playback.append(event)

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        _event: linklab.ConversationCancelledEvent,
    ) -> None:
        pass


@dataclass(slots=True)
class Harness:
    server: linklab.VoiceServer
    client: linklab.VoiceClient
    session: linklab.ServerSession
    handler: Handler
    callbacks: Callbacks

    async def close(self) -> None:
        await self.client.close()
        await self.server.close()


async def open_harness() -> Harness:
    sessions: list[linklab.ServerSession] = []
    handler = Handler()
    callbacks = Callbacks()
    port = unused_port()

    def factory(session: linklab.ServerSession) -> Handler:
        sessions.append(session)
        return handler

    server = linklab.VoiceServer(linklab.ServerConfig(port, (PCM_16K,)), factory, object())
    await server.serve()
    client = linklab.VoiceClient(
        linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,)),
        callbacks,
        object(),
    )
    await client.connect()
    assert client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x01\x02", 0, False, True, True)) is (
        linklab.AudioSubmitResult.ACCEPTED
    )
    await asyncio.wait_for(handler.input_ready.wait(), 1)
    return Harness(server, client, sessions[0], handler, callbacks)


async def wait_for_event(events: list[object], event_type: type[object]) -> object:
    async with asyncio.timeout(1):
        while True:
            for event in events:
                if isinstance(event, event_type):
                    return event
            await asyncio.sleep(0)


async def wait_for_state(events: list[object], state: linklab.CoarseState) -> linklab.StateEvent:
    async with asyncio.timeout(1):
        while True:
            for event in events:
                if isinstance(event, linklab.StateEvent) and event.state is state:
                    return event
            await asyncio.sleep(0)


async def wait_for_playback(handler: Handler, count: int = 1) -> None:
    async with asyncio.timeout(1):
        while len(handler.playback) < count:
            await asyncio.sleep(0)


async def start_output(
    harness: Harness,
    *,
    end_conversation: bool = False,
) -> tuple[linklab.ResponseWriter, linklab.OutputWriter]:
    response = await harness.session.start_response(linklab.InputId(1), end_conversation=end_conversation)
    output = await response.start_output()
    await output.send_audio(b"\x03\x04" * 2)
    await wait_for_event(harness.callbacks.events, linklab.OutputAudioEvent)
    return response, output


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("end_conversation", [False, True])
def test_public_client_finished_completes_response_at_either_arrival_order(
    early: bool,
    end_conversation: bool,
) -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response, output = await start_output(harness, end_conversation=end_conversation)
        await output.finish()
        await wait_for_event(harness.callbacks.events, linklab.OutputEndedEvent)

        if early:
            assert harness.client.playback_finished(output.output_id, 2)
            await wait_for_playback(harness.handler)
            await response.finish()
        else:
            await response.finish()
            assert harness.client.playback_finished(output.output_id, 2)
        if end_conversation:
            terminal = await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent)
            assert cast(linklab.ConversationEndedEvent, terminal).reason is linklab.ConversationEndReason.COMPLETED
        else:
            await wait_for_state(harness.callbacks.events, linklab.CoarseState.WAITING)
        await wait_for_playback(harness.handler)
        with pytest.raises(linklab.WriterClosed):
            await response.send_text_delta("late")
        await harness.close()

    run(scenario())


@pytest.mark.parametrize(
    ("reason", "end_conversation", "expected_end", "expected_error"),
    [
        (linklab.PlaybackInterruptReason.LOCAL_CANCEL, False, None, None),
        (
            linklab.PlaybackInterruptReason.LOCAL_CANCEL,
            True,
            linklab.ConversationEndReason.CANCELLED,
            None,
        ),
        (
            linklab.PlaybackInterruptReason.SHUTDOWN,
            False,
            linklab.ConversationEndReason.CANCELLED,
            None,
        ),
        (
            linklab.PlaybackInterruptReason.PLAYBACK_FAILED,
            False,
            linklab.ConversationEndReason.PLAYBACK_FAILED,
            linklab.ErrorCode.PLAYBACK_FAILED,
        ),
        (
            linklab.PlaybackInterruptReason.OVERFLOW,
            True,
            linklab.ConversationEndReason.PLAYBACK_FAILED,
            linklab.ErrorCode.PLAYBACK_FAILED,
        ),
    ],
)
def test_public_client_interruption_applies_response_and_conversation_consequences(
    reason: linklab.PlaybackInterruptReason,
    end_conversation: bool,
    expected_end: linklab.ConversationEndReason | None,
    expected_error: linklab.ErrorCode | None,
) -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response, output = await start_output(harness, end_conversation=end_conversation)
        assert harness.client.playback_interrupted(
            output.output_id,
            1,
            linklab.PlaybackPosition.EXACT,
            reason,
        )
        await wait_for_playback(harness.handler)
        await wait_for_event(harness.callbacks.events, linklab.ResponseCancelledEvent)
        with pytest.raises(linklab.WriterClosed):
            await output.finish()

        errors = [event for event in harness.callbacks.events if isinstance(event, linklab.ErrorEvent)]
        if expected_error is None:
            assert not errors
        else:
            error = await wait_for_event(harness.callbacks.events, linklab.ErrorEvent)
            assert cast(linklab.ErrorEvent, error).code is expected_error
        if expected_end is None:
            state = await wait_for_state(harness.callbacks.events, linklab.CoarseState.WAITING)
            assert state.conversation_id == linklab.ConversationId(1)
        else:
            ended = cast(
                linklab.ConversationEndedEvent,
                await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent),
            )
            assert ended.reason is expected_end
        await harness.close()

    run(scenario())


def test_barge_in_revokes_terminal_intent_and_rejects_stale_producer() -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response, output = await start_output(harness, end_conversation=True)
        assert (
            harness.client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x05\x06", 0, False, True, True))
            is linklab.AudioSubmitResult.ACCEPTED
        )
        assert harness.client.playback_interrupted(
            output.output_id,
            0,
            linklab.PlaybackPosition.EXACT,
            linklab.PlaybackInterruptReason.BARGE_IN,
        )
        await asyncio.wait_for(harness.handler.later_input_started.wait(), 1)
        await wait_for_playback(harness.handler)
        with pytest.raises(linklab.WriterClosed):
            await response.finish()
        assert not any(isinstance(event, linklab.ConversationEndedEvent) for event in harness.callbacks.events)
        assert any(
            isinstance(event, linklab.StateEvent) and event.state is linklab.CoarseState.LISTENING
            for event in harness.callbacks.events
        )
        await harness.close()

    run(scenario())


def test_awaiting_playback_cancellation_allows_one_late_accounting_event() -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response, output = await start_output(harness)
        await output.finish()
        await response.finish()
        await harness.session.end_conversation(linklab.ConversationEndReason.CANCELLED)
        await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent)

        assert harness.client.playback_interrupted(
            output.output_id,
            1,
            linklab.PlaybackPosition.ESTIMATED,
            linklab.PlaybackInterruptReason.LOCAL_CANCEL,
        )
        await wait_for_playback(harness.handler)
        assert not harness.client.playback_finished(output.output_id, 2)
        await harness.close()

    run(scenario())


@pytest.mark.parametrize("end_conversation", [False, True])
@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (linklab.ResponseCancelReason.GENERATION_FAILED, linklab.ErrorCode.GENERATION_FAILED),
        (linklab.ResponseCancelReason.TTS_FAILED, linklab.ErrorCode.TTS_FAILED),
        (linklab.ResponseCancelReason.OVERFLOW, linklab.ErrorCode.OUTPUT_OVERFLOW),
    ],
)
def test_response_failure_matrix_covers_both_terminal_intents(
    reason: linklab.ResponseCancelReason,
    code: linklab.ErrorCode,
    end_conversation: bool,
) -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response = await harness.session.start_response(linklab.InputId(1), end_conversation=end_conversation)
        output = await response.start_output() if reason is linklab.ResponseCancelReason.OVERFLOW else None
        await response.cancel(reason)
        error = cast(linklab.ErrorEvent, await wait_for_event(harness.callbacks.events, linklab.ErrorEvent))
        assert error.code is code
        await wait_for_event(harness.callbacks.events, linklab.ResponseCancelledEvent)
        if end_conversation:
            ended = cast(
                linklab.ConversationEndedEvent,
                await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent),
            )
            assert ended.reason is linklab.ConversationEndReason.SERVER_FAILED
        else:
            await wait_for_state(harness.callbacks.events, linklab.CoarseState.WAITING)
        with pytest.raises(linklab.WriterClosed):
            if output is None:
                await response.finish()
            else:
                await output.finish()
        await harness.close()

    run(scenario())


def test_peer_cancel_with_active_output_orders_child_before_conversation_end() -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response, output = await start_output(harness)
        assert harness.client.cancel_conversation(linklab.ConversationCancelReason.USER)
        await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent)
        terminal = [
            event
            for event in harness.callbacks.events
            if isinstance(event, (linklab.ResponseCancelledEvent, linklab.ConversationEndedEvent))
        ]
        assert [type(event) for event in terminal] == [
            linklab.ResponseCancelledEvent,
            linklab.ConversationEndedEvent,
        ]
        with pytest.raises(linklab.WriterClosed):
            await output.send_audio(b"\x00\x00")
        assert harness.session._core.snapshots()[0].occupancy == 0
        await harness.close()

    run(scenario())


@pytest.mark.parametrize("failure_first", [False, True])
def test_generation_failure_and_conversation_cancel_have_one_winning_response_terminal(
    failure_first: bool,
) -> None:
    async def scenario() -> None:
        harness = await open_harness()
        response = await harness.session.start_response(linklab.InputId(1))
        await wait_for_event(harness.callbacks.events, linklab.ResponseStartedEvent)

        if failure_first:
            await response.cancel(linklab.ResponseCancelReason.GENERATION_FAILED)
            await wait_for_event(harness.callbacks.events, linklab.ResponseCancelledEvent)
            assert harness.client.cancel_conversation(linklab.ConversationCancelReason.USER)
        else:
            assert harness.client.cancel_conversation(linklab.ConversationCancelReason.USER)
            await wait_for_event(harness.callbacks.events, linklab.ResponseCancelledEvent)
            with pytest.raises(linklab.WriterClosed):
                await response.cancel(linklab.ResponseCancelReason.GENERATION_FAILED)

        await wait_for_event(harness.callbacks.events, linklab.ConversationEndedEvent)
        terminals = [event for event in harness.callbacks.events if isinstance(event, linklab.ResponseCancelledEvent)]
        assert len(terminals) == 1
        expected = (
            linklab.ResponseCancelReason.GENERATION_FAILED
            if failure_first
            else linklab.ResponseCancelReason.CONVERSATION_CANCELLED
        )
        assert terminals[0].reason is expected
        await harness.close()

    run(scenario())
