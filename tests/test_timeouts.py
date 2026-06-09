import socket
import asyncio
from typing import Any, cast
from dataclasses import dataclass
from collections.abc import Coroutine

import pytest

import lumivox_linklab as linklab
from tests.helpers import NULL_LOGGER
from lumivox_linklab._handshake import _open_client_websocket

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


class Handler:
    def __init__(self) -> None:
        self.events: list[linklab.Message] = []

    async def on_conversation_started(
        self, _session: linklab.ServerSession, event: linklab.ConversationStartedEvent
    ) -> None:
        self.events.append(event)

    async def on_input_started(self, _session: linklab.ServerSession, event: linklab.InputStartedEvent) -> None:
        self.events.append(event)

    async def on_input_audio(self, _session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
        self.events.append(event)

    async def on_input_aborted(self, _session: linklab.ServerSession, event: linklab.InputAbortedEvent) -> None:
        self.events.append(event)

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        self.events.append(event)

    async def on_conversation_cancelled(
        self, _session: linklab.ServerSession, event: linklab.ConversationCancelledEvent
    ) -> None:
        self.events.append(event)


@dataclass(slots=True)
class Harness:
    server: linklab.VoiceServer
    client: Any
    session: linklab.ServerSession
    handler: Handler
    limits: linklab.ConnectionLimits

    async def close(self) -> None:
        await self.client.connection.close()
        await self.server.close()

    async def send(self, message: linklab.Message) -> None:
        await self.client.connection.send(linklab.encode_message(message))

    async def receive(self, count: int) -> list[linklab.Message]:
        messages: list[linklab.Message] = []
        async with asyncio.timeout(1):
            for _ in range(count):
                frame = await self.client.connection.recv()
                assert type(frame) is bytes
                messages.append(
                    linklab.decode_message(
                        frame,
                        direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                        limits=self.limits,
                    )
                )
        return messages


async def open_harness(
    *,
    limits: linklab.ConnectionLimits | None = None,
    waiting_timeout_s: float = 60,
    input_timeout_s: float = 120,
    processing_timeout_s: float = 120,
) -> Harness:
    sessions: list[linklab.ServerSession] = []
    handler = Handler()
    port = unused_port()
    configured_limits = limits or linklab.ConnectionLimits()
    config = linklab.ServerConfig(
        port,
        (PCM_16K,),
        limits=configured_limits,
        waiting_timeout_s=waiting_timeout_s,
        input_timeout_s=input_timeout_s,
        processing_timeout_s=processing_timeout_s,
    )

    def factory(session: linklab.ServerSession) -> Handler:
        sessions.append(session)
        return handler

    server = linklab.VoiceServer(config, factory, NULL_LOGGER)
    await server.serve()
    client = await _open_client_websocket(linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,)))
    async with asyncio.timeout(1):
        while not sessions:
            await asyncio.sleep(0)
    return Harness(server, client, sessions[0], handler, client.server_hello.limits)


async def start_input(harness: Harness, *, with_audio: bool = True) -> None:
    conversation_id = linklab.ConversationId(1)
    input_id = linklab.InputId(1)
    await harness.send(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    await harness.send(linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0))
    if with_audio:
        await harness.send(linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0\0" * 2))
    expected = 3 if with_audio else 2
    async with asyncio.timeout(1):
        while len(harness.handler.events) < expected:
            await asyncio.sleep(0)


def fire_object_timeout(session: linklab.ServerSession, *, before: bool = False) -> None:
    object_deadline, _ = session._timeouts.deadlines
    assert object_deadline is not None
    session._timeouts.run_due(object_deadline - 0.001 if before else object_deadline)


def assert_phase(session: linklab.ServerSession, expected: str) -> None:
    phase = session._timeout_phase()
    assert phase is not None
    assert phase[0] == expected


def assert_active_deadlines(session: linklab.ServerSession, *, object_timer: bool, idle_timer: bool) -> None:
    object_deadline, idle_deadline = session._timeouts.deadlines
    assert (object_deadline is not None) is object_timer
    assert (idle_deadline is not None) is idle_timer


def test_elapsed_input_and_processing_boundaries_emit_complete_sequences() -> None:
    async def scenario() -> None:
        harness = await open_harness(input_timeout_s=7, processing_timeout_s=11)
        await start_input(harness)
        input_ = harness.session._core.validator._data.inputs[-1]

        fire_object_timeout(harness.session, before=True)
        assert input_.terminal is None
        fire_object_timeout(harness.session)
        assert harness.session._core.validator._data.inputs[-1].transcript_final is None
        assert await harness.receive(3) == [
            linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING),
            linklab.InputClosedEvent(
                linklab.ConversationId(1), linklab.InputId(1), 2, linklab.InputCloseReason.MAX_DURATION
            ),
            linklab.StateEvent(linklab.ConversationId(1), 2, linklab.CoarseState.PROCESSING),
        ]

        fire_object_timeout(harness.session)
        assert await harness.receive(3) == [
            linklab.ErrorEvent(
                linklab.ErrorScope.INPUT,
                linklab.ErrorCode.PROCESSING_TIMEOUT,
                True,
                linklab.ConversationId(1),
                linklab.InputId(1),
            ),
            linklab.TranscriptFinalEvent(linklab.ConversationId(1), linklab.InputId(1), ""),
            linklab.StateEvent(linklab.ConversationId(1), 3, linklab.CoarseState.WAITING),
        ]
        await harness.close()

    run(scenario())


@pytest.mark.parametrize("finalized", [False, True])
def test_processing_timeout_preserves_partial_or_application_final(finalized: bool) -> None:
    async def scenario() -> None:
        harness = await open_harness(processing_timeout_s=9)
        await start_input(harness, with_audio=False)
        session = harness.session
        if not finalized:
            await session.update_transcript(linklab.InputId(1), 1, "partial")
        await session.close_input(linklab.InputId(1), linklab.InputCloseReason.ENDPOINT)
        if finalized:
            await session.finalize_transcript(linklab.InputId(1), "final")
        fire_object_timeout(session)

        input_ = session._core.validator._data.inputs[-1]
        assert (input_.transcript_update is not None) is (not finalized)
        assert input_.transcript_final is not None
        assert input_.transcript_final.text == ("final" if finalized else "")
        assert input_.failure is not None
        assert input_.failure.code is linklab.ErrorCode.PROCESSING_TIMEOUT
        assert_phase(session, "waiting")
        await harness.close()

    run(scenario())


@pytest.mark.parametrize("reason", list(linklab.InputCloseReason))
def test_processing_timeout_applies_after_every_input_close_reason(reason: linklab.InputCloseReason) -> None:
    async def scenario() -> None:
        harness = await open_harness(processing_timeout_s=13)
        await start_input(harness, with_audio=False)
        await harness.session.close_input(linklab.InputId(1), reason)
        fire_object_timeout(harness.session)
        input_ = harness.session._core.validator._data.inputs[-1]
        assert input_.failure is not None
        assert input_.transcript_final is not None and input_.transcript_final.text == ""
        await harness.close()

    run(scenario())


def test_processing_wins_simultaneous_idle_and_resets_idle_from_waiting() -> None:
    async def scenario() -> None:
        limits = linklab.ConnectionLimits(idle_timeout_ms=10_000)
        harness = await open_harness(limits=limits, waiting_timeout_s=20, processing_timeout_s=10)
        await start_input(harness, with_audio=False)
        await harness.session.close_input(linklab.InputId(1), linklab.InputCloseReason.NO_SPEECH)
        object_deadline, idle_deadline = harness.session._timeouts.deadlines
        assert object_deadline == idle_deadline

        harness.session._timeouts.run_due(cast(float, object_deadline))
        assert_phase(harness.session, "waiting")
        assert harness.session._core.validator._data.conversation is not None
        new_object, new_idle = harness.session._timeouts.deadlines
        assert new_object is not None and new_idle is not None
        assert new_idle > cast(float, idle_deadline)
        await harness.close()

    run(scenario())


def test_idle_uses_negotiated_limit_and_lifecycle_only_while_active_work_suppresses_it() -> None:
    async def scenario() -> None:
        limits = linklab.ConnectionLimits(idle_timeout_ms=10_000)
        idle_harness = await open_harness(limits=limits, waiting_timeout_s=30)
        await idle_harness.send(linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word"))
        async with asyncio.timeout(1):
            while not idle_harness.handler.events:
                await asyncio.sleep(0)
        first_deadline = idle_harness.session._timeouts.deadlines[1]
        assert first_deadline is not None
        idle_harness.session._timeouts.sync()  # Ping/Pong and other non-lifecycle activity cannot reset idle.
        assert idle_harness.session._timeouts.deadlines[1] == first_deadline
        idle_harness.session._timeouts.run_due(first_deadline)
        assert await idle_harness.receive(2) == [
            linklab.ErrorEvent(
                linklab.ErrorScope.CONVERSATION,
                linklab.ErrorCode.IDLE_TIMEOUT,
                True,
                linklab.ConversationId(1),
            ),
            linklab.ConversationEndedEvent(linklab.ConversationId(1), linklab.ConversationEndReason.IDLE_TIMEOUT),
        ]
        await idle_harness.close()

        active_harness = await open_harness(limits=limits, waiting_timeout_s=30)
        await active_harness.send(linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word"))
        await active_harness.send(
            linklab.InputStartedEvent(
                linklab.ConversationId(1), linklab.InputId(1), linklab.InputStartReason.ACTIVATION, 0
            )
        )
        async with asyncio.timeout(1):
            while len(active_harness.handler.events) < 2:
                await asyncio.sleep(0)
        assert_active_deadlines(active_harness.session, object_timer=True, idle_timer=False)
        await active_harness.session.close_input(linklab.InputId(1), linklab.InputCloseReason.NO_SPEECH)
        await active_harness.session.finalize_transcript(linklab.InputId(1), "")
        response = await active_harness.session.start_response(linklab.InputId(1))
        assert active_harness.session._timeouts.deadlines == (None, None)
        await response.finish()
        assert all(deadline is not None for deadline in active_harness.session._timeouts.deadlines)
        await active_harness.close()

    run(scenario())


def test_waiting_timeout_ends_conversation_and_stale_timer_cannot_publish_again() -> None:
    async def scenario() -> None:
        harness = await open_harness(waiting_timeout_s=3)
        await start_input(harness, with_audio=False)
        await harness.session.close_input(linklab.InputId(1), linklab.InputCloseReason.NO_SPEECH)
        await harness.session.finalize_transcript(linklab.InputId(1), "")
        response = await harness.session.start_response(linklab.InputId(1))
        await response.finish()
        object_deadline, _ = harness.session._timeouts.deadlines
        assert object_deadline is not None
        harness.session._timeouts.run_due(object_deadline)
        assert harness.session._core.validator._data.conversation is None
        before = harness.session._core.validator._data
        harness.session._timeouts.run_due(object_deadline + 100)
        assert harness.session._core.validator._data == before
        await harness.close()

    run(scenario())


@pytest.mark.parametrize("crosses", [False, True])
def test_aggregate_frame_limit_closes_exact_boundary_or_rejects_crossing_chunk(crosses: bool) -> None:
    async def scenario() -> None:
        limits = linklab.ConnectionLimits(max_input_frames=16_000, max_output_audio_frames=1_600)
        harness = await open_harness(limits=limits)
        await start_input(harness, with_audio=False)
        sent = 0
        while sent < 15_999:
            frames = min(1_600, 15_999 - sent)
            await harness.send(
                linklab.InputAudioEvent(linklab.ConversationId(1), linklab.InputId(1), sent, True, b"\0\0" * frames)
            )
            sent += frames
        async with asyncio.timeout(1):
            while (
                sum(
                    len(event.audio) // 2
                    for event in harness.handler.events
                    if isinstance(event, linklab.InputAudioEvent)
                )
                < sent
            ):
                await asyncio.sleep(0)

        final_frames = 2 if crosses else 1
        await harness.send(
            linklab.InputAudioEvent(linklab.ConversationId(1), linklab.InputId(1), sent, True, b"\0\0" * final_frames)
        )
        async with asyncio.timeout(1):
            while harness.session._core.validator._data.inputs[-1].terminal is None:
                await asyncio.sleep(0)
        input_ = harness.session._core.validator._data.inputs[-1]
        if crosses:
            assert input_.committed_end_frame == 15_999
            assert input_.received_end_frame == 15_999
            assert input_.failure is not None and input_.failure.code is linklab.ErrorCode.INPUT_TOO_LONG
            assert input_.transcript_final is not None and input_.transcript_final.text == ""
            assert (
                sum(
                    len(event.audio) // 2
                    for event in harness.handler.events
                    if isinstance(event, linklab.InputAudioEvent)
                )
                == 15_999
            )
            assert_phase(harness.session, "waiting")
        else:
            assert input_.committed_end_frame == 16_000
            assert isinstance(input_.terminal, linklab.InputClosedEvent)
            assert input_.terminal.reason is linklab.InputCloseReason.MAX_DURATION
            assert input_.transcript_final is None
            assert_phase(harness.session, "processing")
        await harness.close()

    run(scenario())
