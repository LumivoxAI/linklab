import socket
import asyncio
from typing import Any, cast
from dataclasses import replace
from collections.abc import Coroutine

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._protocol import _IdAllocator
from lumivox_linklab._handshake import _open_client_websocket
from lumivox_linklab._transport import _QueueLane

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


class Handler:
    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        _event: linklab.ConversationStartedEvent,
    ) -> None:
        pass

    async def on_input_started(self, _session: linklab.ServerSession, _event: linklab.InputStartedEvent) -> None:
        pass

    async def on_input_audio(self, _session: linklab.ServerSession, _event: linklab.InputAudioEvent) -> None:
        pass

    async def on_input_aborted(self, _session: linklab.ServerSession, _event: linklab.InputAbortedEvent) -> None:
        pass

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        _event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        pass

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        _event: linklab.ConversationCancelledEvent,
    ) -> None:
        pass


async def open_ready_session(
    *, output_queue_ms: int = 500
) -> tuple[Any, Any, linklab.ServerSession, linklab.ConversationId, linklab.InputId]:
    port = unused_port()
    config = linklab.ServerConfig(port=port, output_formats=(PCM_16K,), output_queue_ms=output_queue_ms)
    sessions: list[linklab.ServerSession] = []

    def factory(session: linklab.ServerSession) -> Handler:
        sessions.append(session)
        return Handler()

    server = linklab.VoiceServer(config, factory, object())
    await server.serve()
    client = await _open_client_websocket(linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,)))
    async with asyncio.timeout(1):
        while not sessions:
            await asyncio.sleep(0)
    session = sessions[0]
    conversation_id = linklab.ConversationId(1)
    input_id = linklab.InputId(1)
    await client.connection.send(linklab.encode_message(linklab.ConversationStartedEvent(conversation_id, "wake_word")))
    await client.connection.send(
        linklab.encode_message(
            linklab.InputStartedEvent(
                conversation_id,
                input_id,
                linklab.InputStartReason.ACTIVATION,
                0,
            )
        )
    )
    async with asyncio.timeout(1):
        while session._core.validator.state.input_id != input_id:
            await asyncio.sleep(0)
    await session.close_input(input_id, linklab.InputCloseReason.NO_SPEECH)
    await session.finalize_transcript(input_id, "")
    await receive_messages(client, 4)
    return server, client, session, conversation_id, input_id


async def receive_messages(client: Any, count: int) -> list[linklab.Message]:
    messages: list[linklab.Message] = []
    async with asyncio.timeout(1):
        for _ in range(count):
            frame = await client.connection.recv()
            assert type(frame) is bytes
            messages.append(
                linklab.decode_message(
                    frame,
                    direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                    limits=client.server_hello.limits,
                )
            )
    return messages


async def close_pair(server: Any, client: Any) -> None:
    await client.connection.close()
    await server.close()


def test_server_shutdown_cancels_response_and_invalidates_output_writer() -> None:
    async def scenario() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        output = await response.start_output()
        await output.send_audio(b"\x01\x02")
        await receive_messages(client, 4)

        closing = asyncio.create_task(server.close())
        assert await receive_messages(client, 2) == [
            linklab.ResponseCancelledEvent(
                conversation_id,
                linklab.ResponseId(1),
                linklab.ResponseCancelReason.SHUTDOWN,
            ),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.CANCELLED),
        ]
        with pytest.raises(linklab.WriterClosed):
            await output.send_audio(b"\x03\x04")
        with pytest.raises(linklab.WriterClosed):
            await response.finish()
        await closing
        await client.connection.wait_closed()
        assert client.connection.close_code == 1001

    run(scenario())


def test_response_writer_sequences_text_and_finishes_without_output() -> None:
    async def scenario() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        writer = await session.start_response(input_id)
        assert writer.response_id == linklab.ResponseId(1)
        with pytest.raises(AttributeError):
            writer.response_id = linklab.ResponseId(2)  # type: ignore[misc]
        with pytest.raises(linklab.ProtocolViolation, match="finalized terminal input"):
            await session.start_response(input_id)

        await writer.send_text_delta("hello ")
        await writer.send_text_delta("world")
        await writer.finalize_text("hello world")
        await writer.finish()
        assert await receive_messages(client, 7) == [
            linklab.ResponseStartedEvent(conversation_id, linklab.ResponseId(1), input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.ResponseTextDeltaEvent(conversation_id, linklab.ResponseId(1), 0, "hello "),
            linklab.ResponseTextDeltaEvent(conversation_id, linklab.ResponseId(1), 1, "world"),
            linklab.ResponseTextFinalEvent(conversation_id, linklab.ResponseId(1), "hello world"),
            linklab.ResponseEndedEvent(conversation_id, linklab.ResponseId(1)),
            linklab.StateEvent(conversation_id, 4, linklab.CoarseState.WAITING),
        ]
        with pytest.raises(linklab.WriterClosed):
            await writer.send_text_delta("late")
        await close_pair(server, client)

    run(scenario())


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        (linklab.ResponseCancelReason.GENERATION_FAILED, linklab.ErrorCode.GENERATION_FAILED),
        (linklab.ResponseCancelReason.TTS_FAILED, linklab.ErrorCode.TTS_FAILED),
    ],
)
def test_failure_cancel_emits_exact_batch(reason: linklab.ResponseCancelReason, code: linklab.ErrorCode) -> None:
    async def scenario() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        writer = await session.start_response(input_id)
        await writer.cancel(reason)
        response_id = linklab.ResponseId(1)
        assert await receive_messages(client, 5) == [
            linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.ErrorEvent(linklab.ErrorScope.RESPONSE, code, True, conversation_id, response_id=response_id),
            linklab.ResponseCancelledEvent(conversation_id, response_id, reason),
            linklab.StateEvent(conversation_id, 4, linklab.CoarseState.WAITING),
        ]
        with pytest.raises(linklab.WriterClosed):
            await writer.finish()
        await close_pair(server, client)

    run(scenario())


def test_terminal_intent_and_context_manager_outcomes() -> None:
    async def completed() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        async with await session.start_response(input_id, end_conversation=True) as writer:
            assert writer.response_id == linklab.ResponseId(1)
        assert await receive_messages(client, 4) == [
            linklab.ResponseStartedEvent(conversation_id, linklab.ResponseId(1), input_id, True),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.ResponseEndedEvent(conversation_id, linklab.ResponseId(1)),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.COMPLETED),
        ]
        await close_pair(server, client)

    async def failed() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        with pytest.raises(LookupError, match="generation crashed"):
            async with await session.start_response(input_id, end_conversation=True):
                raise LookupError("generation crashed")
        response_id = linklab.ResponseId(1)
        assert await receive_messages(client, 5) == [
            linklab.ResponseStartedEvent(conversation_id, response_id, input_id, True),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.ErrorEvent(
                linklab.ErrorScope.RESPONSE,
                linklab.ErrorCode.GENERATION_FAILED,
                True,
                conversation_id,
                response_id=response_id,
            ),
            linklab.ResponseCancelledEvent(
                conversation_id,
                response_id,
                linklab.ResponseCancelReason.GENERATION_FAILED,
            ),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.SERVER_FAILED),
        ]
        await close_pair(server, client)

    run(completed())
    run(failed())


def test_invalid_cancel_id_exhaustion_and_external_invalidation_are_atomic() -> None:
    async def invalid_cancel() -> None:
        server, client, session, _, input_id = await open_ready_session()
        writer = await session.start_response(input_id)
        before = session._core.validator._data
        with pytest.raises(linklab.ProtocolViolation, match="peer or session"):
            await writer.cancel(linklab.ResponseCancelReason.BARGE_IN)
        with pytest.raises(linklab.ProtocolViolation, match="started output"):
            await writer.cancel(linklab.ResponseCancelReason.OVERFLOW)
        assert session._core.validator._data == before
        await writer.cancel(linklab.ResponseCancelReason.SHUTDOWN)
        await receive_messages(client, 4)
        await close_pair(server, client)

    async def exhausted() -> None:
        server, client, session, _, input_id = await open_ready_session()
        data = session._core.validator._data
        session._core.validator._data = replace(data, response_ids=_IdAllocator(4_294_967_296))
        before = session._core.validator._data
        with pytest.raises(linklab.ProtocolViolation, match="id_exhausted"):
            await session.start_response(input_id)
        assert session._core.validator._data == before
        await close_pair(server, client)

    async def externally_ended() -> None:
        server, client, session, _, input_id = await open_ready_session()
        writer = await session.start_response(input_id)
        await session.end_conversation(linklab.ConversationEndReason.CANCELLED)
        with pytest.raises(linklab.WriterClosed):
            await writer.finalize_text("late")
        await receive_messages(client, 4)
        await close_pair(server, client)

    async def output_id_exhausted() -> None:
        server, client, session, _, input_id = await open_ready_session()
        writer = await session.start_response(input_id)
        data = session._core.validator._data
        session._core.validator._data = replace(data, output_ids=_IdAllocator(4_294_967_296))
        before = session._core.validator._data
        with pytest.raises(linklab.ProtocolViolation, match="id_exhausted"):
            await writer.start_output()
        assert session._core.validator._data == before
        await writer.cancel(linklab.ResponseCancelReason.SHUTDOWN)
        await receive_messages(client, 4)
        await close_pair(server, client)

    run(invalid_cancel())
    run(exhausted())
    run(externally_ended())
    run(output_id_exhausted())


def test_output_writer_copies_splits_and_finishes_before_response() -> None:
    async def scenario() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        output = await response.start_output()
        assert output.output_id == linklab.OutputId(1)
        with pytest.raises(AttributeError):
            output.output_id = linklab.OutputId(2)  # type: ignore[misc]

        pcm = bytearray(b"\x01\x02" * 1_601)
        await output.send_audio(pcm)
        pcm[:] = b"\xff" * len(pcm)

        initial = await receive_messages(client, 5)
        assert initial == [
            linklab.ResponseStartedEvent(conversation_id, linklab.ResponseId(1), input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.OutputStartedEvent(conversation_id, linklab.ResponseId(1), linklab.OutputId(1)),
            linklab.OutputAudioEvent(
                conversation_id,
                linklab.ResponseId(1),
                linklab.OutputId(1),
                0,
                b"\x01\x02" * 1_600,
            ),
            linklab.OutputAudioEvent(
                conversation_id,
                linklab.ResponseId(1),
                linklab.OutputId(1),
                1_600,
                b"\x01\x02",
            ),
        ]

        with pytest.raises(linklab.ProtocolViolation, match="before its output"):
            await response.finish()
        await output.finish()
        await response.finish()
        assert await receive_messages(client, 2) == [
            linklab.OutputEndedEvent(
                conversation_id,
                linklab.ResponseId(1),
                linklab.OutputId(1),
                1_601,
            ),
            linklab.ResponseEndedEvent(conversation_id, linklab.ResponseId(1)),
        ]
        with pytest.raises(linklab.WriterClosed):
            await output.send_audio(b"\x00\x00")
        await close_pair(server, client)

    run(scenario())


def test_output_writer_rejects_invalid_and_empty_audio_without_mutation() -> None:
    async def scenario() -> None:
        server, client, session, _, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        output = await response.start_output()
        before = session._core.validator._data

        with pytest.raises(ValueError, match="must not be empty"):
            await output.send_audio(b"")
        with pytest.raises(ValueError, match="frame-aligned"):
            await output.send_audio(b"\x00")
        source = memoryview(bytearray(8))[::2]
        with pytest.raises(ValueError, match="contiguous"):
            await output.send_audio(source)
        with pytest.raises(linklab.ProtocolViolation, match="at least one"):
            await output.finish()
        assert session._core.validator._data == before

        await response.cancel(linklab.ResponseCancelReason.TTS_FAILED)
        await receive_messages(client, 6)
        await close_pair(server, client)

    run(scenario())


def test_output_backpressure_rejects_whole_chunk_and_commits_terminal_batch() -> None:
    async def scenario() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session(output_queue_ms=10)
        response = await session.start_response(input_id)
        output = await response.start_output()
        await output.send_audio(b"\x00\x00" * 160)
        data_snapshot, _ = session._core.snapshots()
        assert (data_snapshot.capacity, data_snapshot.occupancy, data_snapshot.occupancy_unit) == (160, 160, "frames")

        with pytest.raises(linklab.QueueOverflow):
            await output.send_audio(b"\x01\x02")
        data_snapshot, _ = session._core.snapshots()
        assert data_snapshot.occupancy == 0
        assert data_snapshot.overflow_count == 1
        with pytest.raises(linklab.WriterClosed):
            await output.finish()
        with pytest.raises(linklab.WriterClosed):
            await response.finish()

        response_id = linklab.ResponseId(1)
        assert await receive_messages(client, 6) == [
            linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.OutputStartedEvent(conversation_id, response_id, linklab.OutputId(1)),
            linklab.ErrorEvent(
                linklab.ErrorScope.RESPONSE,
                linklab.ErrorCode.OUTPUT_OVERFLOW,
                True,
                conversation_id,
                response_id=response_id,
            ),
            linklab.ResponseCancelledEvent(conversation_id, response_id, linklab.ResponseCancelReason.OVERFLOW),
            linklab.StateEvent(conversation_id, 4, linklab.CoarseState.WAITING),
        ]
        await close_pair(server, client)

    run(scenario())


def test_server_reserved_control_exhaustion_closes_1011_without_partial_transition() -> None:
    async def scenario() -> None:
        server, client, session, _, input_id = await open_ready_session()
        core = session._core
        before = core.validator._data
        core._queue._occupancy[_QueueLane.CONTROL] = core._queue._capacities[_QueueLane.CONTROL]

        with pytest.raises(linklab.QueueOverflow):
            await session.start_response(input_id)
        await client.connection.wait_closed()

        assert client.connection.close_code == 1011
        assert core.validator._data.conversation == before.conversation
        assert core.snapshots()[1].overflow_count == 1
        await server.close()

    run(scenario())


def test_output_context_manager_normal_and_exceptional_exit() -> None:
    async def normal() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        async with await session.start_response(input_id) as response:
            async with await response.start_output() as output:
                await output.send_audio(b"\x03\x04")
        assert await receive_messages(client, 6) == [
            linklab.ResponseStartedEvent(conversation_id, linklab.ResponseId(1), input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.OutputStartedEvent(conversation_id, linklab.ResponseId(1), linklab.OutputId(1)),
            linklab.OutputAudioEvent(
                conversation_id,
                linklab.ResponseId(1),
                linklab.OutputId(1),
                0,
                b"\x03\x04",
            ),
            linklab.OutputEndedEvent(conversation_id, linklab.ResponseId(1), linklab.OutputId(1), 1),
            linklab.ResponseEndedEvent(conversation_id, linklab.ResponseId(1)),
        ]
        await close_pair(server, client)

    async def exceptional() -> None:
        server, client, session, conversation_id, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        with pytest.raises(LookupError, match="tts crashed"):
            async with await response.start_output():
                raise LookupError("tts crashed")
        response_id = linklab.ResponseId(1)
        assert await receive_messages(client, 6) == [
            linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            linklab.OutputStartedEvent(conversation_id, response_id, linklab.OutputId(1)),
            linklab.ErrorEvent(
                linklab.ErrorScope.RESPONSE,
                linklab.ErrorCode.TTS_FAILED,
                True,
                conversation_id,
                response_id=response_id,
            ),
            linklab.ResponseCancelledEvent(conversation_id, response_id, linklab.ResponseCancelReason.TTS_FAILED),
            linklab.StateEvent(conversation_id, 4, linklab.CoarseState.WAITING),
        ]
        with pytest.raises(linklab.WriterClosed):
            await response.send_text_delta("late")
        await close_pair(server, client)

    run(normal())
    run(exceptional())


def test_empty_normal_output_context_propagates_protocol_violation() -> None:
    async def scenario() -> None:
        server, client, session, _conversation_id, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        with pytest.raises(linklab.ProtocolViolation, match="at least one audio frame"):
            async with await response.start_output():
                pass
        await close_pair(server, client)

    run(scenario())


def test_output_writer_is_invalidated_by_manual_cancel_and_external_termination() -> None:
    async def manual_cancel() -> None:
        server, client, session, _, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        output = await response.start_output()
        await output.send_audio(b"\x00\x00")
        await response.cancel(linklab.ResponseCancelReason.TTS_FAILED)
        assert session._core.snapshots()[0].occupancy == 0
        with pytest.raises(linklab.WriterClosed):
            await output.send_audio(b"\x00\x00")
        await receive_messages(client, 6)
        await close_pair(server, client)

    async def externally_ended() -> None:
        server, client, session, _, input_id = await open_ready_session()
        response = await session.start_response(input_id)
        output = await response.start_output()
        await session.end_conversation(linklab.ConversationEndReason.CANCELLED)
        with pytest.raises(linklab.WriterClosed):
            await output.finish()
        await receive_messages(client, 5)
        await close_pair(server, client)

    run(manual_cancel())
    run(externally_ended())
