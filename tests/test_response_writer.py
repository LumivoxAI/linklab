import socket
import asyncio
from typing import Any, cast
from dataclasses import replace
from collections.abc import Coroutine

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._protocol import _IdAllocator
from lumivox_linklab._handshake import _open_client_websocket

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


async def open_ready_session() -> tuple[Any, Any, linklab.ServerSession, linklab.ConversationId, linklab.InputId]:
    port = unused_port()
    config = linklab.ServerConfig(port=port, output_formats=(PCM_16K,))
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

    run(invalid_cancel())
    run(exhausted())
    run(externally_ended())
