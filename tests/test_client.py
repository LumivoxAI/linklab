import socket
import asyncio
from typing import Any, cast
from collections.abc import Awaitable, Coroutine

import pytest
from websockets.asyncio.server import Server, ServerConnection

import lumivox_linklab as linklab
from lumivox_linklab._handshake import _ClientHandshake, _serve_websocket, _perform_server_handshake

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


class NullCallbacks:
    async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None:
        pass

    async def on_conversation_state(self, event: linklab.StateEvent) -> None:
        pass

    async def on_transcript_update(self, event: linklab.TranscriptUpdateEvent) -> None:
        pass

    async def on_transcript_final(self, event: linklab.TranscriptFinalEvent) -> None:
        pass

    async def on_response_started(self, event: linklab.ResponseStartedEvent) -> None:
        pass

    async def on_response_text_delta(self, event: linklab.ResponseTextDeltaEvent) -> None:
        pass

    async def on_response_text_final(self, event: linklab.ResponseTextFinalEvent) -> None:
        pass

    async def on_response_ended(self, event: linklab.ResponseEndedEvent) -> None:
        pass

    async def on_response_cancelled(self, event: linklab.ResponseCancelledEvent) -> None:
        pass

    async def on_output_started(self, event: linklab.OutputStartedEvent) -> None:
        pass

    async def on_output_audio(self, event: linklab.OutputAudioEvent) -> None:
        pass

    async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None:
        pass

    async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None:
        pass

    async def on_error(self, event: linklab.ErrorEvent) -> None:
        pass


class FakeConnection:
    def __init__(self, *, block_send: bool = False) -> None:
        self.inbound: asyncio.Queue[bytes | str] = asyncio.Queue()
        self.sent: list[bytes] = []
        self.close_codes: list[int] = []
        self.send_started = asyncio.Event()
        self.send_gate = asyncio.Event()
        if not block_send:
            self.send_gate.set()

    async def recv(self) -> bytes | str:
        return await self.inbound.get()

    async def send(self, data: bytes) -> None:
        self.send_started.set()
        await self.send_gate.wait()
        self.sent.append(data)

    async def ping(self) -> Awaitable[float]:
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        future.set_result(0.001)
        return future

    async def close(self, code: int) -> None:
        self.close_codes.append(code)


def fake_handshake(connection: FakeConnection) -> _ClientHandshake:
    limits = linklab.ConnectionLimits(max_output_audio_frames=1_600)
    client_hello = linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,))
    server_hello = linklab.ServerHello(1, CAPABILITIES, PCM_16K, limits)
    return _ClientHandshake(cast(Any, connection), client_hello, server_hello)


async def close_server(server: Server) -> None:
    server.close()
    await server.wait_closed()


def assert_disconnected(client: linklab.VoiceClient) -> None:
    assert client.connection_state is linklab.ConnectionState.DISCONNECTED
    assert client.output_format is None


def test_loopback_facade_handoffs_capture_thread_control_and_closes_idempotently() -> None:
    async def scenario() -> None:
        port = unused_port()
        server_config = linklab.ServerConfig(port, (PCM_16K,))
        received: list[linklab.Message] = []

        async def handler(connection: ServerConnection) -> None:
            handshake = await _perform_server_handshake(connection, server_config)
            assert handshake is not None
            while len(received) < 4:
                frame = await connection.recv()
                assert type(frame) is bytes
                received.append(
                    linklab.decode_message(
                        frame,
                        direction=linklab.MessageDirection.CLIENT_TO_SERVER,
                        limits=handshake.server_hello.limits,
                    )
                )
            await connection.wait_closed()

        server = await _serve_websocket(server_config, handler)
        client = linklab.VoiceClient(
            linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,)),
            NullCallbacks(),
            object(),
        )
        try:
            assert_disconnected(client)
            await client.connect()
            assert client.connection_state is linklab.ConnectionState.READY
            assert client.output_format == PCM_16K

            source = bytearray(b"\x01\x02" * 4)
            result = await asyncio.to_thread(
                client.submit_annotated_audio,
                linklab.AnnotatedAudio(source, 0, False, True, True),
            )
            source[:] = b"\x09\x09" * 4
            assert result is linklab.AudioSubmitResult.ACCEPTED
            assert await asyncio.to_thread(client.abort_input, linklab.InputAbortReason.CAPTURE_FAILED) is True
            assert await asyncio.to_thread(client.abort_input, linklab.InputAbortReason.CAPTURE_FAILED) is False

            async with asyncio.timeout(1):
                while len(received) < 4:
                    await asyncio.sleep(0)
            assert [message.type for message in received] == [
                "conversation.start",
                "input.start",
                "input.audio",
                "input.abort",
            ]
            audio = cast(linklab.InputAudioEvent, received[2])
            assert audio.audio == b"\x01\x02" * 4

            await asyncio.gather(client.close(), client.close(), client.close())
            assert_disconnected(client)
            await client.wait_closed()
        finally:
            await client.close()
            await close_server(server)

    run(scenario())


def test_context_manager_propagates_body_exception() -> None:
    async def scenario() -> None:
        port = unused_port()
        server_config = linklab.ServerConfig(port, (PCM_16K,))

        async def handler(connection: ServerConnection) -> None:
            assert await _perform_server_handshake(connection, server_config) is not None
            await connection.wait_closed()

        server = await _serve_websocket(server_config, handler)
        try:
            client = linklab.VoiceClient(
                linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,)),
                NullCallbacks(),
                object(),
            )
            try:
                async with client:
                    raise LookupError("body failed")
            except LookupError as error:
                assert str(error) == "body failed"
            else:
                raise AssertionError("context manager suppressed the body exception")
            assert client.connection_state is linklab.ConnectionState.DISCONNECTED
        finally:
            await close_server(server)

    run(scenario())


def test_input_close_discards_transport_queued_audio_but_not_writer_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        connection = FakeConnection(block_send=True)

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,), input_queue_frames=10),
            NullCallbacks(),
            object(),
        )
        await client.connect()
        assert (
            client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x01\x02", 0, False, True, True))
            is linklab.AudioSubmitResult.ACCEPTED
        )
        await connection.send_started.wait()
        assert (
            client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x03\x04", 0, False, True, True))
            is linklab.AudioSubmitResult.ACCEPTED
        )
        core = client._core
        assert core is not None
        async with asyncio.timeout(1):
            while core.snapshots()[0].occupancy != 1:
                await asyncio.sleep(0)

        closed = linklab.InputClosedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            1,
            linklab.InputCloseReason.ENDPOINT,
        )
        await connection.inbound.put(linklab.encode_message(closed))
        async with asyncio.timeout(1):
            while core.snapshots()[0].occupancy:
                await asyncio.sleep(0)

        connection.send_gate.set()
        async with asyncio.timeout(1):
            while len(connection.sent) != 3:
                await asyncio.sleep(0)
        decoded = [
            linklab.decode_message(
                frame,
                direction=linklab.MessageDirection.CLIENT_TO_SERVER,
                limits=linklab.ConnectionLimits(max_output_audio_frames=1_600),
            )
            for frame in connection.sent
        ]
        assert [message.type for message in decoded] == ["conversation.start", "input.start", "input.audio"]
        assert cast(linklab.InputAudioEvent, decoded[-1]).audio == b"\x01\x02"
        await client.close()

    run(scenario())


def test_event_handoff_is_bounded_and_loop_affinity_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: list[linklab.VoiceClient] = []

    async def scenario() -> None:
        connection = FakeConnection()

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,), websocket_max_queue=1),
            NullCallbacks(),
            object(),
        )
        holder.append(client)
        await client.connect()
        assert (
            client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x01\x02", 0, False, True, True))
            is linklab.AudioSubmitResult.ACCEPTED
        )
        async with asyncio.timeout(1):
            while len(connection.sent) != 3:
                await asyncio.sleep(0)

        state = linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING)
        closed = linklab.InputClosedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            1,
            linklab.InputCloseReason.ENDPOINT,
        )
        await connection.inbound.put(linklab.encode_message(state))
        await connection.inbound.put(linklab.encode_message(closed))
        await client.wait_closed()

        assert client._events.qsize() == 1
        assert connection.close_codes == [1011]

    run(scenario())

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await holder[0].wait_closed()

    run(wrong_loop())
