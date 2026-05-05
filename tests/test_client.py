import socket
import asyncio
from typing import Any, cast
from collections.abc import Awaitable, Coroutine

import pytest
from websockets.asyncio.server import Server, ServerConnection

import lumivox_linklab as linklab
from lumivox_linklab._client import _CallbackPath, _CallbackRaised, _CallbackQueueSaturated
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


class RecordingCallbacks(NullCallbacks):
    def __init__(self) -> None:
        self.events: list[object] = []
        self.block_state = False
        self.block_output = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_state = False

    async def _record(self, event: object) -> None:
        self.events.append(event)

    async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None:
        await self._record(event)

    async def on_conversation_state(self, event: linklab.StateEvent) -> None:
        await self._record(event)
        if self.fail_state:
            self.fail_state = False
            raise RuntimeError("state callback failed")
        if self.block_state:
            self.entered.set()
            await self.release.wait()

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
        if self.block_output:
            self.entered.set()
            await self.release.wait()

    async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None:
        await self._record(event)

    async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None:
        await self._record(event)

    async def on_error(self, event: linklab.ErrorEvent) -> None:
        await self._record(event)


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


async def send_server(connection: FakeConnection, *messages: linklab.ServerMessage) -> None:
    for message in messages:
        await connection.inbound.put(linklab.encode_message(message))


async def wait_for_sent(connection: FakeConnection, count: int) -> None:
    async with asyncio.timeout(1):
        while len(connection.sent) < count:
            await asyncio.sleep(0)


async def open_input(client: linklab.VoiceClient, connection: FakeConnection) -> None:
    assert (
        client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x01\x02", 0, False, True, True))
        is linklab.AudioSubmitResult.ACCEPTED
    )
    await wait_for_sent(connection, 3)


async def start_response(connection: FakeConnection, *, end_conversation: bool = False) -> None:
    await send_server(
        connection,
        linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING),
        linklab.InputClosedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            1,
            linklab.InputCloseReason.ENDPOINT,
        ),
        linklab.StateEvent(linklab.ConversationId(1), 2, linklab.CoarseState.PROCESSING),
        linklab.TranscriptFinalEvent(linklab.ConversationId(1), linklab.InputId(1), "hello"),
        linklab.ResponseStartedEvent(
            linklab.ConversationId(1),
            linklab.ResponseId(1),
            linklab.InputId(1),
            end_conversation,
        ),
        linklab.StateEvent(linklab.ConversationId(1), 3, linklab.CoarseState.RESPONDING),
    )


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


def test_callbacks_are_ordered_include_terminal_rearm_and_stop_before_close_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        callbacks = RecordingCallbacks()

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(linklab.ClientConfig("ws://unused", (PCM_16K,)), callbacks, object())
        await client.connect()
        await open_input(client, connection)
        await start_response(connection, end_conversation=True)
        await send_server(
            connection,
            linklab.ResponseTextDeltaEvent(linklab.ConversationId(1), linklab.ResponseId(1), 0, "answer"),
            linklab.ResponseTextFinalEvent(linklab.ConversationId(1), linklab.ResponseId(1), "answer"),
            linklab.ResponseEndedEvent(linklab.ConversationId(1), linklab.ResponseId(1)),
            linklab.ConversationEndedEvent(linklab.ConversationId(1), linklab.ConversationEndReason.COMPLETED),
        )
        async with asyncio.timeout(1):
            while not any(isinstance(event, linklab.ConversationEndedEvent) for event in callbacks.events):
                await asyncio.sleep(0)

        await client.close()
        event_types = [type(event) for event in callbacks.events]
        assert event_types == [
            linklab.ConnectionStateEvent,
            linklab.ConnectionStateEvent,
            linklab.StateEvent,
            linklab.StateEvent,
            linklab.TranscriptFinalEvent,
            linklab.ResponseStartedEvent,
            linklab.StateEvent,
            linklab.ResponseTextDeltaEvent,
            linklab.ResponseTextFinalEvent,
            linklab.ResponseEndedEvent,
            linklab.ConversationEndedEvent,
            linklab.ConnectionStateEvent,
            linklab.ConnectionStateEvent,
        ]
        connection_states = [
            event.state for event in callbacks.events if isinstance(event, linklab.ConnectionStateEvent)
        ]
        assert connection_states == [
            linklab.ConnectionState.HANDSHAKING,
            linklab.ConnectionState.READY,
            linklab.ConnectionState.CLOSING,
            linklab.ConnectionState.DISCONNECTED,
        ]
        assert client._callback_task is None

    run(scenario())


def test_callback_failures_are_typed_and_dispatch_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        callbacks = RecordingCallbacks()
        callbacks.fail_state = True

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(linklab.ClientConfig("ws://unused", (PCM_16K,)), callbacks, object())
        await client.connect()
        await open_input(client, connection)
        await send_server(
            connection,
            linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING),
            linklab.TranscriptUpdateEvent(linklab.ConversationId(1), linklab.InputId(1), 1, "partial"),
        )
        async with asyncio.timeout(1):
            while not any(isinstance(event, linklab.TranscriptUpdateEvent) for event in callbacks.events):
                await asyncio.sleep(0)
        signal = client._callback_signals.get_nowait()
        assert isinstance(signal, _CallbackRaised)
        assert signal.path is _CallbackPath.EVENT
        assert isinstance(signal.error, RuntimeError)
        assert client.connection_state is linklab.ConnectionState.READY
        await client.close()

    run(scenario())


def test_slow_non_output_callback_is_bounded_and_does_not_block_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    holder: list[linklab.VoiceClient] = []

    async def scenario() -> None:
        connection = FakeConnection()
        callbacks = RecordingCallbacks()
        callbacks.block_state = True

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,), websocket_max_queue=1),
            callbacks,
            object(),
        )
        holder.append(client)
        await client.connect()
        await open_input(client, connection)

        state = linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING)
        await send_server(connection, state)
        await callbacks.entered.wait()
        await send_server(
            connection,
            linklab.TranscriptUpdateEvent(linklab.ConversationId(1), linklab.InputId(1), 1, "one"),
            linklab.TranscriptUpdateEvent(linklab.ConversationId(1), linklab.InputId(1), 2, "two"),
        )
        async with asyncio.timeout(1):
            while client._callback_signals.empty():
                await asyncio.sleep(0)
        signal = client._callback_signals.get_nowait()
        assert isinstance(signal, _CallbackQueueSaturated)
        assert signal.path is _CallbackPath.EVENT
        assert client._events.qsize() == 1
        assert client.connection_state is linklab.ConnectionState.READY
        assert connection.close_codes == []
        callbacks.release.set()
        await client.close()

    run(scenario())

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await holder[0].wait_closed()

    run(wrong_loop())


def test_barge_in_flushes_queued_pcm_and_late_stale_events(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        callbacks = RecordingCallbacks()
        callbacks.block_output = True

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(linklab.ClientConfig("ws://unused", (PCM_16K,)), callbacks, object())
        await client.connect()
        await open_input(client, connection)
        await start_response(connection)
        await send_server(
            connection,
            linklab.OutputStartedEvent(linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1)),
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 0, b"\x01\x02"
            ),
        )
        await callbacks.entered.wait()
        await send_server(
            connection,
            linklab.ResponseTextDeltaEvent(linklab.ConversationId(1), linklab.ResponseId(1), 0, "stale"),
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 1, b"\x03\x04"
            ),
        )
        async with asyncio.timeout(1):
            while client._events.qsize() < 2:
                await asyncio.sleep(0)

        assert (
            client.submit_annotated_audio(linklab.AnnotatedAudio(b"\x05\x06", 0, False, True, True))
            is linklab.AudioSubmitResult.ACCEPTED
        )
        assert client._events.qsize() == 1
        await send_server(
            connection,
            linklab.ResponseCancelledEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.ResponseCancelReason.BARGE_IN
            ),
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 2, b"\x07\x08"
            ),
        )
        callbacks.release.set()
        async with asyncio.timeout(1):
            while not any(isinstance(event, linklab.ResponseCancelledEvent) for event in callbacks.events):
                await asyncio.sleep(0)

        output_audio = [event for event in callbacks.events if isinstance(event, linklab.OutputAudioEvent)]
        assert [event.start_frame for event in output_audio] == [0]
        assert not any(isinstance(event, linklab.ResponseTextDeltaEvent) for event in callbacks.events)
        await client.close()

    run(scenario())


def test_slow_output_callback_uses_frame_capacity_and_reports_output_saturation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        connection = FakeConnection()
        callbacks = RecordingCallbacks()
        callbacks.block_output = True

        async def open_fake(_config: linklab.ClientConfig) -> _ClientHandshake:
            return fake_handshake(connection)

        monkeypatch.setattr("lumivox_linklab._client._open_client_websocket", open_fake)
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,), playback_queue_ms=1),
            callbacks,
            object(),
        )
        await client.connect()
        await open_input(client, connection)
        await start_response(connection)
        await send_server(
            connection,
            linklab.OutputStartedEvent(linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1)),
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 0, b"\x01\x02"
            ),
        )
        await callbacks.entered.wait()
        ten_frames = b"\x03\x04" * 10
        await send_server(
            connection,
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 1, ten_frames
            ),
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 11, ten_frames
            ),
            linklab.OutputEndedEvent(linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 21),
        )
        async with asyncio.timeout(1):
            while client._callback_signals.empty():
                await asyncio.sleep(0)
        signal = client._callback_signals.get_nowait()
        assert isinstance(signal, _CallbackQueueSaturated)
        assert signal.path is _CallbackPath.OUTPUT
        assert isinstance(signal.event, linklab.OutputAudioEvent)
        assert signal.event.start_frame == 11
        assert client.connection_state is linklab.ConnectionState.READY

        callbacks.release.set()
        async with asyncio.timeout(1):
            while not any(isinstance(event, linklab.OutputEndedEvent) for event in callbacks.events):
                await asyncio.sleep(0)
        output_audio = [event for event in callbacks.events if isinstance(event, linklab.OutputAudioEvent)]
        assert [event.start_frame for event in output_audio] == [0, 1]
        await client.close()

    run(scenario())
