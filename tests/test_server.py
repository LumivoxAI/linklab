import ssl
import socket
import asyncio
from typing import Any, cast
from collections.abc import Coroutine

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.client import connect

import lumivox_linklab as linklab
from lumivox_linklab._server import _HandlerRaised, _HandlerQueueFull
from lumivox_linklab._handshake import _SUBPROTOCOL, _open_client_websocket
from lumivox_linklab._transport import _QueueLane

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def server_config(**changes: Any) -> linklab.ServerConfig:
    values: dict[str, Any] = {
        "port": unused_port(),
        "output_formats": (PCM_16K,),
    }
    values.update(changes)
    return linklab.ServerConfig(**values)


def client_config(port: int) -> linklab.ClientConfig:
    return linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,))


class RecordingHandler:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationStartedEvent,
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
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationCancelledEvent,
    ) -> None:
        self.events.append(event)


async def wait_for_count(items: list[object], count: int) -> None:
    async with asyncio.timeout(1):
        while len(items) < count:
            await asyncio.sleep(0)


async def send_message(connection: Any, message: linklab.Message) -> None:
    await connection.send(linklab.encode_message(message))


def test_server_binds_returns_and_isolates_sessions_and_handlers() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=2)
        sessions: list[linklab.ServerSession] = []
        handlers: list[RecordingHandler] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            handler = RecordingHandler()
            sessions.append(session)
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        first = await _open_client_websocket(client_config(config.port))
        second = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 2)

        assert sessions[0] is not sessions[1]
        assert handlers[0] is not handlers[1]
        assert all(not handler.events for handler in handlers)

        await asyncio.gather(first.connection.close(), second.connection.close())
        await asyncio.gather(server.close(), server.close())
        await server.wait_closed()

    run(scenario())


def test_connection_cap_rejects_with_1008_and_releases_slot() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=1)
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        first = await _open_client_websocket(client_config(config.port))

        rejected = await connect(
            f"ws://127.0.0.1:{config.port}",
            subprotocols=[_SUBPROTOCOL],
            compression=None,
            proxy=None,
        )
        await rejected.wait_closed()
        assert rejected.close_code == 1008
        assert len(sessions) == 1

        await first.connection.close()
        async with asyncio.timeout(1):
            while True:
                try:
                    admitted = await _open_client_websocket(client_config(config.port))
                except ConnectionClosed as error:
                    assert error.rcvd is not None and error.rcvd.code == 1008
                    await asyncio.sleep(0)
                    continue
                break
        assert len(sessions) == 2

        await admitted.connection.close()
        await server.close()

    run(scenario())


def test_factory_failure_is_connection_local_and_releases_slot() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=2)
        sessions: list[linklab.ServerSession] = []
        handlers: list[RecordingHandler] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            if len(sessions) == 2:
                raise RuntimeError("factory failed")
            handler = RecordingHandler()
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        healthy = await _open_client_websocket(client_config(config.port))
        failed = await _open_client_websocket(client_config(config.port))
        await failed.connection.wait_closed()

        assert failed.connection.close_code == 1011
        assert healthy.connection.close_code is None
        assert len(handlers) == 1
        assert not handlers[0].events

        recovered = await _open_client_websocket(client_config(config.port))
        assert len(sessions) == 3
        assert len(handlers) == 2
        await asyncio.gather(healthy.connection.close(), recovered.connection.close())
        await server.close()

    run(scenario())


def test_serve_is_one_shot_and_context_manager_does_not_suppress() -> None:
    async def concurrent_serve() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        results = await asyncio.gather(server.serve(), server.serve(), return_exceptions=True)
        assert sum(result is None for result in results) == 1
        errors = [result for result in results if isinstance(result, RuntimeError)]
        assert len(errors) == 1
        with pytest.raises(RuntimeError, match="only be called once"):
            await server.serve()
        await server.close()

    async def context_manager() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        with pytest.raises(LookupError, match="body failed"):
            async with server as entered:
                assert entered is server
                raise LookupError("body failed")
        await server.wait_closed()
        with pytest.raises(OSError):
            await connect(f"ws://127.0.0.1:{config.port}", proxy=None)

    run(concurrent_serve())
    run(context_manager())


def test_close_before_serve_is_idempotent_and_ssl_context_is_caller_owned() -> None:
    async def scenario() -> None:
        plain = linklab.VoiceServer(server_config(), lambda _session: RecordingHandler(), object())
        await asyncio.gather(plain.close(), plain.close())
        await plain.wait_closed()

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        config = server_config(ssl_context=context)
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        await server.serve()
        assert config.ssl_context is context
        await server.close()
        await server.wait_closed()

    run(scenario())


def test_invalid_handler_result_closes_only_that_connection() -> None:
    async def scenario() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: cast(Any, object()), object())
        await server.serve()
        handshake = await _open_client_websocket(client_config(config.port))
        await handshake.connection.wait_closed()
        assert handshake.connection.close_code == 1011
        await server.close()

    run(scenario())


@pytest.mark.parametrize("interrupted", [False, True])
def test_handler_dispatches_every_inbound_event_in_validated_order(interrupted: bool) -> None:
    class InspectingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.committed_during_audio: list[int] = []

        async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.committed_during_audio.append(session._core.validator._data.inputs[-1].committed_end_frame)
            self.events.append(event)

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = InspectingHandler()

        def factory(session: linklab.ServerSession) -> InspectingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        connection_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(connection_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(connection_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        source = bytearray(b"\x01\x02\x03\x04")
        await send_message(
            client.connection,
            linklab.InputAudioEvent(connection_id, input_id, 0, True, cast(bytes, source)),
        )
        source[:] = b"\x00\x00\x00\x00"
        await send_message(
            client.connection,
            linklab.InputAbortedEvent(connection_id, input_id, linklab.InputAbortReason.CAPTURE_FAILED),
        )
        await wait_for_count(handler.events, 4)

        session = sessions[0]
        assert cast(linklab.InputAudioEvent, handler.events[2]).audio == b"\x01\x02\x03\x04"
        assert handler.committed_during_audio == [2]
        committed = session._core.validator._data.inputs[-1].committed_end_frame
        assert committed == 2

        response_id = linklab.ResponseId(1)
        output_id = linklab.OutputId(1)
        session._core.enqueue_batch(
            (
                linklab.TranscriptFinalEvent(connection_id, input_id, ""),
                linklab.ResponseStartedEvent(connection_id, response_id, input_id, False),
                linklab.OutputStartedEvent(connection_id, response_id, output_id),
                linklab.OutputAudioEvent(connection_id, response_id, output_id, 0, b"\x00\x00"),
                linklab.OutputEndedEvent(connection_id, response_id, output_id, 1),
                linklab.ResponseEndedEvent(connection_id, response_id),
            ),
            lane=_QueueLane.CONTROL,
        )
        if interrupted:
            outcome: linklab.Message = linklab.PlaybackInterruptedEvent(
                connection_id,
                response_id,
                output_id,
                1,
                linklab.PlaybackPosition.EXACT,
                linklab.PlaybackInterruptReason.LOCAL_CANCEL,
            )
        else:
            outcome = linklab.PlaybackFinishedEvent(connection_id, response_id, output_id, 1)
        await send_message(client.connection, outcome)
        await send_message(
            client.connection,
            linklab.ConversationCancelledEvent(connection_id, linklab.ConversationCancelReason.USER),
        )
        await wait_for_count(handler.events, 6)

        assert [event.type for event in cast(list[linklab.Message], handler.events)] == [
            "conversation.start",
            "input.start",
            "input.audio",
            "input.abort",
            outcome.type,
            "conversation.cancel",
        ]
        await client.connection.close()
        await server.close()

    run(scenario())


def test_handler_queue_full_and_callback_failure_are_explicit_signals() -> None:
    class BlockingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def on_conversation_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.ConversationStartedEvent,
        ) -> None:
            self.events.append(event)
            raise LookupError("handler failed")

        async def on_input_audio(self, _session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.events.append(event)
            self.entered.set()
            await self.release.wait()

    async def scenario() -> None:
        config = server_config(input_queue_frames=2)
        sessions: list[linklab.ServerSession] = []
        handler = BlockingHandler()

        def factory(session: linklab.ServerSession) -> BlockingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0" * 4))
        await handler.entered.wait()
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 2, True, b"\0" * 4))
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 4, True, b"\0" * 4))

        session = sessions[0]
        first = await asyncio.wait_for(session._handler_signals.get(), 1)
        second = await asyncio.wait_for(session._handler_signals.get(), 1)
        assert isinstance(first, _HandlerRaised)
        assert isinstance(first.error, LookupError)
        assert isinstance(second, _HandlerQueueFull)
        assert cast(linklab.InputAudioEvent, second.event).start_frame == 4

        handler.release.set()
        await client.connection.close()
        await server.close()

    run(scenario())


def test_endpoint_late_pcm_and_identical_terminal_events_are_not_dispatched() -> None:
    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = RecordingHandler()

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0" * 4))
        await wait_for_count(handler.events, 3)

        sessions[0]._core.enqueue_batch(
            (linklab.InputClosedEvent(conversation_id, input_id, 2, linklab.InputCloseReason.ENDPOINT),),
            lane=_QueueLane.CONTROL,
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 2, False, b"\0" * 4))
        cancel = linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.USER)
        await send_message(client.connection, cancel)
        await send_message(client.connection, cancel)
        await wait_for_count(handler.events, 4)
        await asyncio.sleep(0.01)

        assert [event.type for event in cast(list[linklab.Message], handler.events)] == [
            "conversation.start",
            "input.start",
            "input.audio",
            "conversation.cancel",
        ]
        await client.connection.close()
        await server.close()

    run(scenario())


def test_server_session_rejects_use_from_another_event_loop() -> None:
    retained: list[linklab.ServerSession] = []

    async def create_session() -> None:
        config = server_config()

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            retained.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], retained), 1)
        await client.connection.close()
        await server.close()

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await retained[0]._close_dispatcher()

    run(create_session())
    run(wrong_loop())
