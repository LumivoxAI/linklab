import ssl
import socket
import asyncio
from typing import Any, cast
from collections.abc import Coroutine

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.client import connect

import lumivox_linklab as linklab
from lumivox_linklab._handshake import _SUBPROTOCOL, _open_client_websocket

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
