from __future__ import annotations

import asyncio
from typing import Self, Protocol, runtime_checkable
from contextlib import suppress
from collections.abc import Callable

from websockets.exceptions import ConnectionClosed as WebSocketConnectionClosed
from websockets.asyncio.server import Server, ServerConnection

from ._enums import EndpointRole
from ._config import ServerConfig
from ._messages import (
    Message,
    InputAudioEvent,
    InputAbortedEvent,
    InputStartedEvent,
    PlaybackFinishedEvent,
    ConversationStartedEvent,
    PlaybackInterruptedEvent,
    ConversationCancelledEvent,
)
from ._protocol import _TransitionResult
from ._handshake import _serve_websocket, _perform_server_handshake
from ._transport import _TransportCore

_CONTROL_CAPACITY = 16


@runtime_checkable
class ServerHandler(Protocol):
    async def on_conversation_started(self, session: ServerSession, event: ConversationStartedEvent) -> None: ...

    async def on_input_started(self, session: ServerSession, event: InputStartedEvent) -> None: ...

    async def on_input_audio(self, session: ServerSession, event: InputAudioEvent) -> None: ...

    async def on_input_aborted(self, session: ServerSession, event: InputAbortedEvent) -> None: ...

    async def on_playback_outcome(
        self,
        session: ServerSession,
        event: PlaybackFinishedEvent | PlaybackInterruptedEvent,
    ) -> None: ...

    async def on_conversation_cancelled(self, session: ServerSession, event: ConversationCancelledEvent) -> None: ...


class ServerSession:
    def __init__(self, core: _TransportCore) -> None:
        self._core = core
        self._handler: ServerHandler | None = None

    def _set_handler(self, handler: ServerHandler) -> None:
        self._handler = handler


class VoiceServer:
    def __init__(
        self,
        config: ServerConfig,
        handler_factory: Callable[[ServerSession], ServerHandler],
        logger: object,
    ) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("VoiceServer must be created in a running event loop") from error
        if not isinstance(config, ServerConfig):
            raise TypeError("config must be ServerConfig")
        if not callable(handler_factory):
            raise TypeError("handler_factory must be callable")
        self._config = config
        self._handler_factory = handler_factory
        self._logger = logger
        self._listener: Server | None = None
        self._serve_task: asyncio.Task[object] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._serve_started = False
        self._accepting = False
        self._connections: set[ServerConnection] = set()
        self._sessions: set[ServerSession] = set()
        self._closed = asyncio.Event()
        self._closed.set()

    async def serve(self) -> None:
        self._check_loop()
        if self._serve_started:
            raise RuntimeError("VoiceServer.serve() may only be called once")
        self._serve_started = True
        self._close_task = None
        self._closed.clear()
        self._accepting = True
        current = asyncio.current_task()
        assert current is not None
        self._serve_task = current
        try:
            self._listener = await _serve_websocket(self._config, self._handle_connection)
        except BaseException:
            self._accepting = False
            self._closed.set()
            raise
        finally:
            self._serve_task = None

    async def close(self) -> None:
        self._check_loop()
        if self._close_task is None:
            self._close_task = self._loop.create_task(self._close(), name="linklab-server-close")
        await asyncio.shield(self._close_task)

    async def wait_closed(self) -> None:
        self._check_loop()
        await self._closed.wait()

    async def __aenter__(self) -> Self:
        await self.serve()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> bool:
        await self.close()
        return False

    async def _handle_connection(self, connection: ServerConnection) -> None:
        if not self._accepting or len(self._connections) >= self._config.max_connections:
            await connection.close(code=1008)
            return
        self._connections.add(connection)
        core: _TransportCore | None = None
        session: ServerSession | None = None
        try:
            handshake = await _perform_server_handshake(connection, self._config)
            if handshake is None:
                return
            output_capacity = max(
                1,
                handshake.server_hello.output_format.sample_rate_hz * self._config.output_queue_ms // 1_000,
            )
            core = _TransportCore(
                connection,
                role=EndpointRole.SERVER,
                limits=handshake.server_hello.limits,
                data_capacity=output_capacity,
                control_capacity=_CONTROL_CAPACITY,
                close_timeout_s=self._config.close_timeout_s,
                on_transition=self._ignore_transition,
                on_transport_loss=self._ignore_transport_loss,
                ping_interval_s=self._config.ping_interval_s,
                ping_timeout_s=self._config.ping_timeout_s,
                occupancy_unit="frames",
            )
            session = ServerSession(core)
            self._sessions.add(session)
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
            handler = self._handler_factory(session)
            if not isinstance(handler, ServerHandler):
                raise TypeError("handler_factory must return a ServerHandler")
            session._set_handler(handler)
            await core.wait_closed()
        except WebSocketConnectionClosed:
            pass
        except asyncio.CancelledError:
            if core is not None:
                with suppress(Exception):
                    await core.close(1001)
            raise
        except Exception:
            if core is not None:
                with suppress(Exception):
                    await core.close(1011)
            else:
                with suppress(Exception):
                    await connection.close(code=1011)
        finally:
            if session is not None:
                self._sessions.discard(session)
            self._connections.discard(connection)

    async def _close(self) -> None:
        self._accepting = False
        listener = self._listener
        if listener is None:
            serve_task = self._serve_task
            if serve_task is not None and serve_task is not asyncio.current_task():
                serve_task.cancel()
                with suppress(asyncio.CancelledError):
                    await serve_task
            self._closed.set()
            return
        listener.close()
        try:
            await listener.wait_closed()
        finally:
            self._listener = None
            self._closed.set()

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("VoiceServer operation requires its event loop") from error
        if loop is not self._loop:
            raise RuntimeError("VoiceServer is bound to a different event loop")

    @staticmethod
    def _ignore_transition(_message: Message, _transition: _TransitionResult) -> None:
        return None

    @staticmethod
    def _ignore_transport_loss(_error: BaseException) -> None:
        return None
