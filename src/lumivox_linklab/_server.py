from __future__ import annotations

import asyncio
from typing import Self, Protocol, runtime_checkable
from contextlib import suppress
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable

from websockets.exceptions import ConnectionClosed as WebSocketConnectionClosed
from websockets.asyncio.server import Server, ServerConnection

from ._enums import (
    ErrorCode,
    ErrorScope,
    CoarseState,
    EndpointRole,
    InputCloseReason,
    ResponseCancelReason,
    ConversationEndReason,
)
from ._config import ServerConfig
from ._errors import ProtocolViolation
from ._values import InputId, ConversationId
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
    InputAudioEvent,
    InputClosedEvent,
    InputAbortedEvent,
    InputStartedEvent,
    TranscriptFinalEvent,
    PlaybackFinishedEvent,
    TranscriptUpdateEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ConversationStartedEvent,
    PlaybackInterruptedEvent,
    ConversationCancelledEvent,
)
from ._protocol import _Input, _Response, _TransitionResult
from ._handshake import _serve_websocket, _perform_server_handshake
from ._transport import _QueueLane, _TransportCore

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


@dataclass(frozen=True, slots=True)
class _HandlerQueueFull:
    event: Message


@dataclass(frozen=True, slots=True)
class _HandlerRaised:
    event: Message
    error: Exception


type _HandlerSignal = _HandlerQueueFull | _HandlerRaised


class _HandlerQueue:
    def __init__(self, audio_capacity_frames: int, event_capacity: int) -> None:
        self._audio_capacity_frames = audio_capacity_frames
        self._event_capacity = event_capacity
        self._items: deque[tuple[Message, int]] = deque()
        self._audio_frames = 0
        self._events = 0
        self._available = asyncio.Event()
        self._closed = False

    def put_nowait(self, event: Message) -> bool:
        if self._closed:
            return False
        frames = len(event.audio) // 2 if isinstance(event, InputAudioEvent) else 0
        if frames:
            if self._audio_frames + frames > self._audio_capacity_frames:
                return False
            self._audio_frames += frames
        else:
            if self._events >= self._event_capacity:
                return False
            self._events += 1
        self._items.append((event, frames))
        self._available.set()
        return True

    async def get(self) -> Message:
        while not self._items:
            if self._closed:
                raise RuntimeError("handler queue is closed")
            self._available.clear()
            await self._available.wait()
        event, frames = self._items.popleft()
        if frames:
            self._audio_frames -= frames
        else:
            self._events -= 1
        if not self._items:
            self._available.clear()
        return event

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._audio_frames = 0
        self._events = 0
        self._available.set()


class ServerSession:
    def __init__(self, core: _TransportCore, *, input_queue_frames: int, event_capacity: int) -> None:
        self._loop = asyncio.get_running_loop()
        self._core = core
        self._handler: ServerHandler | None = None
        self._events = _HandlerQueue(input_queue_frames, event_capacity)
        self._handler_signals: asyncio.Queue[_HandlerSignal] = asyncio.Queue(maxsize=_CONTROL_CAPACITY)
        self._handler_signal_overflow = False
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._server_closed_inputs: set[InputId] = set()

    def _set_handler(self, handler: ServerHandler) -> None:
        self._check_loop()
        if self._handler is not None:
            raise RuntimeError("server handler is already set")
        self._handler = handler
        self._dispatcher_task = self._loop.create_task(self._run_dispatcher(), name="linklab-server-handler")

    def _accept_inbound(self, message: Message, transition: _TransitionResult) -> None:
        self._check_loop()
        if transition.dispatch and not self._events.put_nowait(message):
            self._record_handler_signal(_HandlerQueueFull(message))
        if isinstance(message, InputStartedEvent):
            self._enqueue_state(CoarseState.LISTENING)
        elif isinstance(message, InputAbortedEvent):
            self._enqueue_state(CoarseState.PROCESSING)
        elif isinstance(message, ConversationCancelledEvent) and transition.dispatch:
            self._end_cancelled_conversation(message)

    async def close_input(self, input_id: InputId, reason: InputCloseReason) -> None:
        self._check_loop()
        conversation_id = self._require_conversation_id()
        input_ = self._require_current_input(input_id)
        close = InputClosedEvent(conversation_id, input_id, input_.committed_end_frame, reason)
        state = self._next_state(CoarseState.PROCESSING)
        self._enqueue_control((close,) if state is None else (close, state))
        self._server_closed_inputs.add(input_id)

    async def update_transcript(
        self,
        input_id: InputId,
        revision: int,
        text: str,
        language: str | None = None,
    ) -> None:
        self._check_loop()
        conversation_id = self._require_conversation_id()
        self._enqueue_control((TranscriptUpdateEvent(conversation_id, input_id, revision, text, language),))

    async def finalize_transcript(
        self,
        input_id: InputId,
        text: str,
        language: str | None = None,
    ) -> None:
        self._check_loop()
        conversation_id = self._require_conversation_id()
        self._enqueue_control((TranscriptFinalEvent(conversation_id, input_id, text, language),))

    async def end_conversation(self, reason: ConversationEndReason) -> None:
        self._check_loop()
        conversation_id = self._require_conversation_id()
        messages: list[Message] = []
        closed_input_id: InputId | None = None
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        input_ = self._core.validator._current_input(self._core.validator._data, conversation)
        if input_ is not None:
            if input_.terminal is None:
                messages.append(
                    InputClosedEvent(
                        conversation_id,
                        input_.start.input_id,
                        input_.committed_end_frame,
                        InputCloseReason.FAILED,
                    )
                )
                closed_input_id = input_.start.input_id
            if input_.transcript_final is None:
                messages.append(TranscriptFinalEvent(conversation_id, input_.start.input_id, ""))
        messages.extend(self._active_response_terminals())
        messages.append(ConversationEndedEvent(conversation_id, reason))
        self._enqueue_control(tuple(messages))
        if closed_input_id is not None:
            self._server_closed_inputs.add(closed_input_id)
        self._response_terminated()

    async def fail(self, scope: ErrorScope, code: ErrorCode, message: str | None = None) -> None:
        self._check_loop()
        if not isinstance(scope, ErrorScope):
            raise TypeError("scope must be ErrorScope")
        if not isinstance(code, ErrorCode):
            raise TypeError("code must be ErrorCode")
        if scope is ErrorScope.CONNECTION:
            self._enqueue_control((ErrorEvent(scope, code, True, message=message),))
            await self._core.close(1002, drain=True)
            return

        conversation_id = self._require_conversation_id()
        if scope is ErrorScope.CONVERSATION:
            conversation = self._core.validator._data.conversation
            assert conversation is not None
            input_ = self._core.validator._current_input(self._core.validator._data, conversation)
            event = ErrorEvent(scope, code, True, conversation_id, message=message)
            self._enqueue_control((event,))
            if input_ is not None and input_.terminal is None:
                self._server_closed_inputs.add(input_.start.input_id)
            self._response_terminated()
            return
        if scope is ErrorScope.INPUT:
            input_ = self._require_recoverable_input()
            event = ErrorEvent(scope, code, True, conversation_id, input_.start.input_id, message=message)
            state = self._next_state(CoarseState.WAITING)
            self._enqueue_control((event,) if state is None else (event, state))
            self._server_closed_inputs.add(input_.start.input_id)
            return

        assert scope is ErrorScope.RESPONSE
        response = self._require_live_response()
        event = ErrorEvent(scope, code, True, conversation_id, response_id=response.start.response_id, message=message)
        state = None
        if not response.start.end_conversation and code in (
            ErrorCode.GENERATION_FAILED,
            ErrorCode.TTS_FAILED,
            ErrorCode.OUTPUT_OVERFLOW,
        ):
            state = self._next_state(CoarseState.WAITING)
        self._enqueue_control((event,) if state is None else (event, state))
        self._response_terminated()

    async def _close_dispatcher(self) -> None:
        self._check_loop()
        self._events.close()
        task = self._dispatcher_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._dispatcher_task = None

    async def _run_dispatcher(self) -> None:
        while True:
            try:
                event = await self._events.get()
            except RuntimeError:
                return
            try:
                if isinstance(event, InputAudioEvent):
                    if event.input_id in self._server_closed_inputs:
                        continue
                    end_frame = event.start_frame + len(event.audio) // 2
                    self._core.commit_input_audio(event.input_id, end_frame)
                await self._dispatch(event)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._record_handler_signal(_HandlerRaised(event, error))

    async def _dispatch(self, event: Message) -> None:
        handler = self._handler
        assert handler is not None
        if isinstance(event, ConversationStartedEvent):
            await handler.on_conversation_started(self, event)
        elif isinstance(event, InputStartedEvent):
            await handler.on_input_started(self, event)
        elif isinstance(event, InputAudioEvent):
            await handler.on_input_audio(self, event)
        elif isinstance(event, InputAbortedEvent):
            await handler.on_input_aborted(self, event)
        elif isinstance(event, (PlaybackFinishedEvent, PlaybackInterruptedEvent)):
            await handler.on_playback_outcome(self, event)
        elif isinstance(event, ConversationCancelledEvent):
            await handler.on_conversation_cancelled(self, event)

    def _record_handler_signal(self, signal: _HandlerSignal) -> None:
        try:
            self._handler_signals.put_nowait(signal)
        except asyncio.QueueFull:
            self._handler_signal_overflow = True

    def _enqueue_control(self, messages: tuple[Message, ...]) -> None:
        self._core.enqueue_batch(messages, lane=_QueueLane.CONTROL)

    def _enqueue_state(self, state: CoarseState) -> None:
        event = self._next_state(state)
        if event is not None:
            self._enqueue_control((event,))

    def _next_state(self, state: CoarseState) -> StateEvent | None:
        conversation = self._core.validator._data.conversation
        if conversation is None:
            raise ProtocolViolation("operation requires an open conversation")
        previous = conversation.state
        if previous is not None and previous.state is state:
            return None
        revision = 1 if previous is None else previous.revision + 1
        return StateEvent(conversation.conversation_id, revision, state)

    def _require_conversation_id(self) -> ConversationId:
        conversation = self._core.validator._data.conversation
        if conversation is None:
            raise ProtocolViolation("operation requires an open conversation")
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("operation requires a live conversation")
        return conversation.conversation_id

    def _require_current_input(self, input_id: InputId) -> _Input:
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        input_ = self._core.validator._find_input(self._core.validator._data, input_id)
        if input_ is None or input_.start.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("operation targets no input in the open conversation")
        if conversation.input_id != input_id:
            raise ProtocolViolation("operation requires the current input")
        return input_

    def _require_recoverable_input(self) -> _Input:
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        input_ = self._core.validator._current_input(self._core.validator._data, conversation)
        if input_ is None or input_.response_id is not None or input_.failure is not None:
            raise ProtocolViolation("input failure requires one current recoverable input")
        return input_

    def _require_live_response(self) -> _Response:
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        response = self._core.validator._current_response(self._core.validator._data, conversation)
        if response is None or response.terminal is not None or response.failure is not None:
            raise ProtocolViolation("response failure requires one current live response")
        return response

    def _active_response_terminals(self) -> tuple[Message, ...]:
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        response = self._core.validator._current_response(self._core.validator._data, conversation)
        if response is None:
            return ()
        return (
            ResponseCancelledEvent(
                conversation.conversation_id,
                response.start.response_id,
                ResponseCancelReason.CONVERSATION_CANCELLED,
            ),
        )

    def _end_cancelled_conversation(self, message: ConversationCancelledEvent) -> None:
        messages = (
            *self._active_response_terminals(),
            ConversationEndedEvent(message.conversation_id, ConversationEndReason.CANCELLED),
        )
        self._enqueue_control(messages)
        self._response_terminated()

    def _response_terminated(self) -> None:
        """Hook for task 16 writer invalidation and producer cancellation."""

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("ServerSession operation requires its event loop") from error
        if loop is not self._loop:
            raise RuntimeError("ServerSession is bound to a different event loop")


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

            def accept_inbound(message: Message, transition: _TransitionResult) -> None:
                assert session is not None
                session._accept_inbound(message, transition)

            core = _TransportCore(
                connection,
                role=EndpointRole.SERVER,
                limits=handshake.server_hello.limits,
                data_capacity=output_capacity,
                control_capacity=_CONTROL_CAPACITY,
                close_timeout_s=self._config.close_timeout_s,
                on_transition=accept_inbound,
                on_transport_loss=self._ignore_transport_loss,
                ping_interval_s=self._config.ping_interval_s,
                ping_timeout_s=self._config.ping_timeout_s,
                occupancy_unit="frames",
            )
            session = ServerSession(
                core,
                input_queue_frames=self._config.input_queue_frames,
                event_capacity=self._config.websocket_max_queue,
            )
            self._sessions.add(session)
            handler = self._handler_factory(session)
            if not isinstance(handler, ServerHandler):
                raise TypeError("handler_factory must return a ServerHandler")
            session._set_handler(handler)
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
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
                await session._close_dispatcher()
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
    def _ignore_transport_loss(_error: BaseException) -> None:
        return None
