from __future__ import annotations

import asyncio
from typing import Self, Literal, Protocol, runtime_checkable
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
    ConnectionState,
    InputCloseReason,
    ProtocolObjectKind,
    ResponseCancelReason,
    ConversationEndReason,
    PlaybackInterruptReason,
)
from ._config import ServerConfig
from ._errors import WriterClosed, QueueOverflow, ProtocolViolation
from ._values import InputId, OutputId, ResponseId, ConversationId, ReadableBuffer
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
    InputAudioEvent,
    InputClosedEvent,
    OutputAudioEvent,
    OutputEndedEvent,
    InputAbortedEvent,
    InputStartedEvent,
    OutputStartedEvent,
    ResponseEndedEvent,
    ResponseStartedEvent,
    TranscriptFinalEvent,
    PlaybackFinishedEvent,
    TranscriptUpdateEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ResponseTextDeltaEvent,
    ResponseTextFinalEvent,
    ConversationStartedEvent,
    PlaybackInterruptedEvent,
    ConversationCancelledEvent,
)
from ._protocol import _Input, _Response, _TransitionResult
from ._handshake import _serve_websocket, _perform_server_handshake
from ._transport import _QueueLane, _TransportCore
from ._observability import _Observer, _QueueSnapshot, _lifecycle_boundary

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


class _LifecycleTimeouts:
    def __init__(self, session: ServerSession, config: ServerConfig) -> None:
        self._session = session
        self._loop = session._loop
        self._waiting_s = config.waiting_timeout_s
        self._input_s = config.input_timeout_s
        self._processing_s = config.processing_timeout_s
        self._idle_s = session._core._limits.idle_timeout_ms / 1_000
        self._key: tuple[str, int | None] | None = None
        self._object_deadline: float | None = None
        self._idle_deadline: float | None = None
        self._handle: asyncio.TimerHandle | None = None
        self._generation = 0

    @property
    def deadlines(self) -> tuple[float | None, float | None]:
        return self._object_deadline, self._idle_deadline

    def sync(self) -> None:
        key = self._session._timeout_phase()
        if key == self._key:
            return
        self._generation += 1
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        self._key = key
        self._object_deadline = None
        self._idle_deadline = None
        if key is None:
            return

        phase, _ = key
        now = self._loop.time()
        if phase == "input":
            self._object_deadline = now + self._input_s
        elif phase == "processing":
            self._object_deadline = now + self._processing_s
            self._idle_deadline = now + self._idle_s
        elif phase == "waiting":
            self._object_deadline = now + self._waiting_s
            self._idle_deadline = now + self._idle_s
        elif phase == "idle":
            self._idle_deadline = now + self._idle_s
        self._schedule()

    def close(self) -> None:
        self._generation += 1
        self._key = None
        self._object_deadline = None
        self._idle_deadline = None
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def run_due(self, now: float | None = None) -> None:
        generation = self._generation
        if self._handle is not None:
            self._handle.cancel()
        self._handle = None
        current = self._loop.time() if now is None else now
        key = self._key
        if key is None or key != self._session._timeout_phase():
            self.sync()
            return

        phase, _ = key
        if self._object_deadline is not None and current >= self._object_deadline:
            self._session._on_object_timeout(phase)
        elif self._idle_deadline is not None and current >= self._idle_deadline:
            self._session._on_idle_timeout()

        if generation == self._generation:
            self._schedule()

    def _schedule(self) -> None:
        deadlines = tuple(deadline for deadline in (self._object_deadline, self._idle_deadline) if deadline is not None)
        if not deadlines:
            return
        generation = self._generation

        def wake() -> None:
            if generation == self._generation:
                try:
                    self.run_due()
                except Exception as error:
                    self._session._core._request_close(1011, drain=False, loss=error)

        self._handle = self._loop.call_at(min(deadlines), wake)


class ResponseWriter:
    def __init__(self, session: ServerSession, response_id: ResponseId) -> None:
        self._session = session
        self._response_id = response_id
        self._next_text_sequence = 0
        self._output_writer: OutputWriter | None = None
        self._closed = False

    @property
    def response_id(self) -> ResponseId:
        return self._response_id

    async def send_text_delta(self, text: str) -> None:
        self._require_open()
        conversation_id = self._session._require_conversation_id()
        event = ResponseTextDeltaEvent(conversation_id, self._response_id, self._next_text_sequence, text)
        self._session._enqueue_control((event,))
        self._next_text_sequence += 1

    async def finalize_text(self, text: str) -> None:
        self._require_open()
        conversation_id = self._session._require_conversation_id()
        self._session._enqueue_control((ResponseTextFinalEvent(conversation_id, self._response_id, text),))

    async def start_output(self) -> OutputWriter:
        self._require_open()
        conversation_id = self._session._require_conversation_id()
        response = self._session._require_writer_response(self)
        if response.output is not None or self._output_writer is not None:
            raise ProtocolViolation("response may have at most one output")
        output_value, _ = self._session._core.validator._data.output_ids.allocate()
        output_id = OutputId(output_value)
        self._session._enqueue_control((OutputStartedEvent(conversation_id, self._response_id, output_id),))
        writer = OutputWriter(self, output_id)
        self._output_writer = writer
        return writer

    async def finish(self) -> None:
        self._require_open()
        conversation_id = self._session._require_conversation_id()
        response = self._session._require_writer_response(self)
        messages: list[Message] = [ResponseEndedEvent(conversation_id, self._response_id)]
        if response.output is not None:
            if response.output.terminal is None:
                raise ProtocolViolation("response cannot finish before its output")
        if response.output is None or isinstance(response.playback, PlaybackFinishedEvent):
            messages.extend(self._session._completed_response_messages(response))
        self._session._enqueue_control(tuple(messages))
        self._session._response_terminated()

    async def cancel(self, reason: ResponseCancelReason) -> None:
        self._require_open()
        if not isinstance(reason, ResponseCancelReason):
            raise TypeError("reason must be ResponseCancelReason")
        if reason not in (
            ResponseCancelReason.GENERATION_FAILED,
            ResponseCancelReason.TTS_FAILED,
            ResponseCancelReason.OVERFLOW,
            ResponseCancelReason.SHUTDOWN,
        ):
            raise ProtocolViolation("response cancellation reason is owned by peer or session logic")

        conversation_id = self._session._require_conversation_id()
        response = self._session._require_writer_response(self)
        if reason is ResponseCancelReason.OVERFLOW and response.output is None:
            raise ProtocolViolation("overflow cancellation requires a started output")

        output_writer = self._output_writer
        if output_writer is not None:
            output_writer._discard_queued()

        messages: list[Message]
        if reason in (ResponseCancelReason.GENERATION_FAILED, ResponseCancelReason.TTS_FAILED):
            code = (
                ErrorCode.GENERATION_FAILED
                if reason is ResponseCancelReason.GENERATION_FAILED
                else ErrorCode.TTS_FAILED
            )
            messages = [
                ErrorEvent(
                    ErrorScope.RESPONSE,
                    code,
                    True,
                    conversation_id,
                    response_id=self._response_id,
                )
            ]
            if not response.start.end_conversation:
                state = self._session._next_state(CoarseState.WAITING)
                if state is not None:
                    messages.append(state)
        elif reason is ResponseCancelReason.SHUTDOWN:
            messages = [
                ResponseCancelledEvent(conversation_id, self._response_id, reason),
                ConversationEndedEvent(conversation_id, ConversationEndReason.CANCELLED),
            ]
        else:
            messages = [
                ErrorEvent(
                    ErrorScope.RESPONSE,
                    ErrorCode.OUTPUT_OVERFLOW,
                    True,
                    conversation_id,
                    response_id=self._response_id,
                )
            ]
            if not response.start.end_conversation:
                state = self._session._next_state(CoarseState.WAITING)
                if state is not None:
                    messages.append(state)

        self._session._enqueue_control(tuple(messages))
        self._session._response_terminated()

    async def __aenter__(self) -> Self:
        self._require_open()
        return self

    async def __aexit__(self, exc_type: object, _exc: object, _traceback: object) -> Literal[False]:
        if exc_type is None:
            await self.finish()
        elif not self._closed:
            await self.cancel(ResponseCancelReason.GENERATION_FAILED)
        return False

    def _require_open(self) -> None:
        self._session._check_loop()
        if self._closed or self._session._response_writer is not self:
            raise WriterClosed("response writer is closed")
        self._session._require_writer_response(self)

    def _close(self) -> None:
        self._closed = True
        if self._output_writer is not None:
            self._output_writer._close()


class OutputWriter:
    def __init__(self, response_writer: ResponseWriter, output_id: OutputId) -> None:
        self._response_writer = response_writer
        self._session = response_writer._session
        self._output_id = output_id
        self._next_frame = 0
        self._closed = False

    @property
    def output_id(self) -> OutputId:
        return self._output_id

    async def send_audio(self, audio: ReadableBuffer) -> None:
        self._require_open()
        pcm = _copy_output_pcm(audio)
        conversation_id = self._session._require_conversation_id()
        limits = self._session._core.validator._data.server_hello
        if limits is None:
            raise ProtocolViolation("output audio requires negotiated limits")
        max_frames = limits.limits.max_output_audio_frames
        frame_count = len(pcm) // 2
        events = tuple(
            OutputAudioEvent(
                conversation_id,
                self._response_writer.response_id,
                self._output_id,
                self._next_frame + offset,
                pcm[offset * 2 : min(offset + max_frames, frame_count) * 2],
            )
            for offset in range(0, frame_count, max_frames)
        )
        try:
            self._session._core.enqueue_batch(
                events,
                lane=_QueueLane.DATA,
                weight=frame_count,
                tag=self,
            )
        except QueueOverflow:
            self._discard_queued()
            await self._response_writer.cancel(ResponseCancelReason.OVERFLOW)
            raise
        self._next_frame += frame_count

    async def finish(self) -> None:
        self._require_open()
        if self._next_frame == 0:
            raise ProtocolViolation("output must contain at least one audio frame")
        conversation_id = self._session._require_conversation_id()
        self._session._enqueue_control(
            (
                OutputEndedEvent(
                    conversation_id,
                    self._response_writer.response_id,
                    self._output_id,
                    self._next_frame,
                ),
            )
        )
        self._close()

    async def __aenter__(self) -> Self:
        self._require_open()
        return self

    async def __aexit__(self, exc_type: object, _exc: object, _traceback: object) -> Literal[False]:
        if exc_type is None:
            await self.finish()
        elif not self._closed:
            await self._response_writer.cancel(ResponseCancelReason.TTS_FAILED)
        return False

    def _require_open(self) -> None:
        self._session._check_loop()
        if self._closed or self._response_writer._output_writer is not self:
            raise WriterClosed("output writer is closed")
        response = self._session._require_writer_response(self._response_writer)
        if (
            response.output is None
            or response.output.start.output_id != self._output_id
            or response.output.terminal is not None
        ):
            self._close()
            raise WriterClosed("output writer is stale")

    def _discard_queued(self) -> None:
        self._session._core.discard_queued(lambda batch: batch.tag is self)

    def _close(self) -> None:
        self._closed = True


def _copy_output_pcm(audio: ReadableBuffer) -> bytes:
    try:
        view = memoryview(audio)
    except (TypeError, ValueError) as error:
        raise TypeError("audio must support the buffer protocol") from error
    if not view.contiguous:
        raise ValueError("audio must be contiguous")
    try:
        pcm = view.cast("B").tobytes()
    except TypeError as error:
        raise ValueError("audio must be a contiguous byte-addressable buffer") from error
    if not pcm:
        raise ValueError("audio must not be empty")
    if len(pcm) % 2:
        raise ValueError("audio must contain frame-aligned PCM S16LE")
    return pcm


class _HandlerQueue:
    def __init__(self, audio_capacity_frames: int, event_capacity: int, clock: Callable[[], float]) -> None:
        self._audio_capacity_frames = audio_capacity_frames
        self._event_capacity = event_capacity
        self._clock = clock
        self._items: deque[tuple[Message, int, float]] = deque()
        self._audio_frames = 0
        self._events = 0
        self._audio_overflows = 0
        self._event_overflows = 0
        self._available = asyncio.Event()
        self._closed = False

    def put_nowait(self, event: Message) -> bool:
        if self._closed:
            return False
        frames = len(event.audio) // 2 if isinstance(event, InputAudioEvent) else 0
        if frames:
            if self._audio_frames + frames > self._audio_capacity_frames:
                self._audio_overflows += 1
                return False
            self._audio_frames += frames
        else:
            if self._events >= self._event_capacity:
                self._event_overflows += 1
                return False
            self._events += 1
        self._items.append((event, frames, self._clock()))
        self._available.set()
        return True

    async def get(self) -> Message:
        while not self._items:
            if self._closed:
                raise RuntimeError("handler queue is closed")
            self._available.clear()
            await self._available.wait()
        event, frames, _ = self._items.popleft()
        if frames:
            self._audio_frames -= frames
        else:
            self._events -= 1
        if not self._items:
            self._available.clear()
        return event

    def discard(self, predicate: Callable[[Message], bool]) -> None:
        retained: deque[tuple[Message, int, float]] = deque()
        while self._items:
            event, frames, enqueued_at = self._items.popleft()
            if predicate(event):
                if frames:
                    self._audio_frames -= frames
                else:
                    self._events -= 1
            else:
                retained.append((event, frames, enqueued_at))
        self._items = retained
        if not self._items:
            self._available.clear()

    def snapshots(self) -> tuple[_QueueSnapshot, _QueueSnapshot]:
        now = self._clock()

        def residence(audio: bool) -> float:
            oldest = next((item for item in self._items if bool(item[1]) is audio), None)
            return 0.0 if oldest is None else max(0.0, (now - oldest[2]) * 1_000)

        return (
            _QueueSnapshot(
                "server.handler.audio",
                self._audio_capacity_frames,
                self._audio_frames,
                self._audio_overflows,
                residence(True),
                "frames",
            ),
            _QueueSnapshot(
                "server.handler.event",
                self._event_capacity,
                self._events,
                self._event_overflows,
                residence(False),
                "events",
            ),
        )

    def close(self) -> None:
        self._closed = True
        self._items.clear()
        self._audio_frames = 0
        self._events = 0
        self._available.set()


class ServerSession:
    def __init__(
        self,
        core: _TransportCore,
        *,
        config: ServerConfig,
        input_queue_frames: int,
        event_capacity: int,
        observer: _Observer | None = None,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._core = core
        self._observer = observer
        self._handler: ServerHandler | None = None
        self._events = _HandlerQueue(input_queue_frames, event_capacity, self._loop.time)
        self._handler_signals: asyncio.Queue[_HandlerSignal] = asyncio.Queue(maxsize=_CONTROL_CAPACITY)
        self._handler_signal_overflow = False
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._server_closed_inputs: set[InputId] = set()
        self._response_writer: ResponseWriter | None = None
        self._timeouts = _LifecycleTimeouts(self, config)
        self._close_timeout_s = config.close_timeout_s
        self._shutdown_task: asyncio.Task[None] | None = None

    def _set_handler(self, handler: ServerHandler) -> None:
        self._check_loop()
        if self._handler is not None:
            raise RuntimeError("server handler is already set")
        self._handler = handler
        self._dispatcher_task = self._loop.create_task(self._run_dispatcher(), name="linklab-server-handler")

    def _accept_inbound(self, message: Message, transition: _TransitionResult) -> None:
        self._check_loop()
        if ProtocolObjectKind.RESPONSE in transition.cancellation_targets:
            self._response_terminated(discard_output=True)
        if isinstance(message, (PlaybackFinishedEvent, PlaybackInterruptedEvent)):
            self._complete_playback_transition(message, transition)
        dispatch = transition.dispatch or (
            isinstance(message, (PlaybackFinishedEvent, PlaybackInterruptedEvent))
            and ProtocolObjectKind.RESPONSE in transition.cancellation_targets
        )
        if dispatch and not self._events.put_nowait(message):
            self._observe_handler_queues()
            self._record_handler_signal(_HandlerQueueFull(message))
            self._handle_handler_saturation(message)
        else:
            self._observe_handler_queues()
        boundary = _lifecycle_boundary(message)
        if boundary is not None:
            self._observe_handler_queues(boundary)
        if isinstance(message, InputStartedEvent):
            self._enqueue_state(CoarseState.LISTENING)
        elif isinstance(message, InputAbortedEvent):
            self._enqueue_state(CoarseState.PROCESSING)
        elif any(isinstance(item, ErrorEvent) and item.scope is ErrorScope.INPUT for item in transition.outbound):
            if isinstance(message, InputAudioEvent):
                self._server_closed_inputs.add(message.input_id)
            self._enqueue_state(CoarseState.WAITING)
        elif isinstance(message, ConversationCancelledEvent) and transition.dispatch:
            self._end_cancelled_conversation(message)
        self._timeouts.sync()

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

    async def start_response(self, input_id: InputId, *, end_conversation: bool = False) -> ResponseWriter:
        self._check_loop()
        conversation_id = self._require_conversation_id()
        if not self._core.validator._response_input_ready(input_id):
            raise ProtocolViolation("response requires the current finalized terminal input")
        response_value, _ = self._core.validator._data.response_ids.allocate()
        response_id = ResponseId(response_value)
        start = ResponseStartedEvent(conversation_id, response_id, input_id, end_conversation)
        state = self._next_state(CoarseState.RESPONDING)
        self._enqueue_control((start,) if state is None else (start, state))
        writer = ResponseWriter(self, response_id)
        self._response_writer = writer
        return writer

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
        self._response_terminated(discard_output=True)

    async def _shutdown(self) -> None:
        self._check_loop()
        if self._shutdown_task is None:
            self._shutdown_task = self._loop.create_task(
                self._shutdown_impl(),
                name="linklab-server-session-close",
            )
        await asyncio.shield(self._shutdown_task)

    async def _shutdown_impl(self) -> None:
        self._timeouts.close()
        if self._core.outcome is not None:
            await self._close_dispatcher()
            await self._core.wait_closed()
            return
        messages: list[Message] = []
        closed_input_id: InputId | None = None
        validator = self._core.validator
        conversation = validator._data.conversation
        if validator.state.connection_state is ConnectionState.READY and conversation is not None:
            input_ = validator._current_input(validator._data, conversation)
            if input_ is not None:
                if input_.terminal is None:
                    messages.append(
                        InputClosedEvent(
                            conversation.conversation_id,
                            input_.start.input_id,
                            input_.committed_end_frame,
                            InputCloseReason.FAILED,
                        )
                    )
                    closed_input_id = input_.start.input_id
                if input_.transcript_final is None:
                    messages.append(TranscriptFinalEvent(conversation.conversation_id, input_.start.input_id, ""))
            response = validator._current_response(validator._data, conversation)
            if response is not None:
                messages.append(
                    ResponseCancelledEvent(
                        conversation.conversation_id,
                        response.start.response_id,
                        ResponseCancelReason.SHUTDOWN,
                    )
                )
            messages.append(ConversationEndedEvent(conversation.conversation_id, ConversationEndReason.CANCELLED))

        if messages:
            self._response_terminated(discard_output=True)
            self._core.seal_orderly(tuple(messages))
            if closed_input_id is not None:
                self._server_closed_inputs.add(closed_input_id)
        await asyncio.gather(
            self._close_dispatcher(),
            self._core.close(1001, drain=bool(messages)),
        )

    def _transport_lost(self) -> None:
        """Invalidate loop-owned application work without attempting wire delivery."""
        self._timeouts.close()
        self._response_terminated(discard_output=True)
        self._events.close()
        task = self._dispatcher_task
        if task is not None and not task.done():
            task.cancel()

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
            self._response_terminated(discard_output=True)
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
        self._response_terminated(discard_output=True)

    async def _close_dispatcher(self) -> None:
        self._check_loop()
        self._observe_handler_queues("dispatcher_close")
        self._timeouts.close()
        self._response_terminated(discard_output=True)
        self._events.close()
        task = self._dispatcher_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
            done, _ = await asyncio.wait((task,), timeout=self._close_timeout_s)
            if done:
                with suppress(asyncio.CancelledError):
                    task.result()
        self._dispatcher_task = None

    async def _run_dispatcher(self) -> None:
        while True:
            try:
                event = await self._events.get()
            except RuntimeError:
                return
            self._observe_handler_queues()
            try:
                if not self._prepare_handler_event(event):
                    continue
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._core._request_close(1011, drain=False, loss=error)
                return
            try:
                await self._dispatch(event)
                if self._observer is not None:
                    self._observer.message(event, direction="inbound", phase="handler_dispatched")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._record_handler_signal(_HandlerRaised(event, error))
                if self._observer is not None:
                    self._observer.handler_failure(event, error)
                try:
                    self._handle_handler_failure(event)
                except Exception as mapping_error:
                    self._core._request_close(1011, drain=False, loss=mapping_error)
                    return

    def _prepare_handler_event(self, event: Message) -> bool:
        if not isinstance(event, InputAudioEvent):
            return True
        if event.input_id in self._server_closed_inputs:
            return False
        end_frame = event.start_frame + len(event.audio) // 2
        limits = self._core.validator._data.server_hello
        assert limits is not None
        input_ = self._core.validator._find_input(self._core.validator._data, event.input_id)
        reaches_limit = end_frame == limits.limits.max_input_frames and input_ is not None and input_.terminal is None
        state = self._next_state(CoarseState.PROCESSING) if reaches_limit else None
        terminals = self._core.commit_input_audio(
            event.input_id,
            end_frame,
            following=() if state is None else (state,),
        )
        if terminals:
            self._server_closed_inputs.add(event.input_id)
            self._timeouts.sync()
        return True

    def _handle_handler_failure(self, event: Message) -> None:
        conversation = self._core.validator._data.conversation
        if conversation is None or conversation.cancel is not None or conversation.failure is not None:
            return
        if isinstance(event, (InputStartedEvent, InputAudioEvent, InputAbortedEvent)):
            input_ = self._core.validator._current_input(self._core.validator._data, conversation)
            if (
                input_ is None
                or input_.start.input_id != event.input_id
                or input_.response_id is not None
                or input_.failure is not None
            ):
                return
            error = ErrorEvent(
                ErrorScope.INPUT,
                ErrorCode.STT_FAILED,
                True,
                conversation.conversation_id,
                event.input_id,
            )
            state = self._next_state(CoarseState.WAITING)
            self._enqueue_control((error,) if state is None else (error, state))
            self._server_closed_inputs.add(event.input_id)
            self._events.discard(
                lambda queued: isinstance(queued, InputAudioEvent) and queued.input_id == event.input_id
            )
            return
        if isinstance(event, (PlaybackFinishedEvent, PlaybackInterruptedEvent)):
            response = self._core.validator._current_response(self._core.validator._data, conversation)
            if response is None or response.start.response_id != event.response_id or response.terminal is not None:
                return
            error = ErrorEvent(
                ErrorScope.RESPONSE,
                ErrorCode.PLAYBACK_FAILED,
                True,
                conversation.conversation_id,
                response_id=event.response_id,
            )
            self._enqueue_control((error,))
            self._response_terminated(discard_output=True)
            return
        error = ErrorEvent(
            ErrorScope.CONVERSATION,
            ErrorCode.CONVERSATION_FAILED,
            True,
            conversation.conversation_id,
        )
        self._enqueue_control((error,))
        self._response_terminated(discard_output=True)

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

    def _observe_handler_queues(self, boundary: str | None = None) -> None:
        if self._observer is not None:
            self._observer.queue_snapshots(self._events.snapshots(), boundary=boundary)

    def _handle_handler_saturation(self, message: Message) -> None:
        if not isinstance(message, InputAudioEvent):
            self._core._request_close(1011, drain=False, loss=QueueOverflow("server handler event capacity exceeded"))
            return

        input_ = self._core.validator._find_input(self._core.validator._data, message.input_id)
        if input_ is None or input_.terminal is not None or input_.failure is not None:
            return
        conversation_id = self._require_conversation_id()
        error = ErrorEvent(
            ErrorScope.INPUT,
            ErrorCode.INPUT_OVERFLOW,
            True,
            conversation_id,
            message.input_id,
        )
        state = self._next_state(CoarseState.WAITING)
        self._enqueue_control((error,) if state is None else (error, state))
        self._server_closed_inputs.add(message.input_id)
        self._events.discard(lambda event: isinstance(event, InputAudioEvent) and event.input_id == message.input_id)

    def _enqueue_control(self, messages: tuple[Message, ...]) -> None:
        try:
            self._core.enqueue_batch(messages, lane=_QueueLane.CONTROL)
        except QueueOverflow as error:
            self._core._request_close(1011, drain=False, loss=error)
            raise
        self._timeouts.sync()

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

    def _require_writer_response(self, writer: ResponseWriter) -> _Response:
        if self._response_writer is not writer:
            raise WriterClosed("response writer is closed")
        try:
            response = self._require_live_response()
        except ProtocolViolation as error:
            writer._close()
            self._response_writer = None
            raise WriterClosed("response writer is closed") from error
        if response.start.response_id != writer.response_id:
            writer._close()
            self._response_writer = None
            raise WriterClosed("response writer is stale")
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

    def _completed_response_messages(self, response: _Response) -> tuple[Message, ...]:
        if response.start.end_conversation:
            return (
                ConversationEndedEvent(
                    response.start.conversation_id,
                    ConversationEndReason.COMPLETED,
                ),
            )
        state = self._next_state(CoarseState.WAITING)
        return () if state is None else (state,)

    def _complete_playback_transition(
        self,
        message: PlaybackFinishedEvent | PlaybackInterruptedEvent,
        transition: _TransitionResult,
    ) -> None:
        response = self._core.validator._find_response(self._core.validator._data, message.response_id)
        assert response is not None
        if isinstance(message, PlaybackFinishedEvent):
            if response.end is not None and not isinstance(response.terminal, ResponseCancelledEvent):
                self._enqueue_control(self._completed_response_messages(response))
            return
        if ProtocolObjectKind.RESPONSE not in transition.cancellation_targets:
            return
        if message.reason not in (
            PlaybackInterruptReason.LOCAL_CANCEL,
            PlaybackInterruptReason.SHUTDOWN,
        ):
            return
        conversation = self._core.validator._data.conversation
        if conversation is None:
            return
        if conversation.expected_end_reason is not None:
            self._enqueue_control(
                (
                    ConversationEndedEvent(
                        conversation.conversation_id,
                        conversation.expected_end_reason,
                    ),
                )
            )
            return
        state = self._next_state(CoarseState.WAITING)
        if state is not None:
            self._enqueue_control((state,))

    def _end_cancelled_conversation(self, message: ConversationCancelledEvent) -> None:
        messages = (
            *self._active_response_terminals(),
            ConversationEndedEvent(message.conversation_id, ConversationEndReason.CANCELLED),
        )
        self._enqueue_control(messages)
        self._response_terminated(discard_output=True)

    def _response_terminated(self, *, discard_output: bool = False) -> None:
        writer = self._response_writer
        if writer is not None:
            output = writer._output_writer
            if discard_output and output is not None:
                output._discard_queued()
            writer._close()
            self._response_writer = None

    def _timeout_phase(self) -> tuple[str, int | None] | None:
        validator = self._core.validator
        if validator.state.connection_state is not ConnectionState.READY:
            return None
        conversation = validator._data.conversation
        if conversation is None or conversation.cancel is not None or conversation.failure is not None:
            return None
        response = validator._current_response(validator._data, conversation)
        if response is not None:
            return ("response", int(response.start.response_id))
        input_ = validator._current_input(validator._data, conversation)
        if input_ is not None:
            if input_.terminal is None and input_.failure is None:
                return ("input", int(input_.start.input_id))
            if input_.failure is None and input_.response_id is None:
                return ("processing", int(input_.start.input_id))
        state = conversation.state
        if state is not None and state.state is CoarseState.WAITING:
            return ("waiting", int(conversation.conversation_id))
        return ("idle", int(conversation.conversation_id))

    def _on_object_timeout(self, phase: str) -> None:
        key = self._timeout_phase()
        if key is None or key[0] != phase:
            self._timeouts.sync()
            return
        conversation_id = self._require_conversation_id()
        conversation = self._core.validator._data.conversation
        assert conversation is not None
        input_ = self._core.validator._current_input(self._core.validator._data, conversation)
        if phase == "input":
            assert input_ is not None
            close = InputClosedEvent(
                conversation_id,
                input_.start.input_id,
                input_.committed_end_frame,
                InputCloseReason.MAX_DURATION,
            )
            state = self._next_state(CoarseState.PROCESSING)
            self._enqueue_control((close,) if state is None else (close, state))
            self._server_closed_inputs.add(input_.start.input_id)
            return
        if phase == "processing":
            assert input_ is not None
            error = ErrorEvent(
                ErrorScope.INPUT,
                ErrorCode.PROCESSING_TIMEOUT,
                True,
                conversation_id,
                input_.start.input_id,
            )
            state = self._next_state(CoarseState.WAITING)
            self._enqueue_control((error,) if state is None else (error, state))
            self._server_closed_inputs.add(input_.start.input_id)
            return
        if phase == "waiting":
            self._expire_conversation(conversation_id)

    def _on_idle_timeout(self) -> None:
        key = self._timeout_phase()
        if key is None or key[0] in ("input", "response"):
            self._timeouts.sync()
            return
        self._expire_conversation(self._require_conversation_id())

    def _expire_conversation(self, conversation_id: ConversationId) -> None:
        self._enqueue_control(
            (
                ErrorEvent(
                    ErrorScope.CONVERSATION,
                    ErrorCode.IDLE_TIMEOUT,
                    True,
                    conversation_id,
                ),
            )
        )
        self._response_terminated(discard_output=True)

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
        self._connection_serial = 0

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
        self._connection_serial += 1
        observer = _Observer(
            self._logger,
            endpoint_role=EndpointRole.SERVER.value,
            connection_id=f"server-{self._connection_serial}",
            clock=self._loop.time,
        )
        observer.lifecycle("connection_accepted", active_connections=len(self._connections))
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

            def transport_lost(_error: BaseException) -> None:
                if session is not None:
                    session._transport_lost()

            core = _TransportCore(
                connection,
                role=EndpointRole.SERVER,
                limits=handshake.server_hello.limits,
                data_capacity=output_capacity,
                control_capacity=_CONTROL_CAPACITY,
                close_timeout_s=self._config.close_timeout_s,
                on_transition=accept_inbound,
                on_transport_loss=transport_lost,
                ping_interval_s=self._config.ping_interval_s,
                ping_timeout_s=self._config.ping_timeout_s,
                occupancy_unit="frames",
                observer=observer,
            )
            session = ServerSession(
                core,
                config=self._config,
                input_queue_frames=self._config.input_queue_frames,
                event_capacity=self._config.websocket_max_queue,
                observer=observer,
            )
            self._sessions.add(session)
            handler = self._handler_factory(session)
            if not isinstance(handler, ServerHandler):
                raise TypeError("handler_factory must return a ServerHandler")
            session._set_handler(handler)
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
            session._observe_handler_queues("ready")
            await core.wait_closed()
        except WebSocketConnectionClosed:
            pass
        except asyncio.CancelledError:
            if core is not None:
                with suppress(Exception):
                    await core.close(1001)
            raise
        except Exception as error:
            observer.failure("server_connection_failed", error)
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
        try:
            sessions = tuple(self._sessions)
            if sessions:
                await asyncio.gather(*(session._shutdown() for session in sessions))
            listener.close()
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
