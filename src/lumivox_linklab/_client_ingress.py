from __future__ import annotations

from enum import StrEnum
from time import monotonic
from typing import Final
from threading import Lock
from collections import deque
from dataclasses import replace, dataclass
from collections.abc import Callable

from ._enums import (
    EndpointRole,
    ConnectionState,
    InputAbortReason,
    InputStartReason,
    PlaybackPosition,
    AudioSubmitResult,
    PlaybackInterruptReason,
    ConversationCancelReason,
)
from ._config import ClientConfig, ConnectionLimits
from ._errors import ProtocolViolation
from ._values import (
    InputId,
    OutputId,
    ResponseId,
    AnnotatedAudio,
    ConversationId,
    ProtocolStateSnapshot,
)
from ._messages import (
    Message,
    ClientHello,
    ServerHello,
    InputAudioEvent,
    InputClosedEvent,
    OutputAudioEvent,
    OutputEndedEvent,
    InputAbortedEvent,
    InputStartedEvent,
    OutputStartedEvent,
    ResponseEndedEvent,
    ResponseStartedEvent,
    PlaybackFinishedEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ResponseTextDeltaEvent,
    ResponseTextFinalEvent,
    ConversationStartedEvent,
    PlaybackInterruptedEvent,
    ConversationCancelledEvent,
)
from ._protocol import ProtocolValidator, _Input, _Response, _ValidatorData, _TransitionResult
from ._observability import _QueueSnapshot

_MAX_SEQUENCE: Final = 4_294_967_295


class _IngressLane(StrEnum):
    DATA = "data"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class _IngressBatch:
    messages: tuple[Message, ...]
    lane: _IngressLane
    audio_frames: int
    input_id: InputId | None
    enqueued_at: float


@dataclass(frozen=True, slots=True)
class _IngressSnapshot:
    capacity_frames: int
    occupancy_frames: int
    overflow_count: int
    control_capacity: int
    control_occupancy: int


@dataclass(frozen=True, slots=True)
class _AudioSpan:
    audio: bytes
    speech: bool
    retained_at: float = 0.0

    @property
    def frames(self) -> int:
        return len(self.audio) // 2


class _ClientAudioIngress:
    """Thread-safe semantic handoff between capture callbacks and the client loop."""

    def __init__(
        self,
        config: ClientConfig,
        *,
        notify: Callable[[], None] | None = None,
        control_capacity: int = 16,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not isinstance(config, ClientConfig):
            raise TypeError("config must be ClientConfig")
        if type(control_capacity) is not int or control_capacity <= 0:
            raise ValueError("control_capacity must be a positive integer")
        self._config = config
        self._notify = notify
        self._control_capacity = control_capacity
        self._clock = clock
        self._lock = Lock()
        self._validator = ProtocolValidator(EndpointRole.CLIENT)
        self._limits: ConnectionLimits | None = None
        self._batches: deque[_IngressBatch] = deque()
        self._occupancy_frames = 0
        self._control_occupancy = 0
        self._overflow_count = 0
        self._control_overflow_count = 0
        self._pre_roll_trim_count = 0
        self._notification_pending = False
        self._failure: RuntimeError | None = None
        self._notification_error: Exception | None = None
        self._generation: int | None = None
        self._pre_roll: deque[_AudioSpan] = deque()
        self._pre_roll_frames = 0
        self._confirmed_response_cancellations: set[ResponseId] = set()
        self._shutting_down = False

    def transport_connected(self) -> None:
        with self._lock:
            self._validator.transport_connected()

    def complete_handshake(self, client_hello: ClientHello, server_hello: ServerHello) -> None:
        if not isinstance(client_hello, ClientHello):
            raise TypeError("client_hello must be ClientHello")
        if not isinstance(server_hello, ServerHello):
            raise TypeError("server_hello must be ServerHello")
        with self._lock:
            data, _ = self._validator._transition(self._validator._data, client_hello)
            data, _ = self._validator._transition(data, server_hello)
            self._validator._data = data
            self._limits = server_hello.limits

    def start(self, client_hello: ClientHello, server_hello: ServerHello) -> None:
        self.transport_connected()
        self.complete_handshake(client_hello, server_hello)

    def begin_close(self) -> None:
        with self._lock:
            if self._validator.state.connection_state in (ConnectionState.HANDSHAKING, ConnectionState.READY):
                self._validator.begin_close()

    def prepare_shutdown(self) -> tuple[Message, ...]:
        """Freeze ingress and create the single reserved shutdown batch."""
        with self._lock:
            if self._shutting_down:
                return ()
            self._shutting_down = True
            data = self._validator._data
            conversation = data.conversation
            if data.connection_state is not ConnectionState.READY or conversation is None:
                return ()
            if conversation.cancel is not None or conversation.failure is not None:
                return ()

            messages: list[Message] = [
                ConversationCancelledEvent(conversation.conversation_id, ConversationCancelReason.SHUTDOWN)
            ]
            response = self._validator._current_response(data, conversation)
            if response is not None and response.output is not None and response.playback is None:
                messages.append(
                    PlaybackInterruptedEvent(
                        conversation.conversation_id,
                        response.start.response_id,
                        response.output.start.output_id,
                        0,
                        PlaybackPosition.ESTIMATED,
                        PlaybackInterruptReason.SHUTDOWN,
                    )
                )

            candidate = data
            for message in messages:
                candidate, _ = self._validator._transition(candidate, message)
            self._validator._data = candidate
            self._batches.clear()
            self._occupancy_frames = 0
            self._control_occupancy = 0
            self._notification_pending = False
            self._clear_pre_roll_locked()
            return tuple(messages)

    def stop(self) -> None:
        with self._lock:
            if self._validator.state.connection_state is not ConnectionState.DISCONNECTED:
                self._validator.transport_disconnected()
            self._limits = None
            self._batches.clear()
            self._occupancy_frames = 0
            self._control_occupancy = 0
            self._notification_pending = False
            self._failure = None
            self._notification_error = None
            self._generation = None
            self._confirmed_response_cancellations.clear()
            self._shutting_down = False
            self._clear_pre_roll_locked()

    def submit(self, annotated: AnnotatedAudio) -> AudioSubmitResult:
        if not isinstance(annotated, AnnotatedAudio):
            raise TypeError("annotated must be AnnotatedAudio")
        audio = self._copy_and_validate(annotated)
        with self._lock:
            result = self._submit_locked(annotated, audio)
        self._deliver_notification()
        return result

    def abort_input(self, reason: InputAbortReason) -> bool:
        if not isinstance(reason, InputAbortReason):
            raise TypeError("reason must be InputAbortReason")
        with self._lock:
            data = self._validator._data
            input_ = self._open_input(data)
            if (
                data.connection_state is not ConnectionState.READY
                or input_ is None
                or self._failure is not None
                or self._shutting_down
            ):
                return False
            message = InputAbortedEvent(input_.start.conversation_id, input_.start.input_id, reason)
            candidate, _ = self._validator._transition(data, message)
            if not self._enqueue_locked((message,), _IngressLane.CONTROL, 0, message.input_id):
                self._fail_locked("reserved ingress control capacity exhausted")
                accepted = False
            else:
                self._validator._data = candidate
                accepted = True
        self._deliver_notification()
        return accepted

    def cancel_conversation(self, reason: ConversationCancelReason) -> bool:
        if not isinstance(reason, ConversationCancelReason):
            raise TypeError("reason must be ConversationCancelReason")
        with self._lock:
            data = self._validator._data
            conversation = data.conversation
            if (
                data.connection_state is not ConnectionState.READY
                or conversation is None
                or conversation.cancel is not None
                or conversation.failure is not None
                or self._failure is not None
                or self._shutting_down
            ):
                return False
            message = ConversationCancelledEvent(conversation.conversation_id, reason)
            accepted = self._commit_control_locked(data, message, conversation.input_id)
        self._deliver_notification()
        return accepted

    def playback_finished(self, output_id: OutputId, played_frames: int) -> bool:
        self._validate_output_id(output_id)
        with self._lock:
            data = self._validator._data
            response = self._response_for_output(data, output_id)
            if (
                data.connection_state is not ConnectionState.READY
                or response is None
                or self._failure is not None
                or self._shutting_down
            ):
                return False
            message = PlaybackFinishedEvent(
                response.start.conversation_id,
                response.start.response_id,
                output_id,
                played_frames,
            )
            accepted = self._commit_control_locked(data, message, None)
        self._deliver_notification()
        return accepted

    def playback_interrupted(
        self,
        output_id: OutputId,
        played_frames: int,
        position: PlaybackPosition,
        reason: PlaybackInterruptReason,
    ) -> bool:
        self._validate_output_id(output_id)
        if not isinstance(position, PlaybackPosition):
            raise TypeError("position must be PlaybackPosition")
        if not isinstance(reason, PlaybackInterruptReason):
            raise TypeError("reason must be PlaybackInterruptReason")
        with self._lock:
            data = self._validator._data
            response = self._response_for_output(data, output_id)
            if (
                data.connection_state is not ConnectionState.READY
                or response is None
                or self._failure is not None
                or self._shutting_down
            ):
                return False
            message = PlaybackInterruptedEvent(
                response.start.conversation_id,
                response.start.response_id,
                output_id,
                played_frames,
                position,
                reason,
            )
            accepted = self._commit_control_locked(data, message, None)
        self._deliver_notification()
        return accepted

    def accept_inbound(self, message: Message) -> _TransitionResult:
        with self._lock:
            data, result = self._validator._transition(self._validator._data, message)
            self._validator._data = data
            if isinstance(message, ResponseCancelledEvent):
                if message.response_id not in self._confirmed_response_cancellations:
                    self._confirmed_response_cancellations.add(message.response_id)
                    result = replace(result, dispatch=True)
            if isinstance(message, InputClosedEvent):
                self._discard_input_audio_locked(message.input_id, message.accepted_end_frame)
            if isinstance(message, ConversationEndedEvent):
                self._generation = None
                self._clear_pre_roll_locked()
            return result

    def is_current(self, message: Message) -> bool:
        """Return whether queued response content is still publishable."""
        with self._lock:
            if isinstance(
                message,
                (
                    ResponseStartedEvent,
                    ResponseTextDeltaEvent,
                    ResponseTextFinalEvent,
                    OutputStartedEvent,
                    OutputAudioEvent,
                    OutputEndedEvent,
                    ResponseEndedEvent,
                ),
            ):
                response = self._validator._find_response(self._validator._data, message.response_id)
                return response is not None and not response.stale
            return True

    def next_batch(self) -> _IngressBatch | None:
        with self._lock:
            return self._batches[0] if self._batches else None

    def acknowledge(self, batch: _IngressBatch) -> None:
        with self._lock:
            if not self._batches or self._batches[0] is not batch:
                raise ValueError("batch is not the current ingress head")
            removed = self._batches.popleft()
            self._release_capacity_locked(removed)

    @property
    def state(self) -> ProtocolStateSnapshot:
        with self._lock:
            return self._validator.state

    @property
    def snapshot(self) -> _IngressSnapshot:
        with self._lock:
            return _IngressSnapshot(
                capacity_frames=self._config.input_queue_frames,
                occupancy_frames=self._occupancy_frames,
                overflow_count=self._overflow_count,
                control_capacity=self._control_capacity,
                control_occupancy=self._control_occupancy,
            )

    def snapshots(self) -> tuple[_QueueSnapshot, _QueueSnapshot, _QueueSnapshot]:
        with self._lock:
            now = self._clock()

            def batch_residence(lane: _IngressLane) -> float:
                oldest = next((batch for batch in self._batches if batch.lane is lane), None)
                return 0.0 if oldest is None else max(0.0, (now - oldest.enqueued_at) * 1_000)

            pre_roll_residence = 0.0 if not self._pre_roll else max(0.0, (now - self._pre_roll[0].retained_at) * 1_000)
            return (
                _QueueSnapshot(
                    "client.ingress.data",
                    self._config.input_queue_frames,
                    self._occupancy_frames,
                    self._overflow_count,
                    batch_residence(_IngressLane.DATA),
                    "frames",
                ),
                _QueueSnapshot(
                    "client.ingress.control",
                    self._control_capacity,
                    self._control_occupancy,
                    self._control_overflow_count,
                    batch_residence(_IngressLane.CONTROL),
                    "messages",
                ),
                _QueueSnapshot(
                    "client.ingress.pre_roll",
                    self._config.waiting_pre_roll_frames,
                    self._pre_roll_frames,
                    self._pre_roll_trim_count,
                    pre_roll_residence,
                    "frames",
                ),
            )

    @property
    def failure(self) -> RuntimeError | None:
        with self._lock:
            return self._failure

    @property
    def notification_error(self) -> Exception | None:
        with self._lock:
            return self._notification_error

    def _submit_locked(self, annotated: AnnotatedAudio, audio: bytes) -> AudioSubmitResult:
        data = self._validator._data
        if (
            data.connection_state is not ConnectionState.READY
            or self._limits is None
            or self._failure is not None
            or self._shutting_down
        ):
            return AudioSubmitResult.IGNORED_INACTIVE

        open_input = self._open_input(data)
        generation_boundary = (
            annotated.discontinuity or self._generation is not None and annotated.generation != self._generation
        )
        if generation_boundary:
            self._clear_pre_roll_locked()
            self._generation = annotated.generation
            if open_input is not None:
                abort = InputAbortedEvent(
                    open_input.start.conversation_id,
                    open_input.start.input_id,
                    InputAbortReason.DISCONTINUITY,
                )
                candidate, _ = self._validator._transition(data, abort)
                if not self._enqueue_locked((abort,), _IngressLane.CONTROL, 0, abort.input_id):
                    self._fail_locked("reserved ingress control capacity exhausted")
                    return AudioSubmitResult.OVERFLOW
                self._validator._data = candidate
                if annotated.activated:
                    self._retain_locked(_AudioSpan(audio, annotated.speech, self._clock()))
                return AudioSubmitResult.CLOSED_INPUT
        elif self._generation is None:
            self._generation = annotated.generation

        data = self._validator._data
        open_input = self._open_input(data)
        if open_input is not None:
            return self._append_open_input_locked(data, open_input.start.input_id, annotated.speech, audio)

        conversation = data.conversation
        if conversation is None:
            if not annotated.activated:
                return AudioSubmitResult.IGNORED_INACTIVE
            return self._start_activation_locked(data, annotated, audio)

        current_input = self._current_input(data)
        if (
            current_input is not None
            and current_input.terminal is not None
            and conversation.input_id is not None
            and conversation.response_id is None
        ):
            return AudioSubmitResult.CLOSED_INPUT

        if not annotated.speech:
            if annotated.activated:
                self._retain_locked(_AudioSpan(audio, False, self._clock()))
            return AudioSubmitResult.IGNORED_WAITING_SILENCE

        spans = (*self._pre_roll, _AudioSpan(audio, True, self._clock()))
        reason = InputStartReason.BARGE_IN if conversation.response_id is not None else InputStartReason.SPEECH
        return self._start_later_input_locked(data, reason, annotated.generation, spans)

    def _start_activation_locked(
        self, data: _ValidatorData, annotated: AnnotatedAudio, audio: bytes
    ) -> AudioSubmitResult:
        conversation_id = ConversationId(data.conversation_ids.next_value)
        input_id = InputId(data.input_ids.next_value)
        start_conversation = ConversationStartedEvent(conversation_id, "wake_word", annotated.wake_word)
        start_input = InputStartedEvent(
            conversation_id,
            input_id,
            InputStartReason.ACTIVATION,
            annotated.generation,
        )
        audio_messages = self._audio_messages(
            conversation_id,
            input_id,
            0,
            (_AudioSpan(audio, annotated.speech, self._clock()),),
        )
        self._require_aggregate_range(0, len(audio) // 2)
        messages: tuple[Message, ...] = (start_conversation, start_input, *audio_messages)
        return self._commit_audio_batch_locked(data, messages, input_id, len(audio) // 2)

    def _start_later_input_locked(
        self,
        data: _ValidatorData,
        reason: InputStartReason,
        generation: int,
        spans: tuple[_AudioSpan, ...],
    ) -> AudioSubmitResult:
        conversation = data.conversation
        assert conversation is not None
        input_id = InputId(data.input_ids.next_value)
        response_id: ResponseId | None = None
        if reason is InputStartReason.BARGE_IN:
            assert conversation.response_id is not None
            response_id = conversation.response_id
        start = InputStartedEvent(
            conversation.conversation_id,
            input_id,
            reason,
            generation,
            response_id,
        )
        audio_messages = self._audio_messages(conversation.conversation_id, input_id, 0, spans)
        messages: tuple[Message, ...] = (start, *audio_messages)
        frames = sum(span.frames for span in spans)
        self._require_aggregate_range(0, frames)
        result = self._commit_audio_batch_locked(data, messages, input_id, frames)
        if result is AudioSubmitResult.ACCEPTED:
            self._clear_pre_roll_locked()
        return result

    def _append_open_input_locked(
        self,
        data: _ValidatorData,
        input_id: InputId,
        speech: bool,
        audio: bytes,
    ) -> AudioSubmitResult:
        input_ = self._validator._find_input(data, input_id)
        assert input_ is not None
        frames = len(audio) // 2
        self._require_aggregate_range(input_.received_end_frame, frames)
        messages = self._audio_messages(
            input_.start.conversation_id,
            input_id,
            input_.received_end_frame,
            (_AudioSpan(audio, speech),),
        )
        if self._occupancy_frames + frames > self._config.input_queue_frames:
            self._overflow_count += 1
            abort = InputAbortedEvent(input_.start.conversation_id, input_id, InputAbortReason.OVERFLOW)
            candidate, _ = self._validator._transition(data, abort)
            if self._enqueue_locked((abort,), _IngressLane.CONTROL, 0, input_id):
                self._validator._data = candidate
            else:
                self._fail_locked("reserved ingress control capacity exhausted")
            return AudioSubmitResult.OVERFLOW
        return self._commit_audio_batch_locked(data, messages, input_id, frames)

    def _commit_audio_batch_locked(
        self,
        data: _ValidatorData,
        messages: tuple[Message, ...],
        input_id: InputId,
        frames: int,
    ) -> AudioSubmitResult:
        if self._occupancy_frames + frames > self._config.input_queue_frames:
            self._overflow_count += 1
            return AudioSubmitResult.OVERFLOW
        candidate = data
        for message in messages:
            candidate, _ = self._validator._transition(candidate, message)
        if not self._enqueue_locked(messages, _IngressLane.DATA, frames, input_id):
            return AudioSubmitResult.OVERFLOW
        self._validator._data = candidate
        return AudioSubmitResult.ACCEPTED

    def _audio_messages(
        self,
        conversation_id: ConversationId,
        input_id: InputId,
        start_frame: int,
        spans: tuple[_AudioSpan, ...],
    ) -> tuple[InputAudioEvent, ...]:
        assert self._limits is not None
        maximum = self._limits.max_input_audio_frames
        messages: list[InputAudioEvent] = []
        offset = start_frame
        for span in spans:
            for byte_offset in range(0, len(span.audio), maximum * 2):
                chunk = span.audio[byte_offset : byte_offset + maximum * 2]
                messages.append(InputAudioEvent(conversation_id, input_id, offset, span.speech, chunk))
                offset += len(chunk) // 2
        return tuple(messages)

    def _enqueue_locked(
        self,
        messages: tuple[Message, ...],
        lane: _IngressLane,
        audio_frames: int,
        input_id: InputId | None,
    ) -> bool:
        if lane is _IngressLane.DATA:
            if self._occupancy_frames + audio_frames > self._config.input_queue_frames:
                self._overflow_count += 1
                return False
        elif self._control_occupancy >= self._control_capacity:
            self._control_overflow_count += 1
            return False
        was_empty = not self._batches
        batch = _IngressBatch(messages, lane, audio_frames, input_id, self._clock())
        self._batches.append(batch)
        if lane is _IngressLane.DATA:
            self._occupancy_frames += audio_frames
        else:
            self._control_occupancy += 1
        if was_empty:
            self._notification_pending = True
        return True

    def _commit_control_locked(
        self,
        data: _ValidatorData,
        message: Message,
        input_id: InputId | None,
    ) -> bool:
        try:
            candidate, _ = self._validator._transition(data, message)
        except ProtocolViolation:
            return False
        if not self._enqueue_locked((message,), _IngressLane.CONTROL, 0, input_id):
            self._fail_locked("reserved ingress control capacity exhausted")
            return False
        self._validator._data = candidate
        return True

    def _discard_input_audio_locked(self, input_id: InputId, accepted_end_frame: int) -> None:
        retained: deque[_IngressBatch] = deque()
        while self._batches:
            batch = self._batches.popleft()
            if batch.input_id != input_id or batch.lane is _IngressLane.CONTROL:
                retained.append(batch)
                continue
            messages = tuple(
                message
                for message in batch.messages
                if not isinstance(message, InputAudioEvent) or message.start_frame < accepted_end_frame
            )
            frames = sum(len(message.audio) // 2 for message in messages if isinstance(message, InputAudioEvent))
            self._occupancy_frames -= batch.audio_frames - frames
            if messages and frames:
                retained.append(_IngressBatch(messages, batch.lane, frames, batch.input_id, batch.enqueued_at))
        self._batches = retained

    def _retain_locked(self, span: _AudioSpan) -> None:
        capacity = self._config.waiting_pre_roll_frames
        if capacity == 0:
            return
        self._pre_roll.append(span)
        self._pre_roll_frames += span.frames
        while self._pre_roll_frames > capacity:
            self._pre_roll_trim_count += 1
            first = self._pre_roll.popleft()
            excess = self._pre_roll_frames - capacity
            if first.frames > excess:
                trimmed = _AudioSpan(first.audio[excess * 2 :], first.speech, first.retained_at)
                self._pre_roll.appendleft(trimmed)
                self._pre_roll_frames -= excess
                break
            self._pre_roll_frames -= first.frames

    def _clear_pre_roll_locked(self) -> None:
        self._pre_roll.clear()
        self._pre_roll_frames = 0

    def _release_capacity_locked(self, batch: _IngressBatch) -> None:
        if batch.lane is _IngressLane.DATA:
            self._occupancy_frames -= batch.audio_frames
        else:
            self._control_occupancy -= 1

    def _deliver_notification(self) -> None:
        callback: Callable[[], None] | None = None
        with self._lock:
            if self._notification_pending:
                self._notification_pending = False
                callback = self._notify
        if callback is not None:
            try:
                callback()
            except Exception as error:
                with self._lock:
                    self._notification_error = error
                    self._notification_pending = True

    def _fail_locked(self, message: str) -> None:
        if self._failure is None:
            self._failure = RuntimeError(message)
        self._notification_pending = True

    def _require_aggregate_range(self, start_frame: int, frames: int) -> None:
        assert self._limits is not None
        if start_frame + frames > self._limits.max_input_frames:
            raise ValueError("audio range exceeds max_input_frames")

    def _open_input(self, data: _ValidatorData) -> _Input | None:
        current = self._current_input(data)
        return current if current is not None and current.terminal is None else None

    def _current_input(self, data: _ValidatorData) -> _Input | None:
        conversation = data.conversation
        if conversation is None or conversation.input_id is None:
            return None
        return self._validator._find_input(data, conversation.input_id)

    def _response_for_output(self, data: _ValidatorData, output_id: OutputId) -> _Response | None:
        for response in reversed(data.responses):
            if response.output is not None and response.output.start.output_id == output_id:
                if response.playback is not None:
                    return None
                return response
        return None

    @staticmethod
    def _validate_output_id(output_id: OutputId) -> None:
        if type(output_id) is not int or not 1 <= output_id <= 4_294_967_295:
            raise ValueError("output_id must be an integer in 1..4294967295")

    @staticmethod
    def _copy_and_validate(annotated: AnnotatedAudio) -> bytes:
        if type(annotated.generation) is not int or not 0 <= annotated.generation <= _MAX_SEQUENCE:
            raise ValueError("generation must be an integer in 0..4294967295")
        if type(annotated.discontinuity) is not bool:
            raise ValueError("discontinuity must be a bool")
        if type(annotated.speech) is not bool:
            raise ValueError("speech must be a bool")
        if type(annotated.activated) is not bool:
            raise ValueError("activated must be a bool")
        if annotated.wake_word is not None:
            if type(annotated.wake_word) is not str:
                raise ValueError("wake_word must be a str or None")
            if not 1 <= len(annotated.wake_word.encode("utf-8")) <= 64:
                raise ValueError("wake_word must contain 1..64 UTF-8 bytes")
        try:
            view = memoryview(annotated.audio)
        except (TypeError, ValueError) as error:
            raise ValueError("audio must support the buffer protocol") from error
        if not view.contiguous:
            raise ValueError("audio must be contiguous")
        try:
            audio = view.cast("B").tobytes()
        except TypeError as error:
            raise ValueError("audio must be a contiguous byte-addressable buffer") from error
        if not audio:
            raise ValueError("audio must not be empty")
        if len(audio) % 2:
            raise ValueError("audio must contain frame-aligned PCM S16LE")
        return audio
