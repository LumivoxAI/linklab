from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from collections.abc import Mapping, Callable

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


@dataclass(frozen=True, slots=True)
class _QueueSnapshot:
    identity: str
    capacity: int
    occupancy: int
    overflow_count: int
    oldest_residence_ms: float
    occupancy_unit: str
    residence_unit: str = "milliseconds"


@dataclass(frozen=True, slots=True)
class _MessageObservation:
    family: str
    fields: tuple[tuple[str, object], ...]


class _SafeLogger:
    """Contain logger failures and expose only explicitly selected fields."""

    def __init__(self, logger: object, context: Mapping[str, object] | None = None) -> None:
        self._logger = logger
        self._context = dict(context or ())

    def bind(self, **values: object) -> _SafeLogger:
        context = {**self._context, **values}
        try:
            bound = getattr(self._logger, "bind")(**values)
        except Exception:
            bound = self._logger
        return _SafeLogger(bound, context)

    def emit(self, level: str, event: str, **fields: object) -> None:
        safe_fields = {**self._context, **fields}
        try:
            method = getattr(self._logger, level)
            method(event, **safe_fields)
        except Exception:
            pass


class _Observer:
    def __init__(
        self,
        logger: object,
        *,
        endpoint_role: str,
        connection_id: str,
        clock: Callable[[], float],
    ) -> None:
        self._logger = _SafeLogger(logger).bind(
            component="lumivox_linklab",
            endpoint_role=endpoint_role,
            connection_id=connection_id,
        )
        self._clock = clock
        self._debug_counts: dict[tuple[str, str, str], int] = {}
        self._queue_overflows: dict[str, int] = {}
        self._input_terminal_at: dict[int, float] = {}
        self._response_inputs: dict[int, int] = {}
        self._output_received_at: dict[tuple[int, int], float] = {}
        self._input_received_at: dict[tuple[int, int], float] = {}
        self._last_speech_at: dict[int, float] = {}
        self._barge_in_at: dict[int, float] = {}
        self._input_started_at: dict[int, float] = {}
        self._first_text_responses: set[int] = set()
        self._first_pcm_responses: set[int] = set()

    def lifecycle(self, event: str, **fields: object) -> None:
        self._logger.emit("info", event, **fields)

    def failure(self, event: str, error: BaseException | None = None, **fields: object) -> None:
        if error is not None:
            fields["exception_type"] = type(error).__name__
        self._logger.emit("error", event, **fields)

    def rtt(self, seconds: float) -> None:
        self._logger.emit("debug", "keepalive_rtt", rtt_ms=max(0.0, seconds * 1_000))

    def queue_snapshots(self, snapshots: tuple[_QueueSnapshot, ...], *, boundary: str | None = None) -> None:
        for snapshot in snapshots:
            previous_overflows = self._queue_overflows.get(snapshot.identity, 0)
            self._queue_overflows[snapshot.identity] = snapshot.overflow_count
            fields: dict[str, object] = {
                "queue": snapshot.identity,
                "capacity": snapshot.capacity,
                "occupancy": snapshot.occupancy,
                "occupancy_unit": snapshot.occupancy_unit,
                "overflow_count": snapshot.overflow_count,
                "oldest_residence": snapshot.oldest_residence_ms,
                "residence_unit": snapshot.residence_unit,
            }
            if boundary is not None:
                fields["boundary"] = boundary
                self._logger.emit("info", "queue_snapshot", **fields)
            elif snapshot.overflow_count > previous_overflows:
                self._logger.emit("warning", "queue_overflow", **fields)
            elif self._allow_debug("queue", snapshot.identity, "change"):
                self._logger.emit("debug", "queue_snapshot", **fields)

    def message(self, message: Message, *, direction: str, phase: str) -> None:
        fields = _message_fields(message)
        fields["direction"] = direction
        fields["phase"] = phase
        fields.update(self._stage_fields(message, phase))
        frequent = isinstance(
            message,
            (InputAudioEvent, OutputAudioEvent, TranscriptUpdateEvent, ResponseTextDeltaEvent),
        )
        if frequent:
            if self._allow_debug(type(message).__name__, direction, phase):
                self._logger.emit("debug", "protocol_message", **fields)
            return
        self._logger.emit("info", "protocol_message", **fields)

    def project(self, message: Message) -> _MessageObservation:
        return _MessageObservation(type(message).__name__, tuple(_message_fields(message).items()))

    def projected(
        self,
        observation: _MessageObservation,
        *,
        direction: str,
        phase: str,
        transport_queue_ms: float,
    ) -> None:
        fields = dict(observation.fields)
        fields.update(direction=direction, phase=phase, transport_queue_ms=max(0.0, transport_queue_ms))
        frequent = observation.family in {
            InputAudioEvent.__name__,
            OutputAudioEvent.__name__,
            TranscriptUpdateEvent.__name__,
            ResponseTextDeltaEvent.__name__,
        }
        if frequent and not self._allow_debug(observation.family, direction, phase):
            return
        self._logger.emit("debug" if frequent else "info", "protocol_message", **fields)

    def callback_failure(self, message: Message, error: BaseException) -> None:
        self.failure("application_callback_failed", error, **_message_fields(message))

    def handler_failure(self, message: Message, error: BaseException) -> None:
        self.failure("application_handler_failed", error, **_message_fields(message))

    def _allow_debug(self, family: str, direction: str, phase: str) -> bool:
        key = (family, direction, phase)
        count = self._debug_counts.get(key, 0) + 1
        self._debug_counts[key] = count
        return count == 1 or count % 64 == 0

    def _stage_fields(self, message: Message, phase: str) -> dict[str, object]:
        now = self._clock()
        fields: dict[str, object] = {}
        if isinstance(message, InputStartedEvent):
            self._input_started_at[int(message.conversation_id)] = now
            if message.reason.value == "barge_in":
                self._barge_in_at[int(message.input_id)] = now
                fields["latency_marker"] = "barge_in_started"
        if isinstance(message, StateEvent) and message.state.value == "listening":
            started = self._input_started_at.get(int(message.conversation_id))
            if started is not None:
                fields["input_start_to_listening_ms"] = (now - started) * 1_000
        if isinstance(message, InputClosedEvent):
            self._input_terminal_at[int(message.input_id)] = now
            fields["latency_marker"] = "input_endpoint"
            last_speech = self._last_speech_at.get(int(message.input_id))
            if last_speech is not None:
                fields["last_speech_to_endpoint_ms"] = (now - last_speech) * 1_000
        elif isinstance(message, InputAudioEvent):
            key = (int(message.input_id), message.start_frame)
            if message.speech:
                self._last_speech_at[int(message.input_id)] = now
                fields["latency_marker"] = "speech_audio"
            if phase == "received":
                self._input_received_at[key] = now
                fields["latency_marker"] = "input_audio_received"
            elif phase == "handler_dispatched" and key in self._input_received_at:
                fields["input_receipt_to_handler_ms"] = (now - self._input_received_at.pop(key)) * 1_000
        elif isinstance(message, TranscriptFinalEvent):
            fields.update(self._since_input_terminal(message.input_id, "endpoint_to_final_ms"))
        elif isinstance(message, ResponseStartedEvent):
            self._response_inputs[int(message.response_id)] = int(message.input_id)
        elif isinstance(message, (ResponseTextDeltaEvent, ResponseTextFinalEvent)):
            input_id = self._response_inputs.get(int(message.response_id))
            response_id = int(message.response_id)
            if input_id is not None and response_id not in self._first_text_responses:
                self._first_text_responses.add(response_id)
                fields.update(self._since_input_terminal(input_id, "endpoint_to_first_text_ms"))
        elif isinstance(message, OutputAudioEvent):
            input_id = self._response_inputs.get(int(message.response_id))
            response_id = int(message.response_id)
            if input_id is not None and message.start_frame == 0 and response_id not in self._first_pcm_responses:
                self._first_pcm_responses.add(response_id)
                fields.update(self._since_input_terminal(input_id, "endpoint_to_first_pcm_ms"))
            key = (int(message.output_id), message.start_frame)
            if phase == "received":
                self._output_received_at[key] = now
                fields["latency_marker"] = "output_pcm_received"
            elif phase == "callback_dispatched" and key in self._output_received_at:
                fields["pcm_receipt_to_callback_ms"] = (now - self._output_received_at.pop(key)) * 1_000
        elif isinstance(message, PlaybackInterruptedEvent) and message.reason.value == "barge_in":
            if self._barge_in_at:
                started = self._barge_in_at.pop(next(reversed(self._barge_in_at)))
                fields["barge_in_to_interruption_ms"] = (now - started) * 1_000
        return fields

    def _since_input_terminal(self, input_id: object, field: str) -> dict[str, object]:
        if type(input_id) is not int:
            return {}
        started = self._input_terminal_at.get(input_id)
        return {} if started is None else {field: (self._clock() - started) * 1_000}


def _message_fields(message: Message) -> dict[str, object]:
    fields: dict[str, object] = {"message_type": type(message).__name__}
    for name in ("conversation_id", "input_id", "response_id", "output_id"):
        value = getattr(message, name, None)
        if type(value) is int:
            fields[name] = value
    for name in ("reason", "state", "scope", "code", "position"):
        value = getattr(message, name, None)
        if isinstance(value, Enum):
            fields[name] = value.value
    for name in ("revision", "sequence", "generation", "accepted_end_frame", "total_frames", "played_frames"):
        value = getattr(message, name, None)
        if type(value) is int:
            fields[name] = value
    for name in ("speech", "fatal", "end_conversation"):
        value = getattr(message, name, None)
        if type(value) is bool:
            fields[name] = value
    text = getattr(message, "text", None)
    if type(text) is str:
        fields["text_bytes"] = len(text.encode("utf-8"))
    diagnostic = getattr(message, "message", None)
    if type(diagnostic) is str:
        fields["diagnostic_bytes"] = len(diagnostic.encode("utf-8"))
    audio = getattr(message, "audio", None)
    start_frame = getattr(message, "start_frame", None)
    if type(audio) is bytes and type(start_frame) is int:
        frame_count = len(audio) // 2
        fields.update(start_frame=start_frame, end_frame=start_frame + frame_count, frame_count=frame_count)
    if isinstance(message, (PlaybackFinishedEvent, PlaybackInterruptedEvent)):
        fields["frame_count"] = message.played_frames
    if isinstance(message, (InputClosedEvent, OutputEndedEvent)):
        fields["frame_start"] = 0
        fields["frame_end"] = (
            message.accepted_end_frame if isinstance(message, InputClosedEvent) else message.total_frames
        )
    if isinstance(message, ResponseStartedEvent):
        fields["input_id"] = int(message.input_id)
    if isinstance(message, StateEvent):
        fields["revision"] = message.revision
    return fields


def _lifecycle_boundary(message: Message) -> str | None:
    if isinstance(
        message,
        (
            ErrorEvent,
            StateEvent,
            InputStartedEvent,
            InputAbortedEvent,
            InputClosedEvent,
            OutputStartedEvent,
            OutputEndedEvent,
            ResponseStartedEvent,
            ResponseEndedEvent,
            ResponseCancelledEvent,
            ConversationStartedEvent,
            ConversationEndedEvent,
            ConversationCancelledEvent,
            PlaybackFinishedEvent,
            PlaybackInterruptedEvent,
        ),
    ):
        return type(message).__name__
    return None


__all__ = ["_MessageObservation", "_Observer", "_QueueSnapshot", "_lifecycle_boundary"]
