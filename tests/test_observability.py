from __future__ import annotations

import logging
from typing import Any, Self

import pytest

from lumivox_linklab import (
    InputId,
    OutputId,
    ErrorCode,
    ErrorEvent,
    ErrorScope,
    ResponseId,
    StateEvent,
    AudioFormat,
    CoarseState,
    ClientConfig,
    ConversationId,
    InputAudioEvent,
    InputAbortReason,
    InputClosedEvent,
    InputCloseReason,
    InputStartReason,
    OutputAudioEvent,
    PlaybackPosition,
    InputAbortedEvent,
    InputStartedEvent,
    ResponseCancelReason,
    ResponseStartedEvent,
    TranscriptFinalEvent,
    ConversationEndReason,
    TranscriptUpdateEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ResponseTextDeltaEvent,
    ResponseTextFinalEvent,
    PlaybackInterruptReason,
    PlaybackInterruptedEvent,
)
from lumivox_linklab._client import VoiceClient, _CallbackPath, _InboundEvent, _CallbackQueue
from lumivox_linklab._server import ServerSession, _HandlerQueue
from lumivox_linklab._transport import _QueueLane, _TransportCore, _BoundedBatchQueue
from lumivox_linklab._observability import _Observer, _QueueSnapshot
from lumivox_linklab._client_ingress import _ClientAudioIngress


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class _RecordingLogger:
    def __init__(
        self,
        records: list[tuple[str, str, dict[str, object]]] | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        self.records = [] if records is None else records
        self.context = {} if context is None else context

    def bind(self, **new_values: Any) -> Self:
        return type(self)(self.records, {**self.context, **new_values})

    def debug(self, event: str, **kwargs: Any) -> None:
        self._record("debug", event, kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        self._record("info", event, kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._record("warning", event, kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._record("error", event, kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._record("critical", event, kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        self._record("exception", event, kwargs)

    def _record(self, level: str, event: str, fields: dict[str, object]) -> None:
        self.records.append((level, event, {**self.context, **fields}))


class _RaisingLogger:
    def bind(self, **new_values: Any) -> Self:
        raise RuntimeError("logger bind failed")

    def debug(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger debug failed")

    def info(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger info failed")

    def warning(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger warning failed")

    def error(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger error failed")

    def critical(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger critical failed")

    def exception(self, event: str, **kwargs: Any) -> None:
        raise RuntimeError("logger exception failed")


class _SurfaceLogger(_RecordingLogger):
    def __init__(self) -> None:
        super().__init__()
        self.accessed: set[str] = set()

    def __getattribute__(self, name: str) -> Any:
        if name in {"bind", "debug", "info", "warning", "error", "exception"}:
            object.__getattribute__(self, "accessed").add(name)
        return super().__getattribute__(name)

    def bind(self, **new_values: Any) -> Self:
        self.context.update(new_values)
        return self


def _client_config() -> ClientConfig:
    return ClientConfig(
        uri="ws://127.0.0.1:8765",
        output_formats=(
            AudioFormat("pcm_s16le", 24_000, 1),
            AudioFormat("pcm_s16le", 16_000, 1),
        ),
    )


def test_every_owned_logical_queue_has_a_complete_snapshot() -> None:
    clock = _Clock()
    client_transport = _BoundedBatchQueue(
        identity="client.transport_outbound",
        data_capacity=10,
        control_capacity=2,
        occupancy_unit="frames",
        clock=clock,
    )
    server_transport = _BoundedBatchQueue(
        identity="server.transport_outbound",
        data_capacity=20,
        control_capacity=2,
        occupancy_unit="frames",
        clock=clock,
    )
    callbacks = _CallbackQueue(4, 40, lambda: None, clock)
    handler = _HandlerQueue(80, 4, clock)
    ingress = _ClientAudioIngress(_client_config(), clock=clock)

    snapshots = (
        client_transport.snapshot(_QueueLane.DATA),
        client_transport.snapshot(_QueueLane.CONTROL),
        server_transport.snapshot(_QueueLane.DATA),
        server_transport.snapshot(_QueueLane.CONTROL),
        *callbacks.snapshots(),
        *handler.snapshots(),
        *ingress.snapshots(),
    )
    assert {snapshot.identity for snapshot in snapshots} == {
        "client.transport_outbound.data",
        "client.transport_outbound.control",
        "server.transport_outbound.data",
        "server.transport_outbound.control",
        "client.callbacks.event",
        "client.callbacks.output",
        "client.callbacks.control",
        "server.handler.audio",
        "server.handler.event",
        "client.ingress.data",
        "client.ingress.control",
        "client.ingress.pre_roll",
    }
    for snapshot in snapshots:
        assert snapshot.capacity >= 0
        assert snapshot.occupancy == 0
        assert snapshot.overflow_count == 0
        assert snapshot.oldest_residence_ms == 0.0
        assert snapshot.occupancy_unit in {"events", "frames", "messages"}
        assert snapshot.residence_unit == "milliseconds"


def test_logger_surface_is_allowlisted_and_process_logging_is_untouched() -> None:
    root = logging.getLogger()
    before = (root.level, tuple(root.handlers), root.disabled, logging.raiseExceptions)
    logger = _SurfaceLogger()
    observer = _Observer(logger, endpoint_role="client", connection_id="client-1", clock=lambda: 1.0)

    observer.lifecycle("ready", state="ready")
    observer.failure("failed", RuntimeError("secret detail"))
    observer.rtt(0.001)
    observer.queue_snapshots((_QueueSnapshot("queue", 1, 1, 1, 1.0, "events"),))

    assert logger.accessed <= {"bind", "debug", "info", "warning", "error", "exception"}
    assert logger.accessed == {"bind", "debug", "info", "warning", "error"}
    assert before == (root.level, tuple(root.handlers), root.disabled, logging.raiseExceptions)


def test_every_real_queue_snapshot_is_emitted_at_lifecycle_boundaries() -> None:
    clock = _Clock()
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=clock)
    client_transport = _BoundedBatchQueue(
        identity="client.transport_outbound",
        data_capacity=10,
        control_capacity=2,
        occupancy_unit="frames",
        clock=clock,
    )
    server_transport = _BoundedBatchQueue(
        identity="server.transport_outbound",
        data_capacity=20,
        control_capacity=2,
        occupancy_unit="frames",
        clock=clock,
    )
    callbacks = _CallbackQueue(4, 40, lambda: None, clock)
    handler = _HandlerQueue(80, 4, clock)
    ingress = _ClientAudioIngress(_client_config(), clock=clock)
    client = object.__new__(VoiceClient)
    client._observer = observer
    client._events = callbacks
    client._ingress = ingress
    client._core = None
    session = object.__new__(ServerSession)
    session._observer = observer
    session._events = handler
    client_core = object.__new__(_TransportCore)
    client_core._observer = observer
    client_core._queue = client_transport
    server_core = object.__new__(_TransportCore)
    server_core._observer = observer
    server_core._queue = server_transport

    client._observe_all_queues("ready")
    session._observe_handler_queues("ready")
    client_core._observe_queues("ready")
    server_core._observe_queues("ready")

    records = [fields for _, event, fields in logger.records if event == "queue_snapshot"]
    assert {fields["queue"] for fields in records} == {
        "client.transport_outbound.data",
        "client.transport_outbound.control",
        "server.transport_outbound.data",
        "server.transport_outbound.control",
        "client.callbacks.event",
        "client.callbacks.output",
        "client.callbacks.control",
        "server.handler.audio",
        "server.handler.event",
        "client.ingress.data",
        "client.ingress.control",
        "client.ingress.pre_roll",
    }
    assert all(fields["boundary"] == "ready" for fields in records)
    assert all(
        {"capacity", "occupancy", "occupancy_unit", "overflow_count", "oldest_residence", "residence_unit"}
        <= fields.keys()
        for fields in records
    )


def test_real_queue_pressure_and_overflow_changes_are_each_emitted() -> None:
    clock = _Clock()
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="client", connection_id="client-1", clock=clock)
    transport = _BoundedBatchQueue(
        identity="client.transport_outbound",
        data_capacity=2,
        control_capacity=1,
        occupancy_unit="frames",
        clock=clock,
    )
    callbacks = _CallbackQueue(1, 1, lambda: None, clock)
    handler = _HandlerQueue(1, 1, clock)
    client = object.__new__(VoiceClient)
    client._observer = observer
    client._events = callbacks
    session = object.__new__(ServerSession)
    session._observer = observer
    session._events = handler
    core = object.__new__(_TransportCore)
    core._observer = observer
    core._queue = transport

    core._observe_queues()
    client._observe_callback_queues()
    session._observe_handler_queues()
    transport.put_nowait((b"one",), lane=_QueueLane.DATA, weight=1)
    callbacks.put_nowait(
        _InboundEvent(StateEvent(ConversationId(1), 1, CoarseState.LISTENING)),
        _CallbackPath.EVENT,
    )
    handler.put_nowait(InputAudioEvent(ConversationId(1), InputId(1), 0, True, b"\0\0"))
    core._observe_queues()
    client._observe_callback_queues()
    session._observe_handler_queues()
    transport.put_nowait((b"two",), lane=_QueueLane.DATA, weight=1)
    assert not callbacks.put_nowait(
        _InboundEvent(StateEvent(ConversationId(1), 2, CoarseState.PROCESSING)),
        _CallbackPath.EVENT,
    )
    assert not handler.put_nowait(InputAudioEvent(ConversationId(1), InputId(1), 1, True, b"\0\0"))
    core._observe_queues()
    client._observe_callback_queues()
    session._observe_handler_queues()

    transport_records = [
        fields
        for _, event, fields in logger.records
        if event == "queue_snapshot" and fields["queue"] == "client.transport_outbound.data"
    ]
    assert [fields["occupancy"] for fields in transport_records] == [0]
    overflows = {fields["queue"] for _, event, fields in logger.records if event == "queue_overflow"}
    assert overflows == {"client.callbacks.event", "server.handler.audio"}


def test_queue_snapshots_are_emitted_with_pressure_and_boundary_context() -> None:
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="client", connection_id="client-1", clock=lambda: 1.0)
    observer.queue_snapshots(
        (_QueueSnapshot("client.ingress.data", 10, 10, 1, 25.0, "frames"),),
    )
    observer.queue_snapshots(
        (_QueueSnapshot("client.ingress.data", 10, 0, 1, 0.0, "frames"),),
        boundary="conversation_end",
    )

    overflow = next(record for record in logger.records if record[1] == "queue_overflow")
    assert overflow[0] == "warning"
    assert overflow[2]["queue"] == "client.ingress.data"
    assert overflow[2]["overflow_count"] == 1
    assert overflow[2]["oldest_residence"] == 25.0
    boundary = logger.records[-1]
    assert boundary[0:2] == ("info", "queue_snapshot")
    assert boundary[2]["boundary"] == "conversation_end"


def test_protocol_logs_never_contain_pcm_full_text_or_peer_diagnostics() -> None:
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=lambda: 1.0)
    pcm_secret = b"PCM-SECRET-1234"
    text_secret = "TRANSCRIPT-SECRET"
    diagnostic_secret = "PEER-DIAGNOSTIC-SECRET"

    observer.message(
        InputAudioEvent(ConversationId(1), InputId(1), 12, True, pcm_secret + b"0"),
        direction="inbound",
        phase="received",
    )
    observer.message(
        TranscriptFinalEvent(ConversationId(1), InputId(1), text_secret),
        direction="outbound",
        phase="queued",
    )
    observer.message(
        ErrorEvent(ErrorScope.CONNECTION, ErrorCode.PROTOCOL_STATE, True, message=diagnostic_secret),
        direction="outbound",
        phase="queued",
    )

    rendered = repr(logger.records)
    assert "PCM-SECRET" not in rendered
    assert text_secret not in rendered
    assert diagnostic_secret not in rendered
    protocol_fields = [fields for _, event, fields in logger.records if event == "protocol_message"]
    assert protocol_fields[0]["start_frame"] == 12
    assert protocol_fields[0]["frame_count"] == len(pcm_secret + b"0") // 2
    assert protocol_fields[1]["text_bytes"] == len(text_secret)
    assert protocol_fields[2]["diagnostic_bytes"] == len(diagnostic_secret)
    assert all(not isinstance(value, bytes) for fields in protocol_fields for value in fields.values())


def test_audio_and_token_debug_logging_is_rate_limited() -> None:
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=lambda: 1.0)
    for frame in range(130):
        observer.message(
            InputAudioEvent(ConversationId(1), InputId(1), frame, True, b"\0\0"),
            direction="inbound",
            phase="received",
        )

    records = [record for record in logger.records if record[1] == "protocol_message"]
    assert len(records) == 3
    assert all(record[0] == "debug" for record in records)


def test_logger_exceptions_are_not_suppressed() -> None:
    with pytest.raises(RuntimeError, match="logger bind failed"):
        _Observer(_RaisingLogger(), endpoint_role="client", connection_id="client-1", clock=lambda: 1.0)


def test_logs_expose_all_ids_transitions_reasons_frame_ranges_and_rtt() -> None:
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=lambda: 1.0)
    messages = (
        StateEvent(ConversationId(11), 3, CoarseState.PROCESSING, "endpoint"),
        InputAbortedEvent(ConversationId(11), InputId(12), InputAbortReason.CAPTURE_FAILED),
        ResponseCancelledEvent(ConversationId(11), ResponseId(13), ResponseCancelReason.TTS_FAILED),
        ConversationEndedEvent(ConversationId(11), ConversationEndReason.SERVER_FAILED),
        PlaybackInterruptedEvent(
            ConversationId(11),
            ResponseId(13),
            OutputId(14),
            19,
            PlaybackPosition.ESTIMATED,
            PlaybackInterruptReason.PLAYBACK_FAILED,
        ),
        OutputAudioEvent(ConversationId(11), ResponseId(13), OutputId(14), 20, b"\0\0" * 4),
    )
    for message in messages:
        observer.message(message, direction="inbound", phase="received")
    observer.rtt(0.0125)

    fields = [item for _, event, item in logger.records if event == "protocol_message"]
    assert any(
        item.get("conversation_id") == 11 and item.get("input_id") == 12 and item.get("reason") == "capture_failed"
        for item in fields
    )
    assert any(
        item.get("response_id") == 13
        and item.get("output_id") == 14
        and item.get("position") == "estimated"
        and item.get("reason") == "playback_failed"
        for item in fields
    )
    assert any(item.get("state") == "processing" and item.get("revision") == 3 for item in fields)
    assert any(item.get("reason") == "server_failed" for item in fields)
    assert any(
        item.get("start_frame") == 20 and item.get("end_frame") == 24 and item.get("frame_count") == 4
        for item in fields
    )
    assert next(item for _, event, item in logger.records if event == "keepalive_rtt")["rtt_ms"] == 12.5


def test_text_and_pcm_logs_are_safe_and_each_frequent_family_is_rate_limited() -> None:
    logger = _RecordingLogger()
    observers = (
        _Observer(logger, endpoint_role="client", connection_id="client-1", clock=lambda: 1.0),
        _Observer(logger, endpoint_role="server", connection_id="server-1", clock=lambda: 1.0),
    )
    transcript_secret = "TRANSCRIPT-FULL-SECRET"
    response_secret = "RESPONSE-FULL-SECRET"
    pcm_secret = b"PCM-FULL-SECRET!"
    for observer in observers:
        for sequence in range(130):
            observer.message(
                TranscriptUpdateEvent(ConversationId(1), InputId(1), sequence + 1, transcript_secret),
                direction="inbound",
                phase="received",
            )
            observer.message(
                ResponseTextDeltaEvent(ConversationId(1), ResponseId(1), sequence, response_secret),
                direction="outbound",
                phase="queued",
            )
            observer.message(
                InputAudioEvent(ConversationId(1), InputId(1), sequence * 8, True, pcm_secret),
                direction="inbound",
                phase="received",
            )
            observer.message(
                OutputAudioEvent(ConversationId(1), ResponseId(1), OutputId(1), sequence * 8, pcm_secret),
                direction="outbound",
                phase="queued",
            )
        observer.message(
            TranscriptFinalEvent(ConversationId(1), InputId(1), transcript_secret),
            direction="outbound",
            phase="queued",
        )
        observer.message(
            ResponseTextFinalEvent(ConversationId(1), ResponseId(1), response_secret),
            direction="inbound",
            phase="received",
        )

    rendered = repr(logger.records)
    assert transcript_secret not in rendered
    assert response_secret not in rendered
    assert "PCM-FULL-SECRET" not in rendered
    frequent = [record for record in logger.records if record[1] == "protocol_message" and record[0] == "debug"]
    for endpoint_role in ("client", "server"):
        for family in (
            "TranscriptUpdateEvent",
            "ResponseTextDeltaEvent",
            "InputAudioEvent",
            "OutputAudioEvent",
        ):
            assert (
                sum(
                    fields["endpoint_role"] == endpoint_role and fields["message_type"] == family
                    for _, _, fields in frequent
                )
                == 3
            )
    normal = [record for record in logger.records if record[0] == "info" and record[1] == "protocol_message"]
    assert {fields["message_type"] for _, _, fields in normal} == {
        "TranscriptFinalEvent",
        "ResponseTextFinalEvent",
    }


def test_all_required_latency_span_boundaries_are_correlatable() -> None:
    clock = _Clock()
    logger = _RecordingLogger()
    client = _Observer(logger, endpoint_role="client", connection_id="client-1", clock=clock)
    server = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=clock)
    input_audio = InputAudioEvent(ConversationId(7), InputId(8), 32, True, b"\0\0" * 4)

    client.message(input_audio, direction="outbound", phase="queued")
    observation = client.project(input_audio)
    clock.now += 0.004
    client.projected(observation, direction="outbound", phase="sent", transport_queue_ms=4.0)
    clock.now += 0.003
    server.message(input_audio, direction="inbound", phase="received")
    clock.now += 0.002
    server.message(input_audio, direction="inbound", phase="handler_dispatched")

    started = InputStartedEvent(ConversationId(7), InputId(8), InputStartReason.SPEECH, 1)
    server.message(started, direction="inbound", phase="received")
    clock.now += 0.002
    server.message(StateEvent(ConversationId(7), 2, CoarseState.LISTENING), direction="outbound", phase="queued")
    clock.now += 0.005
    closed = InputClosedEvent(ConversationId(7), InputId(8), 36, InputCloseReason.ENDPOINT)
    server.message(closed, direction="outbound", phase="queued")
    clock.now += 0.003
    server.message(
        TranscriptFinalEvent(ConversationId(7), InputId(8), "not logged"),
        direction="outbound",
        phase="queued",
    )
    server.message(
        ResponseStartedEvent(ConversationId(7), ResponseId(9), InputId(8), False),
        direction="outbound",
        phase="queued",
    )
    clock.now += 0.004
    server.message(
        ResponseTextDeltaEvent(ConversationId(7), ResponseId(9), 0, "not logged"),
        direction="outbound",
        phase="queued",
    )
    first_pcm = OutputAudioEvent(ConversationId(7), ResponseId(9), OutputId(10), 0, b"\0\0")
    clock.now += 0.002
    server.message(first_pcm, direction="outbound", phase="queued")

    client.message(closed, direction="inbound", phase="received")
    client.message(
        ResponseStartedEvent(ConversationId(7), ResponseId(9), InputId(8), False),
        direction="inbound",
        phase="received",
    )
    clock.now += 0.003
    client.message(first_pcm, direction="inbound", phase="received")
    clock.now += 0.001
    client.message(first_pcm, direction="inbound", phase="callback_dispatched")
    barge_in = InputStartedEvent(ConversationId(7), InputId(11), InputStartReason.BARGE_IN, 2, ResponseId(9))
    client.message(barge_in, direction="outbound", phase="queued")
    clock.now += 0.002
    client.message(
        PlaybackInterruptedEvent(
            ConversationId(7),
            ResponseId(9),
            OutputId(10),
            1,
            PlaybackPosition.EXACT,
            PlaybackInterruptReason.BARGE_IN,
        ),
        direction="outbound",
        phase="queued",
    )

    protocol = [fields for _, event, fields in logger.records if event == "protocol_message"]
    audio_key = (7, 8, 32, 36)
    capture_send_receipt = {
        (fields["conversation_id"], fields["input_id"], fields["start_frame"], fields["end_frame"])
        for fields in protocol
        if fields.get("message_type") == "InputAudioEvent"
        and fields.get("phase") in {"queued", "sent", "received", "handler_dispatched"}
    }
    assert capture_send_receipt == {audio_key}
    assert {fields["phase"] for fields in protocol if fields.get("message_type") == "InputAudioEvent"} >= {
        "queued",
        "sent",
        "received",
        "handler_dispatched",
    }
    assert any(fields.get("transport_queue_ms") == 4.0 for fields in protocol)
    assert any(fields.get("input_start_to_listening_ms") == pytest.approx(2.0) for fields in protocol)
    assert any(fields.get("last_speech_to_endpoint_ms") == pytest.approx(9.0) for fields in protocol)
    assert any(fields.get("endpoint_to_final_ms") == pytest.approx(3.0) for fields in protocol)
    assert any(fields.get("endpoint_to_first_text_ms") == pytest.approx(7.0) for fields in protocol)
    assert any(fields.get("endpoint_to_first_pcm_ms") == pytest.approx(9.0) for fields in protocol)
    assert any(fields.get("pcm_receipt_to_callback_completion_ms") == pytest.approx(1.0) for fields in protocol)
    assert any(
        fields.get("barge_in_to_interruption_report_ms") == pytest.approx(2.0)
        and fields.get("response_id") == 9
        and fields.get("output_id") == 10
        for fields in protocol
    )


def test_stage_durations_use_monotonic_time_and_shared_ids() -> None:
    clock = _Clock()
    logger = _RecordingLogger()
    observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=clock)
    observer.message(
        InputClosedEvent(ConversationId(1), InputId(1), 320, InputCloseReason.ENDPOINT),
        direction="outbound",
        phase="queued",
    )
    observer.message(
        ResponseStartedEvent(ConversationId(1), ResponseId(1), InputId(1), False),
        direction="outbound",
        phase="queued",
    )
    clock.now += 0.025
    observer.message(
        TranscriptFinalEvent(ConversationId(1), InputId(1), "safe"),
        direction="outbound",
        phase="queued",
    )
    output = OutputAudioEvent(ConversationId(1), ResponseId(1), OutputId(1), 0, b"\0\0")
    observer.message(output, direction="outbound", phase="queued")
    clock.now += 0.010
    observer.message(output, direction="inbound", phase="received")
    clock.now += 0.005
    observer.message(
        output,
        direction="inbound",
        phase="callback_dispatched",
    )

    fields = [record[2] for record in logger.records]
    assert any(item.get("endpoint_to_final_ms") == pytest.approx(25.0) for item in fields)
    assert any(item.get("endpoint_to_first_pcm_ms") == pytest.approx(25.0) for item in fields)
    callback_latency = next(
        item["pcm_receipt_to_callback_completion_ms"]
        for item in fields
        if "pcm_receipt_to_callback_completion_ms" in item
    )
    assert isinstance(callback_latency, float)
    assert abs(callback_latency - 5.0) < 1e-9
