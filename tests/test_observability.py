from __future__ import annotations

from typing import Self

import pytest

from lumivox_linklab import (
    InputId,
    OutputId,
    ErrorCode,
    ErrorEvent,
    ErrorScope,
    ResponseId,
    AudioFormat,
    ClientConfig,
    ConversationId,
    InputAudioEvent,
    InputClosedEvent,
    InputCloseReason,
    OutputAudioEvent,
    ResponseStartedEvent,
    TranscriptFinalEvent,
)
from lumivox_linklab._client import _CallbackQueue
from lumivox_linklab._server import _HandlerQueue
from lumivox_linklab._transport import _QueueLane, _BoundedBatchQueue
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

    def bind(self, **new_values: object) -> Self:
        return type(self)(self.records, {**self.context, **new_values})

    def debug(self, event: str, **kwargs: object) -> None:
        self._record("debug", event, kwargs)

    def info(self, event: str, **kwargs: object) -> None:
        self._record("info", event, kwargs)

    def warning(self, event: str, **kwargs: object) -> None:
        self._record("warning", event, kwargs)

    def error(self, event: str, **kwargs: object) -> None:
        self._record("error", event, kwargs)

    def exception(self, event: str, **kwargs: object) -> None:
        self._record("exception", event, kwargs)

    def _record(self, level: str, event: str, fields: dict[str, object]) -> None:
        self.records.append((level, event, {**self.context, **fields}))


class _RaisingLogger:
    def bind(self, **_new_values: object) -> Self:
        raise RuntimeError("logger bind failed")

    def debug(self, _event: str, **_kwargs: object) -> None:
        raise RuntimeError("logger debug failed")

    def info(self, _event: str, **_kwargs: object) -> None:
        raise RuntimeError("logger info failed")

    def warning(self, _event: str, **_kwargs: object) -> None:
        raise RuntimeError("logger warning failed")

    def error(self, _event: str, **_kwargs: object) -> None:
        raise RuntimeError("logger error failed")

    def exception(self, _event: str, **_kwargs: object) -> None:
        raise RuntimeError("logger exception failed")


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


def test_logger_exceptions_cannot_change_observed_protocol_flow() -> None:
    observer = _Observer(_RaisingLogger(), endpoint_role="client", connection_id="client-1", clock=lambda: 1.0)
    observer.lifecycle("ready", state="ready")
    observer.queue_snapshots((_QueueSnapshot("queue", 1, 1, 1, 2.0, "events"),))
    observer.message(
        OutputAudioEvent(ConversationId(1), ResponseId(1), OutputId(1), 0, b"\0\0"),
        direction="inbound",
        phase="received",
    )
    observer.failure("failed", RuntimeError("payload must not escape"))
    observer.rtt(0.01)


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
        item["pcm_receipt_to_callback_ms"] for item in fields if "pcm_receipt_to_callback_ms" in item
    )
    assert isinstance(callback_latency, float)
    assert abs(callback_latency - 5.0) < 1e-9
