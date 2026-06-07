import ssl
from enum import StrEnum
from typing import get_type_hints
from dataclasses import FrozenInstanceError, fields, replace

import pytest

import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
PCM_24K = linklab.AudioFormat("pcm_s16le", 24_000, 1)
PCM_48K = linklab.AudioFormat("pcm_s16le", 48_000, 1)

ENUM_VALUES: dict[type[StrEnum], tuple[str, ...]] = {
    linklab.InputStartReason: ("activation", "speech", "barge_in"),
    linklab.InputAbortReason: ("discontinuity", "overflow", "capture_failed", "shutdown"),
    linklab.PlaybackPosition: ("exact", "estimated"),
    linklab.PlaybackInterruptReason: (
        "barge_in",
        "local_cancel",
        "playback_failed",
        "overflow",
        "shutdown",
    ),
    linklab.ConversationCancelReason: ("user", "shutdown", "client_failed"),
    linklab.CoarseState: ("waiting", "listening", "processing", "responding"),
    linklab.InputCloseReason: ("endpoint", "max_duration", "no_speech", "failed"),
    linklab.ResponseCancelReason: (
        "barge_in",
        "local_cancel",
        "conversation_cancelled",
        "generation_failed",
        "tts_failed",
        "playback_failed",
        "overflow",
        "shutdown",
    ),
    linklab.ConversationEndReason: (
        "completed",
        "cancelled",
        "idle_timeout",
        "client_failed",
        "server_failed",
        "playback_failed",
    ),
    linklab.ErrorScope: ("connection", "conversation", "input", "response"),
    linklab.ErrorCode: (
        "malformed_message",
        "message_too_large",
        "unknown_message",
        "unsupported_version",
        "capability_mismatch",
        "format_mismatch",
        "handshake_timeout",
        "protocol_state",
        "id_exhausted",
        "peer_unresponsive",
        "conversation_failed",
        "idle_timeout",
        "input_discontinuity",
        "input_overflow",
        "input_too_long",
        "capture_failed",
        "stt_failed",
        "processing_timeout",
        "generation_failed",
        "tts_failed",
        "playback_failed",
        "output_overflow",
        "response_cancelled",
    ),
    linklab.ConnectionState: ("disconnected", "handshaking", "ready", "closing"),
    linklab.MessageDirection: ("client_to_server", "server_to_client"),
    linklab.EndpointRole: ("client", "server"),
    linklab.ProtocolObjectKind: ("conversation", "input", "response", "output"),
    linklab.AudioSubmitResult: (
        "accepted",
        "ignored_inactive",
        "ignored_waiting_silence",
        "closed_input",
        "overflow",
    ),
}


@pytest.mark.parametrize(("enum_type", "expected"), ENUM_VALUES.items())
def test_enum_values(enum_type: type[StrEnum], expected: tuple[str, ...]) -> None:
    assert issubclass(enum_type, StrEnum)
    assert tuple(item.value for item in enum_type) == expected


@pytest.mark.parametrize(
    "exception_type",
    [
        linklab.CodecError,
        linklab.ProtocolViolation,
        linklab.ConnectionClosed,
        linklab.QueueOverflow,
        linklab.WriterClosed,
    ],
)
def test_exception_hierarchy_has_independent_leaves(exception_type: type[Exception]) -> None:
    assert exception_type.__bases__ == (linklab.LinklabError,)


def test_typed_id_factories_are_distinct() -> None:
    factories: tuple[object, ...] = (
        linklab.ConversationId,
        linklab.InputId,
        linklab.ResponseId,
        linklab.OutputId,
    )
    assert len({id(factory) for factory in factories}) == 4
    assert tuple(int(factory(1)) for factory in factories) == (1, 1, 1, 1)  # type: ignore[operator]


@pytest.mark.parametrize(
    "value",
    [
        linklab.AnnotatedAudio(b"\x00\x00", 0, False, True, True),
        linklab.ProtocolStateSnapshot(linklab.ConnectionState.READY, None, None, None, None, None, None),
        linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.CONVERSATION,
            linklab.ConversationId(1),
            None,
            None,
            None,
            "ended",
        ),
        PCM_16K,
        linklab.ConnectionLimits(),
        linklab.ClientConfig("ws://localhost", (PCM_16K,)),
        linklab.ServerConfig(9000, (PCM_16K,)),
    ],
)
def test_public_values_are_frozen_and_slotted(value: object) -> None:
    assert not hasattr(value, "__dict__")
    field_name = fields(value)[0].name  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, getattr(value, field_name))


@pytest.mark.parametrize(
    ("changes", "exception_type"),
    [
        ({"encoding": 1}, TypeError),
        ({"encoding": "float32"}, ValueError),
        ({"sample_rate_hz": True}, TypeError),
        ({"sample_rate_hz": 8_000}, ValueError),
        ({"channels": True}, TypeError),
        ({"channels": 2}, ValueError),
    ],
)
def test_audio_format_validation(changes: dict[str, object], exception_type: type[Exception]) -> None:
    with pytest.raises(exception_type):
        replace(PCM_16K, **changes)  # type: ignore[arg-type]


LIMIT_BOUNDARIES = {
    "max_message_bytes": (16_384, 262_144),
    "max_input_audio_frames": (160, 1_600),
    "max_output_audio_frames": (160, 4_800),
    "max_text_bytes": (1_024, 65_536),
    "max_input_frames": (16_000, 1_920_000),
    "idle_timeout_ms": (10_000, 600_000),
}


def test_connection_limit_defaults() -> None:
    assert linklab.ConnectionLimits() == linklab.ConnectionLimits(
        max_message_bytes=262_144,
        max_input_audio_frames=1_600,
        max_output_audio_frames=4_800,
        max_text_bytes=16_384,
        max_input_frames=1_920_000,
        idle_timeout_ms=120_000,
    )


@pytest.mark.parametrize(("name", "bounds"), LIMIT_BOUNDARIES.items())
def test_connection_limit_boundaries(name: str, bounds: tuple[int, int]) -> None:
    minimum, maximum = bounds
    assert getattr(replace(linklab.ConnectionLimits(), **{name: minimum}), name) == minimum
    assert getattr(replace(linklab.ConnectionLimits(), **{name: maximum}), name) == maximum
    with pytest.raises(ValueError):
        replace(linklab.ConnectionLimits(), **{name: minimum - 1})
    with pytest.raises(ValueError):
        replace(linklab.ConnectionLimits(), **{name: maximum + 1})
    with pytest.raises(TypeError):
        replace(linklab.ConnectionLimits(), **{name: True})


def _client(**changes: object) -> linklab.ClientConfig:
    values: dict[str, object] = {"uri": "ws://localhost", "output_formats": (PCM_24K, PCM_48K, PCM_16K)}
    values.update(changes)
    return linklab.ClientConfig(**values)  # type: ignore[arg-type]


def _server(**changes: object) -> linklab.ServerConfig:
    values: dict[str, object] = {"port": 9000, "output_formats": (PCM_48K, PCM_16K, PCM_24K)}
    values.update(changes)
    return linklab.ServerConfig(**values)  # type: ignore[arg-type]


def test_client_defaults() -> None:
    config = linklab.ClientConfig("ws://localhost", (PCM_16K,))
    assert config == linklab.ClientConfig(
        uri="ws://localhost",
        output_formats=(PCM_16K,),
        discovery_service_id=None,
        discovery_timeout_s=10.0,
        ssl_context=None,
        input_queue_frames=16_000,
        playback_queue_ms=2_000,
        waiting_pre_roll_frames=8_000,
        connect_timeout_s=10.0,
        handshake_timeout_s=5.0,
        close_timeout_s=10.0,
        ping_interval_s=20.0,
        ping_timeout_s=20.0,
        websocket_max_queue=16,
        websocket_write_limit=65_536,
        reconnect=False,
        reconnect_initial_s=0.5,
        reconnect_max_s=30.0,
        agent="lumivox-linklab",
    )


@pytest.mark.parametrize("formats", [(PCM_16K,), (PCM_24K, PCM_16K), (PCM_48K, PCM_16K), (PCM_24K, PCM_48K, PCM_16K)])
def test_client_accepts_canonical_output_format_subsets(formats: tuple[linklab.AudioFormat, ...]) -> None:
    assert linklab.ClientConfig("ws://localhost", formats).output_formats == formats


@pytest.mark.parametrize(
    "formats",
    [(), (PCM_24K,), (PCM_16K, PCM_24K), (PCM_48K, PCM_24K, PCM_16K), (PCM_16K, PCM_16K)],
)
def test_client_rejects_invalid_output_formats(formats: tuple[linklab.AudioFormat, ...]) -> None:
    with pytest.raises(ValueError):
        linklab.ClientConfig("ws://localhost", formats)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("uri", 1),
        ("discovery_service_id", "Production"),
        ("discovery_timeout_s", 0),
        ("output_formats", [PCM_16K]),
        ("ssl_context", object()),
        ("input_queue_frames", 0),
        ("playback_queue_ms", 0),
        ("waiting_pre_roll_frames", -1),
        ("connect_timeout_s", 10.1),
        ("handshake_timeout_s", 5.1),
        ("close_timeout_s", 10.1),
        ("ping_interval_s", 20.1),
        ("ping_timeout_s", 20.1),
        ("websocket_max_queue", 17),
        ("websocket_write_limit", 65_537),
        ("reconnect", 1),
        ("reconnect_initial_s", 0.0),
        ("reconnect_max_s", float("inf")),
        ("agent", ""),
    ],
)
def test_client_rejects_invalid_fields(name: str, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _client(**{name: invalid})


@pytest.mark.parametrize(
    "name",
    [
        "input_queue_frames",
        "playback_queue_ms",
        "waiting_pre_roll_frames",
        "websocket_max_queue",
        "websocket_write_limit",
    ],
)
def test_client_integer_fields_reject_bool(name: str) -> None:
    with pytest.raises(TypeError):
        _client(**{name: True})


@pytest.mark.parametrize(
    "name",
    [
        "connect_timeout_s",
        "discovery_timeout_s",
        "handshake_timeout_s",
        "close_timeout_s",
        "ping_interval_s",
        "ping_timeout_s",
        "reconnect_initial_s",
        "reconnect_max_s",
    ],
)
def test_client_float_fields_reject_bool(name: str) -> None:
    with pytest.raises(TypeError):
        _client(**{name: True})


def test_client_float_fields_accept_integers() -> None:
    assert _client(connect_timeout_s=1).connect_timeout_s == 1


def test_client_reconnect_range_relation() -> None:
    with pytest.raises(ValueError):
        _client(reconnect_initial_s=2.0, reconnect_max_s=1.0)


def test_discovery_configuration_invariants() -> None:
    assert _client(uri=None, discovery_service_id="production").uri is None
    assert _client(uri="ws://direct", discovery_service_id="ignored").uri == "ws://direct"
    with pytest.raises(ValueError, match="uri or discovery_service_id"):
        _client(uri=None)
    for invalid in ("", "-bad", "UPPER", "a" * 64, "bad.name", 1):
        with pytest.raises((TypeError, ValueError)):
            _client(discovery_service_id=invalid)
    for accepted in ("a", "production-2", "a" * 63):
        assert _client(discovery_service_id=accepted).discovery_service_id == accepted

    for host in ("127.0.0.1", "127.1.2.3", "::1", "localhost", "LOCALHOST."):
        with pytest.raises(ValueError, match="loopback"):
            _server(host=host, discovery_service_id="production")
    assert _server(host="0.0.0.0", discovery_service_id="production")
    assert _server(host="192.0.2.1", discovery_service_id="production")


def test_config_accepts_ssl_context() -> None:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    assert _client(ssl_context=context).ssl_context is context
    assert _server(ssl_context=context).ssl_context is context


def test_server_defaults() -> None:
    config = linklab.ServerConfig(9000, (PCM_16K,))
    assert config.port == 9000
    assert config.output_formats == (PCM_16K,)
    assert config.host == "127.0.0.1"
    assert config.discovery_service_id is None
    assert config.ssl_context is None
    assert config.limits == linklab.ConnectionLimits()
    assert config.max_connections == 8
    assert config.input_queue_frames == 32_000
    assert config.output_queue_ms == 500
    assert config.waiting_timeout_s == 60.0
    assert config.input_timeout_s == 120.0
    assert config.processing_timeout_s == 120.0
    assert config.handshake_timeout_s == 5.0
    assert config.close_timeout_s == 10.0
    assert config.ping_interval_s == 20.0
    assert config.ping_timeout_s == 20.0
    assert config.websocket_max_queue == 16
    assert config.websocket_write_limit == 65_536
    assert config.agent == "lumivox-linklab"


def test_server_accepts_any_unique_supported_format_order() -> None:
    assert _server().output_formats == (PCM_48K, PCM_16K, PCM_24K)


@pytest.mark.parametrize("formats", [(), (PCM_24K,), (PCM_16K, PCM_16K)])
def test_server_rejects_invalid_output_formats(formats: tuple[linklab.AudioFormat, ...]) -> None:
    with pytest.raises(ValueError):
        linklab.ServerConfig(9000, formats)


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("port", 0),
        ("port", 65_536),
        ("port", True),
        ("output_formats", [PCM_16K]),
        ("host", 1),
        ("discovery_service_id", "bad_name"),
        ("ssl_context", object()),
        ("limits", object()),
        ("max_connections", 0),
        ("input_queue_frames", 0),
        ("output_queue_ms", 0),
        ("waiting_timeout_s", 0.0),
        ("input_timeout_s", float("nan")),
        ("processing_timeout_s", -1.0),
        ("handshake_timeout_s", 5.1),
        ("close_timeout_s", 10.1),
        ("ping_interval_s", 20.1),
        ("ping_timeout_s", 20.1),
        ("websocket_max_queue", 17),
        ("websocket_write_limit", 65_537),
        ("agent", "x" * 65),
    ],
)
def test_server_rejects_invalid_fields(name: str, invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _server(**{name: invalid})


@pytest.mark.parametrize(
    "name",
    ["max_connections", "input_queue_frames", "output_queue_ms", "websocket_max_queue", "websocket_write_limit"],
)
def test_server_integer_fields_reject_bool(name: str) -> None:
    with pytest.raises(TypeError):
        _server(**{name: True})


@pytest.mark.parametrize(
    "name",
    [
        "waiting_timeout_s",
        "input_timeout_s",
        "processing_timeout_s",
        "handshake_timeout_s",
        "close_timeout_s",
        "ping_interval_s",
        "ping_timeout_s",
    ],
)
def test_server_float_fields_reject_bool(name: str) -> None:
    with pytest.raises(TypeError):
        _server(**{name: True})


def test_server_float_fields_accept_integers() -> None:
    assert _server(waiting_timeout_s=1).waiting_timeout_s == 1


@pytest.mark.parametrize(("rate", "minimum"), [(16_000, 160), (24_000, 240), (48_000, 480)])
def test_server_output_cap_uses_each_selected_rate_minimum(rate: int, minimum: int) -> None:
    selected = linklab.AudioFormat("pcm_s16le", rate, 1)
    formats = (selected,) if rate == 16_000 else (selected, PCM_16K)
    assert linklab.ServerConfig(9000, formats, limits=linklab.ConnectionLimits(max_output_audio_frames=minimum))
    with pytest.raises(ValueError):
        linklab.ServerConfig(9000, formats, limits=linklab.ConnectionLimits(max_output_audio_frames=minimum - 1))


def test_every_client_config_field_accepts_exact_bounds_and_rejects_adjacent_wrong_types_nan_inf() -> None:
    accepted: object
    invalid: object
    name: str
    maximum: int | float
    assert tuple(field.name for field in fields(linklab.ClientConfig)) == (
        "uri",
        "output_formats",
        "discovery_service_id",
        "discovery_timeout_s",
        "ssl_context",
        "input_queue_frames",
        "playback_queue_ms",
        "waiting_pre_roll_frames",
        "connect_timeout_s",
        "handshake_timeout_s",
        "close_timeout_s",
        "ping_interval_s",
        "ping_timeout_s",
        "websocket_max_queue",
        "websocket_write_limit",
        "reconnect",
        "reconnect_initial_s",
        "reconnect_max_s",
        "agent",
    )
    assert get_type_hints(linklab.ClientConfig) == {
        "uri": str | None,
        "output_formats": tuple[linklab.AudioFormat, ...],
        "discovery_service_id": str | None,
        "discovery_timeout_s": float,
        "ssl_context": ssl.SSLContext | None,
        "input_queue_frames": int,
        "playback_queue_ms": int,
        "waiting_pre_roll_frames": int,
        "connect_timeout_s": float,
        "handshake_timeout_s": float,
        "close_timeout_s": float,
        "ping_interval_s": float,
        "ping_timeout_s": float,
        "websocket_max_queue": int,
        "websocket_write_limit": int,
        "reconnect": bool,
        "reconnect_initial_s": float,
        "reconnect_max_s": float,
        "agent": str,
    }
    assert _client(uri="").uri == ""
    assert _client(discovery_service_id="production").discovery_service_id == "production"
    assert _client(discovery_timeout_s=0.001).discovery_timeout_s == 0.001
    assert _client(output_formats=(PCM_16K,)).output_formats == (PCM_16K,)
    assert _client(ssl_context=None).ssl_context is None
    for name in ("input_queue_frames", "playback_queue_ms"):
        assert getattr(_client(**{name: 1}), name) == 1
        with pytest.raises(ValueError):
            _client(**{name: 0})
        with pytest.raises(TypeError):
            _client(**{name: True})
    assert _client(waiting_pre_roll_frames=0).waiting_pre_roll_frames == 0
    with pytest.raises(ValueError):
        _client(waiting_pre_roll_frames=-1)
    with pytest.raises(TypeError):
        _client(waiting_pre_roll_frames=True)

    bounded_floats = {
        "connect_timeout_s": 10.0,
        "handshake_timeout_s": 5.0,
        "close_timeout_s": 10.0,
        "ping_interval_s": 20.0,
        "ping_timeout_s": 20.0,
    }
    for name, maximum in bounded_floats.items():
        assert getattr(_client(**{name: maximum}), name) == maximum
        assert getattr(_client(**{name: 0.001}), name) == 0.001
        for invalid in (0, maximum + 0.001, float("nan"), float("inf"), True, "1"):
            with pytest.raises((TypeError, ValueError)):
                _client(**{name: invalid})
    for name, maximum in (("websocket_max_queue", 16), ("websocket_write_limit", 65_536)):
        for accepted in (1, maximum):
            assert getattr(_client(**{name: accepted}), name) == accepted
        for invalid in (0, maximum + 1, True, 1.0):
            with pytest.raises((TypeError, ValueError)):
                _client(**{name: invalid})
    for accepted in (False, True):
        assert _client(reconnect=accepted).reconnect is accepted
    for invalid in (0, 1, None):
        with pytest.raises(TypeError):
            _client(reconnect=invalid)
    for name in ("reconnect_initial_s", "reconnect_max_s"):
        changes = {name: 0.001}
        if name == "reconnect_max_s":
            changes["reconnect_initial_s"] = 0.001
        assert getattr(_client(**changes), name) == 0.001
        for invalid in (0, -1, float("nan"), float("inf"), True, "1"):
            with pytest.raises((TypeError, ValueError)):
                _client(**{name: invalid})
    assert _client(reconnect_initial_s=1.0, reconnect_max_s=1.0)
    with pytest.raises(ValueError):
        _client(reconnect_initial_s=1.001, reconnect_max_s=1.0)
    for accepted in ("x", "é" * 32):
        assert _client(agent=accepted).agent == accepted
    for invalid in ("", "é" * 32 + "x", 1):
        with pytest.raises((TypeError, ValueError)):
            _client(agent=invalid)


def test_every_server_config_field_accepts_exact_bounds_and_rejects_adjacent_wrong_types_nan_inf() -> None:
    accepted: object
    invalid: object
    name: str
    maximum: int | float
    assert tuple(field.name for field in fields(linklab.ServerConfig)) == (
        "port",
        "output_formats",
        "host",
        "discovery_service_id",
        "ssl_context",
        "limits",
        "max_connections",
        "input_queue_frames",
        "output_queue_ms",
        "waiting_timeout_s",
        "input_timeout_s",
        "processing_timeout_s",
        "handshake_timeout_s",
        "close_timeout_s",
        "ping_interval_s",
        "ping_timeout_s",
        "websocket_max_queue",
        "websocket_write_limit",
        "agent",
    )
    assert get_type_hints(linklab.ServerConfig) == {
        "port": int,
        "output_formats": tuple[linklab.AudioFormat, ...],
        "host": str,
        "discovery_service_id": str | None,
        "ssl_context": ssl.SSLContext | None,
        "limits": linklab.ConnectionLimits,
        "max_connections": int,
        "input_queue_frames": int,
        "output_queue_ms": int,
        "waiting_timeout_s": float,
        "input_timeout_s": float,
        "processing_timeout_s": float,
        "handshake_timeout_s": float,
        "close_timeout_s": float,
        "ping_interval_s": float,
        "ping_timeout_s": float,
        "websocket_max_queue": int,
        "websocket_write_limit": int,
        "agent": str,
    }
    for accepted in (1, 65_535):
        assert _server(port=accepted).port == accepted
    for invalid in (0, 65_536, True, 1.0):
        with pytest.raises((TypeError, ValueError)):
            _server(port=invalid)
    assert _server(output_formats=(PCM_16K,)).output_formats == (PCM_16K,)
    assert _server(host="").host == ""
    with pytest.raises(TypeError):
        _server(host=1)
    assert _server(ssl_context=None).ssl_context is None
    assert _server(limits=linklab.ConnectionLimits()).limits == linklab.ConnectionLimits()
    with pytest.raises(TypeError):
        _server(limits=object())

    for name in ("max_connections", "input_queue_frames", "output_queue_ms"):
        assert getattr(_server(**{name: 1}), name) == 1
        for invalid in (0, True, 1.0):
            with pytest.raises((TypeError, ValueError)):
                _server(**{name: invalid})
    unbounded_floats = ("waiting_timeout_s", "input_timeout_s", "processing_timeout_s")
    bounded_floats = {
        "handshake_timeout_s": 5.0,
        "close_timeout_s": 10.0,
        "ping_interval_s": 20.0,
        "ping_timeout_s": 20.0,
    }
    for name in unbounded_floats:
        assert getattr(_server(**{name: 0.001}), name) == 0.001
        for invalid in (0, -1, float("nan"), float("inf"), True, "1"):
            with pytest.raises((TypeError, ValueError)):
                _server(**{name: invalid})
    for name, maximum in bounded_floats.items():
        for accepted in (0.001, maximum):
            assert getattr(_server(**{name: accepted}), name) == accepted
        for invalid in (0, maximum + 0.001, float("nan"), float("inf"), True, "1"):
            with pytest.raises((TypeError, ValueError)):
                _server(**{name: invalid})
    for name, maximum in (("websocket_max_queue", 16), ("websocket_write_limit", 65_536)):
        for accepted in (1, maximum):
            assert getattr(_server(**{name: accepted}), name) == accepted
        for invalid in (0, maximum + 1, True, 1.0):
            with pytest.raises((TypeError, ValueError)):
                _server(**{name: invalid})
    for accepted in ("x", "é" * 32):
        assert _server(agent=accepted).agent == accepted
    for invalid in ("", "é" * 32 + "x", 1):
        with pytest.raises((TypeError, ValueError)):
            _server(agent=invalid)
