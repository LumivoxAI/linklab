import re
import ssl
import math
import ipaddress
from dataclasses import field, dataclass

_ALLOWED_OUTPUT_RATES = (24_000, 48_000, 16_000)
_CLIENT_OUTPUT_ORDER = {rate: index for index, rate in enumerate(_ALLOWED_OUTPUT_RATES)}
_REQUIRED_CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
_SERVICE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")


def _require_int(name: str, value: object, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be in {minimum}{suffix}")


def _require_float(name: str, value: object, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value <= 0 or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be finite, positive{suffix}")


def _require_string(name: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a str")
    return value


def _require_agent(value: object) -> None:
    value = _require_string("agent", value)
    if not 1 <= len(value.encode("utf-8")) <= 64:
        raise ValueError("agent must contain 1..64 UTF-8 bytes")


def _require_ssl_context(value: object) -> None:
    if value is not None and not isinstance(value, ssl.SSLContext):
        raise TypeError("ssl_context must be an SSLContext or None")


def _require_service_id(value: object) -> None:
    if value is None:
        return
    value = _require_string("discovery_service_id", value)
    if _SERVICE_ID.fullmatch(value) is None:
        raise ValueError("discovery_service_id must be a lowercase ASCII DNS label")


def _is_loopback_host(host: str) -> bool:
    if host.rstrip(".").lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_output_formats(value: object, *, canonical_order: bool) -> None:
    if type(value) is not tuple:
        raise TypeError("output_formats must be a tuple")
    if not 1 <= len(value) <= 3:
        raise ValueError("output_formats must contain 1..3 formats")
    if not all(isinstance(item, AudioFormat) for item in value):
        raise TypeError("output_formats entries must be AudioFormat values")

    rates = tuple(item.sample_rate_hz for item in value)
    if len(set(rates)) != len(rates):
        raise ValueError("output_formats must not contain duplicate rates")
    if 16_000 not in rates:
        raise ValueError("output_formats must include 16000 Hz")
    if canonical_order and rates != tuple(sorted(rates, key=_CLIENT_OUTPUT_ORDER.__getitem__)):
        raise ValueError("client output_formats must use canonical preference order")


@dataclass(frozen=True, slots=True)
class AudioFormat:
    encoding: str
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        _require_string("encoding", self.encoding)
        if self.encoding != "pcm_s16le":
            raise ValueError("encoding must be pcm_s16le")
        _require_int("sample_rate_hz", self.sample_rate_hz, 1)
        if self.sample_rate_hz not in _ALLOWED_OUTPUT_RATES:
            raise ValueError("sample_rate_hz must be 16000, 24000, or 48000")
        _require_int("channels", self.channels, 1, 1)


@dataclass(frozen=True, slots=True)
class ConnectionLimits:
    max_message_bytes: int = 262_144
    max_input_audio_frames: int = 1_600
    max_output_audio_frames: int = 4_800
    max_text_bytes: int = 16_384
    max_input_frames: int = 1_920_000
    idle_timeout_ms: int = 120_000

    def __post_init__(self) -> None:
        _require_int("max_message_bytes", self.max_message_bytes, 16_384, 262_144)
        _require_int("max_input_audio_frames", self.max_input_audio_frames, 160, 1_600)
        _require_int("max_output_audio_frames", self.max_output_audio_frames, 160, 4_800)
        _require_int("max_text_bytes", self.max_text_bytes, 1_024, 65_536)
        _require_int("max_input_frames", self.max_input_frames, 16_000, 1_920_000)
        _require_int("idle_timeout_ms", self.idle_timeout_ms, 10_000, 600_000)


@dataclass(frozen=True, slots=True)
class ClientConfig:
    uri: str | None
    output_formats: tuple[AudioFormat, ...]
    discovery_service_id: str | None = None
    discovery_timeout_s: float = 10.0
    ssl_context: ssl.SSLContext | None = None
    input_queue_frames: int = 16_000
    playback_queue_ms: int = 2_000
    waiting_pre_roll_frames: int = 8_000
    connect_timeout_s: float = 10.0
    handshake_timeout_s: float = 5.0
    close_timeout_s: float = 10.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    websocket_max_queue: int = 16
    websocket_write_limit: int = 65_536
    reconnect: bool = False
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 30.0
    agent: str = "lumivox-linklab"

    def __post_init__(self) -> None:
        if self.uri is not None:
            _require_string("uri", self.uri)
        _require_service_id(self.discovery_service_id)
        if self.uri is None and self.discovery_service_id is None:
            raise ValueError("uri or discovery_service_id is required")
        _require_float("discovery_timeout_s", self.discovery_timeout_s)
        _validate_output_formats(self.output_formats, canonical_order=True)
        _require_ssl_context(self.ssl_context)
        _require_int("input_queue_frames", self.input_queue_frames, 1)
        _require_int("playback_queue_ms", self.playback_queue_ms, 1)
        _require_int("waiting_pre_roll_frames", self.waiting_pre_roll_frames, 0)
        _require_float("connect_timeout_s", self.connect_timeout_s, 10.0)
        _require_float("handshake_timeout_s", self.handshake_timeout_s, 5.0)
        _require_float("close_timeout_s", self.close_timeout_s, 10.0)
        _require_float("ping_interval_s", self.ping_interval_s, 20.0)
        _require_float("ping_timeout_s", self.ping_timeout_s, 20.0)
        _require_int("websocket_max_queue", self.websocket_max_queue, 1, 16)
        _require_int("websocket_write_limit", self.websocket_write_limit, 1, 65_536)
        if type(self.reconnect) is not bool:
            raise TypeError("reconnect must be a bool")
        _require_float("reconnect_initial_s", self.reconnect_initial_s)
        _require_float("reconnect_max_s", self.reconnect_max_s)
        if self.reconnect_initial_s > self.reconnect_max_s:
            raise ValueError("reconnect_initial_s must not exceed reconnect_max_s")
        _require_agent(self.agent)


@dataclass(frozen=True, slots=True)
class ServerConfig:
    port: int
    output_formats: tuple[AudioFormat, ...]
    host: str = "127.0.0.1"
    discovery_service_id: str | None = None
    ssl_context: ssl.SSLContext | None = None
    limits: ConnectionLimits = field(default_factory=ConnectionLimits)
    max_connections: int = 8
    input_queue_frames: int = 32_000
    output_queue_ms: int = 500
    waiting_timeout_s: float = 60.0
    input_timeout_s: float = 120.0
    processing_timeout_s: float = 120.0
    handshake_timeout_s: float = 5.0
    close_timeout_s: float = 10.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    websocket_max_queue: int = 16
    websocket_write_limit: int = 65_536
    agent: str = "lumivox-linklab"

    def __post_init__(self) -> None:
        _require_int("port", self.port, 1, 65_535)
        _validate_output_formats(self.output_formats, canonical_order=False)
        _require_string("host", self.host)
        _require_service_id(self.discovery_service_id)
        if self.discovery_service_id is not None and _is_loopback_host(self.host):
            raise ValueError("discovery cannot advertise a loopback-only host")
        _require_ssl_context(self.ssl_context)
        if not isinstance(self.limits, ConnectionLimits):
            raise TypeError("limits must be ConnectionLimits")
        minimum_output_cap = max(item.sample_rate_hz // 100 for item in self.output_formats)
        if self.limits.max_output_audio_frames < minimum_output_cap:
            raise ValueError("max_output_audio_frames is below a configured format's minimum")
        _require_int("max_connections", self.max_connections, 1)
        _require_int("input_queue_frames", self.input_queue_frames, 1)
        _require_int("output_queue_ms", self.output_queue_ms, 1)
        _require_float("waiting_timeout_s", self.waiting_timeout_s)
        _require_float("input_timeout_s", self.input_timeout_s)
        _require_float("processing_timeout_s", self.processing_timeout_s)
        _require_float("handshake_timeout_s", self.handshake_timeout_s, 5.0)
        _require_float("close_timeout_s", self.close_timeout_s, 10.0)
        _require_float("ping_interval_s", self.ping_interval_s, 20.0)
        _require_float("ping_timeout_s", self.ping_timeout_s, 20.0)
        _require_int("websocket_max_queue", self.websocket_max_queue, 1, 16)
        _require_int("websocket_write_limit", self.websocket_write_limit, 1, 65_536)
        _require_agent(self.agent)
