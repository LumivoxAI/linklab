from enum import StrEnum
from typing import cast
from dataclasses import dataclass
from collections.abc import Callable

from ._enums import (
    ErrorCode,
    ErrorScope,
    CoarseState,
    InputAbortReason,
    InputCloseReason,
    InputStartReason,
    MessageDirection,
    PlaybackPosition,
    ResponseCancelReason,
    ConversationEndReason,
    PlaybackInterruptReason,
    ConversationCancelReason,
)
from ._config import AudioFormat, ConnectionLimits
from ._errors import CodecError
from ._values import InputId, OutputId, ResponseId, ConversationId
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
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

type Primitive = None | bool | int | str | bytes | list[Primitive] | dict[str, Primitive]
type _Decoder = Callable[[Primitive, str], object]
type _Encoder = Callable[[object, str], Primitive]

_MISSING = object()


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    decode: _Decoder
    encode: _Encoder
    optional: bool = False


@dataclass(frozen=True, slots=True)
class _Schema:
    direction: MessageDirection
    wire_type: str
    model_type: type[object]
    fields: tuple[_Field, ...]

    def construct(self, source: dict[str, Primitive]) -> Message:
        values: dict[str, object] = {}
        for field in self.fields:
            value = source.get(field.name, _MISSING)
            if value is _MISSING:
                if not field.optional:
                    raise CodecError(f"missing required field: {field.name}")
                values[field.name] = None
            else:
                values[field.name] = field.decode(cast(Primitive, value), field.name)
        return cast(Callable[..., Message], self.model_type)(**values)

    def serialize(self, message: Message) -> dict[str, Primitive]:
        result: dict[str, Primitive] = {"type": self.wire_type}
        for field in self.fields:
            value = getattr(message, field.name)
            if field.optional and value is None:
                continue
            result[field.name] = field.encode(value, field.name)
        # Semantic constructors remain an independent validation boundary.
        self.construct(result)
        return result


def _integer(value: Primitive, name: str) -> int:
    if type(value) is not int:
        raise CodecError(f"{name} must be an integer")
    return value


def _encode_integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise CodecError(f"{name} must be an integer")
    return value


def _boolean(value: Primitive, name: str) -> bool:
    if type(value) is not bool:
        raise CodecError(f"{name} must be a boolean")
    return value


def _encode_boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise CodecError(f"{name} must be a boolean")
    return value


def _string(value: Primitive, name: str) -> str:
    if type(value) is not str:
        raise CodecError(f"{name} must be a string")
    return value


def _encode_string(value: object, name: str) -> str:
    if type(value) is not str:
        raise CodecError(f"{name} must be a string")
    return value


def _binary(value: Primitive, name: str) -> bytes:
    if type(value) is not bytes:
        raise CodecError(f"{name} must be binary")
    return value


def _encode_binary(value: object, name: str) -> bytes:
    if type(value) is not bytes:
        raise CodecError(f"{name} must be binary")
    return value


def _id(kind: Callable[[int], object]) -> tuple[_Decoder, _Encoder]:
    def decode(value: Primitive, name: str) -> object:
        return kind(_integer(value, name))

    return decode, _encode_integer


def _enum[E: StrEnum](kind: type[E]) -> tuple[_Decoder, _Encoder]:
    def decode(value: Primitive, name: str) -> E:
        try:
            return kind(_string(value, name))
        except ValueError as error:
            raise CodecError(f"{name} has an unknown value") from error

    def encode(value: object, name: str) -> str:
        if not isinstance(value, kind):
            raise CodecError(f"{name} must be {kind.__name__}")
        return value.value

    return decode, encode


def _audio_format(value: Primitive, name: str) -> AudioFormat:
    if type(value) is not dict:
        raise CodecError(f"{name} must be a map")
    return AudioFormat(
        _string(value.get("encoding"), "encoding"),
        _integer(value.get("sample_rate_hz"), "sample_rate_hz"),
        _integer(value.get("channels"), "channels"),
    )


def _encode_audio_format(value: object, name: str) -> dict[str, Primitive]:
    if not isinstance(value, AudioFormat):
        raise CodecError(f"{name} must be an AudioFormat")
    return {"encoding": value.encoding, "sample_rate_hz": value.sample_rate_hz, "channels": value.channels}


def _limits(value: Primitive, name: str) -> ConnectionLimits:
    if type(value) is not dict:
        raise CodecError(f"{name} must be a map")
    return ConnectionLimits(
        max_message_bytes=_integer(value.get("max_message_bytes"), "max_message_bytes"),
        max_input_audio_frames=_integer(value.get("max_input_audio_frames"), "max_input_audio_frames"),
        max_output_audio_frames=_integer(value.get("max_output_audio_frames"), "max_output_audio_frames"),
        max_text_bytes=_integer(value.get("max_text_bytes"), "max_text_bytes"),
        max_input_frames=_integer(value.get("max_input_frames"), "max_input_frames"),
        idle_timeout_ms=_integer(value.get("idle_timeout_ms"), "idle_timeout_ms"),
    )


def _encode_limits(value: object, name: str) -> dict[str, Primitive]:
    if not isinstance(value, ConnectionLimits):
        raise CodecError(f"{name} must be ConnectionLimits")
    return {
        "max_message_bytes": value.max_message_bytes,
        "max_input_audio_frames": value.max_input_audio_frames,
        "max_output_audio_frames": value.max_output_audio_frames,
        "max_text_bytes": value.max_text_bytes,
        "max_input_frames": value.max_input_frames,
        "idle_timeout_ms": value.idle_timeout_ms,
    }


def _strings(value: Primitive, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise CodecError(f"{name} must be an array")
    return tuple(_string(item, f"{name} entry") for item in value)


def _encode_strings(value: object, name: str) -> list[Primitive]:
    if type(value) is not tuple or any(type(item) is not str for item in value):
        raise CodecError(f"{name} must be a tuple of strings")
    return list(value)


def _formats(value: Primitive, name: str) -> tuple[AudioFormat, ...]:
    if type(value) is not list:
        raise CodecError(f"{name} must be an array")
    return tuple(_audio_format(item, f"{name} entry") for item in value)


def _encode_formats(value: object, name: str) -> list[Primitive]:
    if type(value) is not tuple:
        raise CodecError(f"{name} must be a tuple")
    return [_encode_audio_format(item, f"{name} entry") for item in value]


def _field(name: str, adapters: tuple[_Decoder, _Encoder], *, optional: bool = False) -> _Field:
    return _Field(name, *adapters, optional)


INT = (_integer, _encode_integer)
BOOL = (_boolean, _encode_boolean)
STR = (_string, _encode_string)
BIN = (_binary, _encode_binary)
CID = _id(ConversationId)
IID = _id(InputId)
RID = _id(ResponseId)
OID = _id(OutputId)
FORMAT = (_audio_format, _encode_audio_format)
LIMITS = (_limits, _encode_limits)
STRINGS = (_strings, _encode_strings)
FORMATS = (_formats, _encode_formats)
F = _field
C2S = MessageDirection.CLIENT_TO_SERVER
S2C = MessageDirection.SERVER_TO_CLIENT

_SCHEMAS = (
    _Schema(
        C2S,
        "hello",
        ClientHello,
        (
            F("version", INT),
            F("capabilities", STRINGS),
            F("input_format", FORMAT),
            F("output_formats", FORMATS),
            F("agent", STR, optional=True),
        ),
    ),
    _Schema(
        C2S,
        "conversation.start",
        ConversationStartedEvent,
        (F("conversation_id", CID), F("activation", STR), F("wake_word", STR, optional=True)),
    ),
    _Schema(
        C2S,
        "input.start",
        InputStartedEvent,
        (
            F("conversation_id", CID),
            F("input_id", IID),
            F("reason", _enum(InputStartReason)),
            F("generation", INT),
            F("interrupts_response_id", RID, optional=True),
        ),
    ),
    _Schema(
        C2S,
        "input.audio",
        InputAudioEvent,
        (F("conversation_id", CID), F("input_id", IID), F("start_frame", INT), F("speech", BOOL), F("audio", BIN)),
    ),
    _Schema(
        C2S,
        "input.abort",
        InputAbortedEvent,
        (F("conversation_id", CID), F("input_id", IID), F("reason", _enum(InputAbortReason))),
    ),
    _Schema(
        C2S,
        "playback.finished",
        PlaybackFinishedEvent,
        (F("conversation_id", CID), F("response_id", RID), F("output_id", OID), F("played_frames", INT)),
    ),
    _Schema(
        C2S,
        "playback.interrupted",
        PlaybackInterruptedEvent,
        (
            F("conversation_id", CID),
            F("response_id", RID),
            F("output_id", OID),
            F("played_frames", INT),
            F("position", _enum(PlaybackPosition)),
            F("reason", _enum(PlaybackInterruptReason)),
        ),
    ),
    _Schema(
        C2S,
        "conversation.cancel",
        ConversationCancelledEvent,
        (F("conversation_id", CID), F("reason", _enum(ConversationCancelReason))),
    ),
    _Schema(
        S2C,
        "hello",
        ServerHello,
        (
            F("version", INT),
            F("capabilities", STRINGS),
            F("output_format", FORMAT),
            F("limits", LIMITS),
            F("agent", STR, optional=True),
        ),
    ),
    _Schema(
        S2C,
        "state",
        StateEvent,
        (
            F("conversation_id", CID),
            F("revision", INT),
            F("state", _enum(CoarseState)),
            F("reason", STR, optional=True),
        ),
    ),
    _Schema(
        S2C,
        "input.closed",
        InputClosedEvent,
        (
            F("conversation_id", CID),
            F("input_id", IID),
            F("accepted_end_frame", INT),
            F("reason", _enum(InputCloseReason)),
        ),
    ),
    _Schema(
        S2C,
        "transcript.update",
        TranscriptUpdateEvent,
        (
            F("conversation_id", CID),
            F("input_id", IID),
            F("revision", INT),
            F("text", STR),
            F("language", STR, optional=True),
        ),
    ),
    _Schema(
        S2C,
        "transcript.final",
        TranscriptFinalEvent,
        (F("conversation_id", CID), F("input_id", IID), F("text", STR), F("language", STR, optional=True)),
    ),
    _Schema(
        S2C,
        "response.start",
        ResponseStartedEvent,
        (F("conversation_id", CID), F("response_id", RID), F("input_id", IID), F("end_conversation", BOOL)),
    ),
    _Schema(
        S2C,
        "response.text.delta",
        ResponseTextDeltaEvent,
        (F("conversation_id", CID), F("response_id", RID), F("sequence", INT), F("text", STR)),
    ),
    _Schema(
        S2C,
        "response.text.final",
        ResponseTextFinalEvent,
        (F("conversation_id", CID), F("response_id", RID), F("text", STR)),
    ),
    _Schema(
        S2C, "output.start", OutputStartedEvent, (F("conversation_id", CID), F("response_id", RID), F("output_id", OID))
    ),
    _Schema(
        S2C,
        "output.audio",
        OutputAudioEvent,
        (F("conversation_id", CID), F("response_id", RID), F("output_id", OID), F("start_frame", INT), F("audio", BIN)),
    ),
    _Schema(
        S2C,
        "output.end",
        OutputEndedEvent,
        (F("conversation_id", CID), F("response_id", RID), F("output_id", OID), F("total_frames", INT)),
    ),
    _Schema(S2C, "response.end", ResponseEndedEvent, (F("conversation_id", CID), F("response_id", RID))),
    _Schema(
        S2C,
        "response.cancelled",
        ResponseCancelledEvent,
        (F("conversation_id", CID), F("response_id", RID), F("reason", _enum(ResponseCancelReason))),
    ),
    _Schema(
        S2C,
        "conversation.end",
        ConversationEndedEvent,
        (F("conversation_id", CID), F("reason", _enum(ConversationEndReason))),
    ),
    _Schema(
        S2C,
        "error",
        ErrorEvent,
        (
            F("scope", _enum(ErrorScope)),
            F("code", _enum(ErrorCode)),
            F("fatal", BOOL),
            F("conversation_id", CID, optional=True),
            F("input_id", IID, optional=True),
            F("response_id", RID, optional=True),
            F("message", STR, optional=True),
        ),
    ),
)

_BY_WIRE = {(schema.direction, schema.wire_type): schema for schema in _SCHEMAS}
_BY_MODEL = {schema.model_type: schema for schema in _SCHEMAS}


def decode_schema_message(source: dict[str, Primitive], direction: MessageDirection) -> Message:
    message_type = _string(source.get("type"), "type")
    schema = _BY_WIRE.get((direction, message_type))
    if schema is None:
        raise CodecError(f"unknown {direction.value} message type: {message_type}")
    return schema.construct(source)


def message_to_primitive(message: Message) -> dict[str, Primitive]:
    schema = _BY_MODEL.get(type(message))
    if schema is None:
        raise CodecError("message must be a wire Message value")
    return schema.serialize(message)


def known_message_types(direction: MessageDirection, *, include_handshake: bool = False) -> frozenset[str]:
    return frozenset(
        schema.wire_type
        for schema in _SCHEMAS
        if schema.direction is direction and (include_handshake or schema.wire_type not in {"hello", "error"})
    )
