from enum import StrEnum
from typing import Never, cast

import msgpack  # type: ignore[import-untyped]

from ._enums import (
    ErrorCode,
    MessageDirection,
)
from ._config import AudioFormat, ConnectionLimits
from ._errors import CodecError
from ._schema import known_message_types, message_to_primitive, decode_schema_message
from ._values import InputId, OutputId, ResponseId, ConversationId, ReadableBuffer
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

_MAX_MESSAGE_BYTES = 262_144
_MAX_FIRST_MESSAGE_BYTES = 16_384
_MAX_DEPTH = 8
_MAX_MAP_ITEMS = 64
_MAX_ARRAY_ITEMS = 32
_MAX_STRING_BYTES = 65_536
_MAX_BINARY_BYTES = 16_384
_MISSING = object()

type Primitive = None | bool | int | str | bytes | list[Primitive] | dict[str, Primitive]


class _StructuralError(Exception):
    pass


class _OperationalLimitError(CodecError):
    pass


class _InboundMessageError(CodecError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_extension(_code: int, _data: bytes) -> Never:
    raise _StructuralError("MessagePack extensions are not allowed")


def _build_map(pairs: list[tuple[object, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str:
            raise _StructuralError("MessagePack map keys must be strings")
        if key in result:
            raise _StructuralError("MessagePack map keys must be unique")
        result[key] = value
    return result


def _as_bytes_view(data: ReadableBuffer) -> memoryview:
    try:
        view = memoryview(data)
    except TypeError as error:
        raise CodecError("message must be a readable buffer") from error
    if not view.contiguous:
        raise CodecError("message buffer must be contiguous")
    try:
        return view.cast("B")
    except TypeError as error:
        raise CodecError("message buffer must be byte-addressable") from error


def _validate_tree(root: object) -> dict[str, Primitive]:
    if type(root) is not dict:
        raise _StructuralError("MessagePack root must be a map")

    stack: list[tuple[object, int]] = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        if type(value) is dict:
            if depth > _MAX_DEPTH:
                raise _StructuralError("MessagePack nesting exceeds the limit")
            for child in value.values():
                if type(child) is dict or type(child) is list:
                    stack.append((child, depth + 1))
                elif child is not None and type(child) not in (bool, int, str, bytes):
                    raise _StructuralError("MessagePack value type is not allowed")
        elif type(value) is list:
            if depth > _MAX_DEPTH:
                raise _StructuralError("MessagePack nesting exceeds the limit")
            for child in value:
                if type(child) is dict or type(child) is list:
                    stack.append((child, depth + 1))
                elif child is not None and type(child) not in (bool, int, str, bytes):
                    raise _StructuralError("MessagePack value type is not allowed")
        else:
            raise _StructuralError("MessagePack value type is not allowed")

    return cast(dict[str, Primitive], root)


def _decode_primitive_message(
    data: ReadableBuffer,
    *,
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
    first_message: bool = False,
) -> dict[str, Primitive]:
    """Decode one bounded MessagePack map without applying a message schema."""
    view = _as_bytes_view(data)
    effective_limit = min(max_message_bytes, _MAX_MESSAGE_BYTES)
    if first_message:
        effective_limit = min(effective_limit, _MAX_FIRST_MESSAGE_BYTES)
    if len(view) > effective_limit:
        raise CodecError(f"message exceeds the {effective_limit}-byte envelope")

    try:
        unpacked = msgpack.unpackb(
            view,
            raw=False,
            use_list=True,
            strict_map_key=True,
            object_pairs_hook=_build_map,
            ext_hook=_reject_extension,
            max_str_len=_MAX_STRING_BYTES,
            max_bin_len=_MAX_BINARY_BYTES,
            max_array_len=_MAX_ARRAY_ITEMS,
            max_map_len=_MAX_MAP_ITEMS,
            max_ext_len=0,
        )
        return _validate_tree(unpacked)
    except CodecError:
        raise
    except (ValueError, TypeError, UnicodeError, msgpack.UnpackException, _StructuralError) as error:
        raise CodecError("invalid restricted MessagePack message") from error


def _field(message: dict[str, Primitive], name: str, *, optional: bool = False) -> Primitive | object:
    if name in message:
        return message[name]
    if optional:
        return _MISSING
    raise CodecError(f"missing required field: {name}")


def _integer(message: dict[str, Primitive], name: str) -> int:
    value = _field(message, name)
    if type(value) is not int:
        raise CodecError(f"{name} must be an integer")
    return value


def _boolean(message: dict[str, Primitive], name: str) -> bool:
    value = _field(message, name)
    if type(value) is not bool:
        raise CodecError(f"{name} must be a boolean")
    return value


def _string(message: dict[str, Primitive], name: str, *, optional: bool = False) -> str | None:
    value = _field(message, name, optional=optional)
    if value is _MISSING:
        return None
    if type(value) is not str:
        raise CodecError(f"{name} must be a string")
    return value


def _binary(message: dict[str, Primitive], name: str) -> bytes:
    value = _field(message, name)
    if type(value) is not bytes:
        raise CodecError(f"{name} must be binary")
    return value


def _map(message: dict[str, Primitive], name: str) -> dict[str, Primitive]:
    value = _field(message, name)
    if type(value) is not dict:
        raise CodecError(f"{name} must be a map")
    return value


def _array(message: dict[str, Primitive], name: str) -> list[Primitive]:
    value = _field(message, name)
    if type(value) is not list:
        raise CodecError(f"{name} must be an array")
    return value


def _audio_format_from_primitive(value: dict[str, Primitive]) -> AudioFormat:
    encoding = _string(value, "encoding")
    assert encoding is not None
    return AudioFormat(encoding, _integer(value, "sample_rate_hz"), _integer(value, "channels"))


def _connection_limits_from_primitive(value: dict[str, Primitive]) -> ConnectionLimits:
    return ConnectionLimits(
        max_message_bytes=_integer(value, "max_message_bytes"),
        max_input_audio_frames=_integer(value, "max_input_audio_frames"),
        max_output_audio_frames=_integer(value, "max_output_audio_frames"),
        max_text_bytes=_integer(value, "max_text_bytes"),
        max_input_frames=_integer(value, "max_input_frames"),
        idle_timeout_ms=_integer(value, "idle_timeout_ms"),
    )


# Kept as small nested-structure adapters for the handshake classifier and the
# schema migration boundary; message field metadata lives in _schema.
_audio_format = _audio_format_from_primitive
_connection_limits = _connection_limits_from_primitive


def _capabilities(message: dict[str, Primitive]) -> tuple[str, ...]:
    return _string_tuple_from_primitive(_field(message, "capabilities"), "capabilities")  # type: ignore[arg-type]


def _output_formats(message: dict[str, Primitive]) -> tuple[AudioFormat, ...]:
    return _audio_formats_from_primitive(_field(message, "output_formats"), "output_formats")  # type: ignore[arg-type]


def _conversation_id(message: dict[str, Primitive]) -> ConversationId:
    return ConversationId(_integer(message, "conversation_id"))


def _input_id(message: dict[str, Primitive]) -> InputId:
    return InputId(_integer(message, "input_id"))


def _response_id(message: dict[str, Primitive]) -> ResponseId:
    return ResponseId(_integer(message, "response_id"))


def _output_id(message: dict[str, Primitive]) -> OutputId:
    return OutputId(_integer(message, "output_id"))


def _enum[E: StrEnum](message: dict[str, Primitive], name: str, enum_type: type[E]) -> E:
    value = _string(message, name)
    assert value is not None
    try:
        return enum_type(value)
    except ValueError as error:
        raise CodecError(f"{name} has an unknown value") from error


def _string_tuple_from_primitive(value: Primitive, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise CodecError(f"{name} must be an array")
    result: list[str] = []
    for item in value:
        if type(item) is not str:
            raise CodecError(f"{name} entries must be strings")
        result.append(item)
    return tuple(result)


def _audio_formats_from_primitive(value: Primitive, name: str) -> tuple[AudioFormat, ...]:
    if type(value) is not list:
        raise CodecError(f"{name} must be an array")
    result: list[AudioFormat] = []
    for item in value:
        if type(item) is not dict:
            raise CodecError(f"{name} entries must be maps")
        result.append(_audio_format_from_primitive(item))
    return tuple(result)


def _audio_format_to_primitive(audio_format: AudioFormat) -> dict[str, Primitive]:
    if not isinstance(audio_format, AudioFormat):
        raise CodecError("audio format field must be an AudioFormat")
    return {
        "encoding": audio_format.encoding,
        "sample_rate_hz": audio_format.sample_rate_hz,
        "channels": audio_format.channels,
    }


def _connection_limits_to_primitive(limits: ConnectionLimits) -> dict[str, Primitive]:
    if not isinstance(limits, ConnectionLimits):
        raise CodecError("limits field must be ConnectionLimits")
    return {
        "max_message_bytes": limits.max_message_bytes,
        "max_input_audio_frames": limits.max_input_audio_frames,
        "max_output_audio_frames": limits.max_output_audio_frames,
        "max_text_bytes": limits.max_text_bytes,
        "max_input_frames": limits.max_input_frames,
        "idle_timeout_ms": limits.idle_timeout_ms,
    }


_audio_format_primitive = _audio_format_to_primitive
_connection_limits_primitive = _connection_limits_to_primitive


def _validate_field_names_and_unknown_binary(root: dict[str, Primitive]) -> None:
    stack: list[Primitive] = [root]
    while stack:
        value = stack.pop()
        if type(value) is dict:
            for key, child in value.items():
                if not key.isascii():
                    raise CodecError("field names must be ASCII")
                stack.append(child)
        elif type(value) is list:
            stack.extend(value)
        elif type(value) is bytes and len(value) > _MAX_BINARY_BYTES:
            raise CodecError("non-audio binary exceeds the absolute limit")


def _message_to_primitive(message: Message) -> dict[str, Primitive]:
    result: dict[str, Primitive]
    if type(message) is ClientHello:
        result = {
            "type": message.type,
            "version": message.version,
            "capabilities": list(message.capabilities),
            "input_format": _audio_format_primitive(message.input_format),
            "output_formats": [_audio_format_primitive(value) for value in message.output_formats],
        }
        if message.agent is not None:
            result["agent"] = message.agent
    elif type(message) is ServerHello:
        result = {
            "type": message.type,
            "version": message.version,
            "capabilities": list(message.capabilities),
            "output_format": _audio_format_primitive(message.output_format),
            "limits": _connection_limits_primitive(message.limits),
        }
        if message.agent is not None:
            result["agent"] = message.agent
    elif type(message) is ConversationStartedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "activation": message.activation,
        }
        if message.wake_word is not None:
            result["wake_word"] = message.wake_word
    elif type(message) is InputStartedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "reason": message.reason.value,
            "generation": message.generation,
        }
        if message.interrupts_response_id is not None:
            result["interrupts_response_id"] = message.interrupts_response_id
    elif type(message) is InputAudioEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "start_frame": message.start_frame,
            "speech": message.speech,
            "audio": message.audio,
        }
    elif type(message) is InputAbortedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "reason": message.reason.value,
        }
    elif type(message) is PlaybackFinishedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "output_id": message.output_id,
            "played_frames": message.played_frames,
        }
    elif type(message) is PlaybackInterruptedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "output_id": message.output_id,
            "played_frames": message.played_frames,
            "position": message.position.value,
            "reason": message.reason.value,
        }
    elif type(message) is ConversationCancelledEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "reason": message.reason.value,
        }
    elif type(message) is StateEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "revision": message.revision,
            "state": message.state.value,
        }
        if message.reason is not None:
            result["reason"] = message.reason
    elif type(message) is InputClosedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "accepted_end_frame": message.accepted_end_frame,
            "reason": message.reason.value,
        }
    elif type(message) is TranscriptUpdateEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "revision": message.revision,
            "text": message.text,
        }
        if message.language is not None:
            result["language"] = message.language
    elif type(message) is TranscriptFinalEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "input_id": message.input_id,
            "text": message.text,
        }
        if message.language is not None:
            result["language"] = message.language
    elif type(message) is ResponseStartedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "input_id": message.input_id,
            "end_conversation": message.end_conversation,
        }
    elif type(message) is ResponseTextDeltaEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "sequence": message.sequence,
            "text": message.text,
        }
    elif type(message) is ResponseTextFinalEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "text": message.text,
        }
    elif type(message) is OutputStartedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "output_id": message.output_id,
        }
    elif type(message) is OutputAudioEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "output_id": message.output_id,
            "start_frame": message.start_frame,
            "audio": message.audio,
        }
    elif type(message) is OutputEndedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "output_id": message.output_id,
            "total_frames": message.total_frames,
        }
    elif type(message) is ResponseEndedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
        }
    elif type(message) is ResponseCancelledEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "response_id": message.response_id,
            "reason": message.reason.value,
        }
    elif type(message) is ConversationEndedEvent:
        result = {
            "type": message.type,
            "conversation_id": message.conversation_id,
            "reason": message.reason.value,
        }
    elif type(message) is ErrorEvent:
        result = {
            "type": message.type,
            "scope": message.scope.value,
            "code": message.code.value,
            "fatal": message.fatal,
        }
        if message.conversation_id is not None:
            result["conversation_id"] = message.conversation_id
        if message.input_id is not None:
            result["input_id"] = message.input_id
        if message.response_id is not None:
            result["response_id"] = message.response_id
        if message.message is not None:
            result["message"] = message.message
    else:
        raise CodecError("message must be a wire Message value")
    return result


def _canonicalize(value: Primitive) -> Primitive:
    if type(value) is dict:
        return {
            key: _canonicalize(child) for key, child in sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
        }
    if type(value) is list:
        return [_canonicalize(child) for child in value]
    return value


def _pack_message(message: Message) -> bytes:
    try:
        primitive = _canonicalize(cast(Primitive, message_to_primitive(message)))
        return cast(bytes, msgpack.packb(primitive, use_bin_type=True, strict_types=True))
    except CodecError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise CodecError("message does not satisfy its schema") from error


def _enforce_encoded_size(encoded: bytes, message: Message, max_message_bytes: int) -> None:
    effective_limit = min(max_message_bytes, _MAX_MESSAGE_BYTES)
    if isinstance(message, ClientHello | ServerHello):
        effective_limit = min(effective_limit, _MAX_FIRST_MESSAGE_BYTES)
    if len(encoded) > effective_limit:
        raise CodecError(f"encoded message exceeds the {effective_limit}-byte envelope")


def encode_message(message: Message) -> bytes:
    """Schema-validate and canonically encode one message with absolute limits."""
    encoded = _pack_message(message)
    _enforce_encoded_size(encoded, message, _MAX_MESSAGE_BYTES)
    return encoded


def _encode_message_with_limits(message: Message, limits: ConnectionLimits) -> bytes:
    """Encode one message while enforcing limits selected for a connection."""
    if not isinstance(limits, ConnectionLimits):
        raise TypeError("limits must be ConnectionLimits")
    _enforce_operational_limits(message, limits)
    encoded = _pack_message(message)
    _enforce_encoded_size(encoded, message, limits.max_message_bytes)
    return encoded


def _enforce_operational_limits(message: Message, limits: ConnectionLimits) -> None:
    if isinstance(message, InputAudioEvent) and len(message.audio) // 2 > limits.max_input_audio_frames:
        raise _OperationalLimitError("input audio exceeds the negotiated frame limit")
    if isinstance(message, OutputAudioEvent) and len(message.audio) // 2 > limits.max_output_audio_frames:
        raise _OperationalLimitError("output audio exceeds the negotiated frame limit")
    if (
        isinstance(
            message,
            (TranscriptUpdateEvent, TranscriptFinalEvent, ResponseTextDeltaEvent, ResponseTextFinalEvent),
        )
        and len(message.text.encode("utf-8")) > limits.max_text_bytes
    ):
        raise _OperationalLimitError("text exceeds the negotiated byte limit")


_CLIENT_MESSAGE_TYPES = known_message_types(MessageDirection.CLIENT_TO_SERVER, include_handshake=True)
_SERVER_MESSAGE_TYPES = known_message_types(MessageDirection.SERVER_TO_CLIENT, include_handshake=True)


def _decode_message_for_transport(
    data: ReadableBuffer,
    *,
    direction: MessageDirection,
    limits: ConnectionLimits,
) -> Message:
    """Decode a peer frame while retaining its stable connection-error category."""
    try:
        size = len(_as_bytes_view(data))
    except CodecError as error:
        raise _InboundMessageError(ErrorCode.MALFORMED_MESSAGE, str(error)) from error
    if size > limits.max_message_bytes:
        raise _InboundMessageError(ErrorCode.MESSAGE_TOO_LARGE, "message exceeds the negotiated envelope")

    try:
        primitive = _decode_primitive_message(data, max_message_bytes=limits.max_message_bytes)
        message_type = _string(primitive, "type")
    except CodecError as error:
        raise _InboundMessageError(ErrorCode.MALFORMED_MESSAGE, str(error)) from error
    assert message_type is not None

    allowed = _CLIENT_MESSAGE_TYPES if direction is MessageDirection.CLIENT_TO_SERVER else _SERVER_MESSAGE_TYPES
    all_known = _CLIENT_MESSAGE_TYPES | _SERVER_MESSAGE_TYPES
    if message_type not in allowed:
        code = ErrorCode.PROTOCOL_STATE if message_type in all_known else ErrorCode.UNKNOWN_MESSAGE
        raise _InboundMessageError(code, f"message type is not valid in {direction.value}")

    try:
        return decode_message(data, direction=direction, limits=limits)
    except _OperationalLimitError as error:
        raise _InboundMessageError(ErrorCode.MESSAGE_TOO_LARGE, str(error)) from error
    except CodecError as error:
        raise _InboundMessageError(ErrorCode.MALFORMED_MESSAGE, str(error)) from error


def decode_message(
    data: ReadableBuffer,
    *,
    direction: MessageDirection,
    limits: ConnectionLimits,
) -> Message:
    """Decode and schema-validate one message without applying lifecycle state."""
    if not isinstance(direction, MessageDirection):
        raise TypeError("direction must be a MessageDirection")
    if not isinstance(limits, ConnectionLimits):
        raise TypeError("limits must be ConnectionLimits")

    primitive = _decode_primitive_message(data, max_message_bytes=limits.max_message_bytes)
    _validate_field_names_and_unknown_binary(primitive)
    message_type = _string(primitive, "type")
    assert message_type is not None
    if len(message_type.encode("utf-8")) > 64:
        raise CodecError("type exceeds the 64-byte limit")

    try:
        message = decode_schema_message(primitive, direction)
    except CodecError:
        raise
    except (TypeError, ValueError) as error:
        raise CodecError("message does not satisfy its schema") from error

    if isinstance(message, ClientHello | ServerHello):
        if len(_as_bytes_view(data)) > _MAX_FIRST_MESSAGE_BYTES:
            raise CodecError("hello exceeds the 16384-byte envelope")
    else:
        _enforce_operational_limits(message, limits)
    return message
