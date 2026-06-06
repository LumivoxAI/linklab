from __future__ import annotations

import asyncio
from typing import cast
from dataclasses import dataclass
from collections.abc import Callable, Sequence, Awaitable

from websockets.typing import Subprotocol
from websockets.protocol import State
from websockets.exceptions import NegotiationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import Server, ServerConnection, serve

from ._codec import Primitive, decode_message, encode_message, _decode_primitive_message
from ._enums import ErrorCode, ErrorScope, MessageDirection
from ._config import (
    _REQUIRED_CAPABILITIES,
    AudioFormat,
    ClientConfig,
    ServerConfig,
    ConnectionLimits,
)
from ._errors import CodecError, ConnectionClosed
from ._schema import known_message_types, decode_schema_message
from ._messages import ErrorEvent, ClientHello, ServerHello

_SUBPROTOCOL = Subprotocol("lumivox.voice.v1")
_VERSION = 1
_MAX_MESSAGE_BYTES = 262_144
_MAX_FIRST_MESSAGE_BYTES = 16_384
_INPUT_FORMAT = AudioFormat("pcm_s16le", 16_000, 1)


@dataclass(frozen=True, slots=True)
class _ClientHandshake:
    connection: ClientConnection
    client_hello: ClientHello
    server_hello: ServerHello


@dataclass(frozen=True, slots=True)
class _ServerHandshake:
    client_hello: ClientHello
    server_hello: ServerHello


class _HandshakeRejected(ConnectionClosed):
    def __init__(self, error: ErrorEvent) -> None:
        super().__init__(f"handshake rejected: {error.code.value}")
        self.error = error


class _InvalidHello(CodecError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _select_subprotocol(_connection: ServerConnection, offered: Sequence[Subprotocol]) -> Subprotocol:
    if _SUBPROTOCOL not in offered:
        raise NegotiationError(f"missing required subprotocol {_SUBPROTOCOL}")
    return _SUBPROTOCOL


async def _open_client_websocket(config: ClientConfig) -> _ClientHandshake:
    if config.uri is None:
        raise ValueError("client WebSocket connection requires a resolved URI")
    connection = await connect(
        config.uri,
        subprotocols=[_SUBPROTOCOL],
        compression=None,
        open_timeout=config.connect_timeout_s,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=config.close_timeout_s,
        max_size=_MAX_MESSAGE_BYTES,
        max_queue=config.websocket_max_queue,
        write_limit=config.websocket_write_limit,
        ssl=config.ssl_context,
        proxy=None,
    )
    try:
        if connection.subprotocol != _SUBPROTOCOL:
            raise ConnectionClosed("server did not negotiate the required subprotocol")

        client_hello = ClientHello(
            _VERSION,
            _REQUIRED_CAPABILITIES,
            _INPUT_FORMAT,
            config.output_formats,
            config.agent,
        )
        await connection.send(encode_message(client_hello))
        try:
            async with asyncio.timeout(config.handshake_timeout_s):
                frame = await connection.recv()
        except TimeoutError as error:
            raise ConnectionClosed("server hello timed out") from error

        if type(frame) is not bytes:
            raise CodecError("WebSocket application messages must be binary")
        message = _decode_first_message(frame, MessageDirection.SERVER_TO_CLIENT)
        if isinstance(message, ErrorEvent):
            await connection.close(code=1002)
            raise _HandshakeRejected(message)
        if not isinstance(message, ServerHello):
            raise CodecError("server hello must be the first server message")
        if message.output_format not in client_hello.output_formats:
            raise CodecError("server selected an output format the client did not advertise")
        return _ClientHandshake(connection, client_hello, message)
    except asyncio.CancelledError:
        if connection.state is not State.CLOSED:
            await connection.close(code=1000)
        raise
    except BaseException:
        if connection.state is not State.CLOSED:
            await connection.close(code=1002)
        raise


async def _perform_server_handshake(connection: ServerConnection, config: ServerConfig) -> _ServerHandshake | None:
    if connection.subprotocol != _SUBPROTOCOL:
        await connection.close(code=1002)
        return None
    try:
        try:
            async with asyncio.timeout(config.handshake_timeout_s):
                frame = await connection.recv()
        except TimeoutError:
            await _reject_server_handshake(connection, ErrorCode.HANDSHAKE_TIMEOUT)
            return None

        if type(frame) is not bytes:
            await _reject_server_handshake(connection, ErrorCode.MALFORMED_MESSAGE)
            return None
        try:
            message = _decode_first_message(frame, MessageDirection.CLIENT_TO_SERVER)
        except _InvalidHello as error:
            await _reject_server_handshake(connection, error.code)
            return None
        if not isinstance(message, ClientHello):
            await _reject_server_handshake(connection, ErrorCode.PROTOCOL_STATE)
            return None

        selected = next((item for item in message.output_formats if item in config.output_formats), None)
        if selected is None:
            await _reject_server_handshake(connection, ErrorCode.FORMAT_MISMATCH)
            return None
        limits = _selected_limits(config.limits, selected)
        server_hello = ServerHello(_VERSION, _REQUIRED_CAPABILITIES, selected, limits, config.agent)
        await connection.send(encode_message(server_hello))
        return _ServerHandshake(message, server_hello)
    except BaseException:
        if connection.state is not State.CLOSED:
            await connection.close(code=1002)
        raise


async def _serve_websocket(
    config: ServerConfig,
    handler: Callable[[ServerConnection], Awaitable[None]],
) -> Server:
    return await serve(
        handler,
        config.host,
        config.port,
        subprotocols=[_SUBPROTOCOL],
        select_subprotocol=_select_subprotocol,
        compression=None,
        open_timeout=10.0,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=config.close_timeout_s,
        max_size=_MAX_MESSAGE_BYTES,
        max_queue=config.websocket_max_queue,
        write_limit=config.websocket_write_limit,
        ssl=config.ssl_context,
    )


async def _reject_server_handshake(connection: ServerConnection, code: ErrorCode) -> None:
    error = ErrorEvent(ErrorScope.CONNECTION, code, True)
    await connection.send(encode_message(error))
    await connection.close(code=1002)


def _selected_limits(configured: ConnectionLimits, output_format: AudioFormat) -> ConnectionLimits:
    return ConnectionLimits(
        max_message_bytes=min(configured.max_message_bytes, _MAX_MESSAGE_BYTES),
        max_input_audio_frames=min(configured.max_input_audio_frames, 1_600),
        max_output_audio_frames=min(configured.max_output_audio_frames, output_format.sample_rate_hz // 10),
        max_text_bytes=min(configured.max_text_bytes, 65_536),
        max_input_frames=min(configured.max_input_frames, 1_920_000),
        idle_timeout_ms=min(configured.idle_timeout_ms, 600_000),
    )


def _decode_first_message(data: bytes, direction: MessageDirection) -> ClientHello | ServerHello | ErrorEvent:
    try:
        primitive = _decode_primitive_message(data, first_message=True)
    except CodecError as error:
        code = ErrorCode.MESSAGE_TOO_LARGE if len(data) > _MAX_FIRST_MESSAGE_BYTES else ErrorCode.MALFORMED_MESSAGE
        raise _InvalidHello(code, str(error)) from error

    message_type = primitive.get("type")
    if type(message_type) is not str:
        raise _InvalidHello(ErrorCode.MALFORMED_MESSAGE, "first message type must be a string")
    if message_type not in ("hello", "error"):
        code = (
            ErrorCode.PROTOCOL_STATE if message_type in _known_message_types(direction) else ErrorCode.UNKNOWN_MESSAGE
        )
        raise _InvalidHello(code, "first message must be hello")
    if message_type == "error" and direction is not MessageDirection.SERVER_TO_CLIENT:
        raise _InvalidHello(ErrorCode.PROTOCOL_STATE, "client cannot send a handshake error")
    if message_type == "hello":
        _classify_hello_fields(primitive, direction)

    try:
        decoded = decode_message(data, direction=direction, limits=ConnectionLimits())
    except CodecError as error:
        raise _InvalidHello(ErrorCode.MALFORMED_MESSAGE, str(error)) from error
    if not isinstance(decoded, (ClientHello, ServerHello, ErrorEvent)):
        raise _InvalidHello(ErrorCode.PROTOCOL_STATE, "first message must be hello")
    if isinstance(decoded, ErrorEvent) and decoded.scope is not ErrorScope.CONNECTION:
        raise _InvalidHello(ErrorCode.PROTOCOL_STATE, "handshake error must have connection scope")
    return decoded


def _classify_hello_fields(message: dict[str, Primitive], direction: MessageDirection) -> None:
    if message.get("version") != _VERSION:
        raise _InvalidHello(ErrorCode.UNSUPPORTED_VERSION, "unsupported application protocol version")
    if "capabilities" not in message:
        raise _InvalidHello(ErrorCode.MALFORMED_MESSAGE, "missing capabilities")
    capabilities = message["capabilities"]
    if type(capabilities) is not list or any(type(value) is not str for value in capabilities):
        raise _InvalidHello(ErrorCode.CAPABILITY_MISMATCH, "invalid capabilities")
    capability_strings = cast(list[str], capabilities)
    if (
        capability_strings != sorted(capability_strings, key=lambda value: value.encode())
        or len(capability_strings) != len(set(capability_strings))
        or not set(_REQUIRED_CAPABILITIES).issubset(capability_strings)
    ):
        raise _InvalidHello(ErrorCode.CAPABILITY_MISMATCH, "invalid capabilities")

    format_fields = (
        ("input_format", "output_formats") if direction is MessageDirection.CLIENT_TO_SERVER else ("output_format",)
    )
    if any(name not in message for name in format_fields):
        raise _InvalidHello(ErrorCode.MALFORMED_MESSAGE, "missing audio format")
    try:
        candidate = {name: value for name, value in message.items() if name != "agent"}
        decode_schema_message(candidate, direction)
    except (CodecError, TypeError, ValueError) as error:
        raise _InvalidHello(ErrorCode.FORMAT_MISMATCH, "invalid audio format") from error


def _hello_without_agent(message: dict[str, Primitive], direction: MessageDirection) -> ClientHello | ServerHello:
    capabilities = tuple(cast(list[str], message["capabilities"]))
    if direction is MessageDirection.CLIENT_TO_SERVER:
        input_format = _primitive_audio_format(message["input_format"])
        output_formats_value = message["output_formats"]
        if type(output_formats_value) is not list:
            raise ValueError("output_formats must be an array")
        output_formats = tuple(_primitive_audio_format(value) for value in output_formats_value)
        return ClientHello(_VERSION, capabilities, input_format, output_formats)
    output_format = _primitive_audio_format(message["output_format"])
    limits_value = message.get("limits")
    if type(limits_value) is not dict:
        raise ValueError("limits must be a map")
    limits = ConnectionLimits(
        max_message_bytes=_primitive_int(limits_value, "max_message_bytes"),
        max_input_audio_frames=_primitive_int(limits_value, "max_input_audio_frames"),
        max_output_audio_frames=_primitive_int(limits_value, "max_output_audio_frames"),
        max_text_bytes=_primitive_int(limits_value, "max_text_bytes"),
        max_input_frames=_primitive_int(limits_value, "max_input_frames"),
        idle_timeout_ms=_primitive_int(limits_value, "idle_timeout_ms"),
    )
    return ServerHello(_VERSION, capabilities, output_format, limits)


def _primitive_audio_format(value: Primitive) -> AudioFormat:
    if type(value) is not dict:
        raise ValueError("audio format must be a map")
    return AudioFormat(
        value.get("encoding"),  # type: ignore[arg-type]
        value.get("sample_rate_hz"),  # type: ignore[arg-type]
        value.get("channels"),  # type: ignore[arg-type]
    )


def _primitive_int(value: dict[str, Primitive], name: str) -> int:
    item = value.get(name)
    if type(item) is not int:
        raise ValueError(f"{name} must be an integer")
    return item


def _known_message_types(direction: MessageDirection) -> frozenset[str]:
    return known_message_types(direction)
