import ssl
import socket
import asyncio
from typing import Any, cast
from collections.abc import Coroutine

import pytest
import msgpack  # type: ignore[import-untyped]
from websockets.protocol import State
from websockets.exceptions import InvalidStatus
from websockets.asyncio.client import connect
from websockets.asyncio.server import Server, ServerConnection

import lumivox_linklab as linklab
from lumivox_linklab._handshake import (
    _SUBPROTOCOL,
    _InvalidHello,
    _selected_limits,
    _serve_websocket,
    _HandshakeRejected,
    _decode_first_message,
    _open_client_websocket,
    _perform_server_handshake,
)
from lumivox_linklab._transport import _TransportCore

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
PCM_24K = linklab.AudioFormat("pcm_s16le", 24_000, 1)
PCM_48K = linklab.AudioFormat("pcm_s16le", 48_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def server_config(**changes: Any) -> linklab.ServerConfig:
    values: dict[str, Any] = {
        "port": unused_port(),
        "output_formats": (PCM_16K, PCM_24K, PCM_48K),
    }
    values.update(changes)
    return linklab.ServerConfig(**values)


def client_config(port: int, **changes: Any) -> linklab.ClientConfig:
    values: dict[str, Any] = {
        "uri": f"ws://127.0.0.1:{port}",
        "output_formats": (PCM_24K, PCM_48K, PCM_16K),
    }
    values.update(changes)
    return linklab.ClientConfig(**values)


async def close_server(server: Server) -> None:
    server.close()
    await server.wait_closed()


def primitive_client_hello() -> dict[str, Any]:
    return {
        "type": "hello",
        "version": 1,
        "capabilities": list(CAPABILITIES),
        "input_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1},
        "output_formats": [{"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1}],
    }


async def raw_handshake(config: linklab.ServerConfig, first: bytes | str) -> tuple[linklab.ErrorEvent, int | None]:
    async def handler(connection: ServerConnection) -> None:
        await _perform_server_handshake(connection, config)

    server = await _serve_websocket(config, handler)
    try:
        async with connect(
            f"ws://127.0.0.1:{config.port}",
            subprotocols=[_SUBPROTOCOL],
            compression=None,
            max_size=262_144,
            proxy=None,
        ) as connection:
            await connection.send(first)
            reply = await connection.recv()
            assert type(reply) is bytes
            error = linklab.decode_message(
                reply,
                direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                limits=linklab.ConnectionLimits(),
            )
            assert isinstance(error, linklab.ErrorEvent)
            await connection.wait_closed()
            return error, connection.close_code
    finally:
        await close_server(server)


def test_loopback_handshake_selects_client_preference_and_imposes_limits() -> None:
    async def scenario() -> None:
        configured_limits = linklab.ConnectionLimits(
            max_message_bytes=32_768,
            max_input_audio_frames=800,
            max_output_audio_frames=4_800,
            max_text_bytes=2_048,
            max_input_frames=32_000,
            idle_timeout_ms=10_000,
        )
        config = server_config(limits=configured_limits)
        accepted: list[Any] = []

        async def handler(connection: ServerConnection) -> None:
            accepted.append(await _perform_server_handshake(connection, config))
            await connection.wait_closed()

        server = await _serve_websocket(config, handler)
        try:
            handshake = await _open_client_websocket(client_config(config.port))
            assert handshake.connection.subprotocol == _SUBPROTOCOL
            assert handshake.connection.protocol.extensions == []
            assert handshake.server_hello.output_format == PCM_24K
            assert handshake.server_hello.limits == linklab.ConnectionLimits(
                max_message_bytes=32_768,
                max_input_audio_frames=800,
                max_output_audio_frames=2_400,
                max_text_bytes=2_048,
                max_input_frames=32_000,
                idle_timeout_ms=10_000,
            )
            await handshake.connection.close()
        finally:
            await close_server(server)

        assert accepted and accepted[0].client_hello.output_formats == (PCM_24K, PCM_48K, PCM_16K)

    run(scenario())


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda value: value.update(version=2), linklab.ErrorCode.UNSUPPORTED_VERSION),
        (
            lambda value: value.update(capabilities=["barge_in", "barge_in", "playback_accounting"]),
            linklab.ErrorCode.CAPABILITY_MISMATCH,
        ),
        (
            lambda value: value.update(capabilities=["speech_spans", "playback_accounting", "barge_in"]),
            linklab.ErrorCode.CAPABILITY_MISMATCH,
        ),
        (lambda value: value.update(capabilities=["barge_in"]), linklab.ErrorCode.CAPABILITY_MISMATCH),
        (lambda value: value.update(output_formats=[]), linklab.ErrorCode.FORMAT_MISMATCH),
        (
            lambda value: value.update(
                output_formats=[
                    {"encoding": "pcm_s16le", "sample_rate_hz": 24_000, "channels": 1},
                ]
            ),
            linklab.ErrorCode.FORMAT_MISMATCH,
        ),
        (
            lambda value: value.update(
                output_formats=[
                    {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1},
                    {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1},
                ]
            ),
            linklab.ErrorCode.FORMAT_MISMATCH,
        ),
        (lambda value: value.update(input_format={"encoding": "bad"}), linklab.ErrorCode.FORMAT_MISMATCH),
    ],
)
def test_server_rejects_invalid_hello_with_specific_safe_error(
    mutate: Any,
    expected: linklab.ErrorCode,
) -> None:
    value = primitive_client_hello()
    mutate(value)
    payload = cast(bytes, msgpack.packb(value, use_bin_type=True))
    error, close_code = run(raw_handshake(server_config(), payload))
    assert error == linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, expected, True)
    assert close_code == 1002


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("text", linklab.ErrorCode.MALFORMED_MESSAGE),
        (b"\xc1", linklab.ErrorCode.MALFORMED_MESSAGE),
        (b"x" * 16_385, linklab.ErrorCode.MESSAGE_TOO_LARGE),
        (
            linklab.encode_message(linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word")),
            linklab.ErrorCode.PROTOCOL_STATE,
        ),
        (cast(bytes, msgpack.packb({"type": "future.message"})), linklab.ErrorCode.UNKNOWN_MESSAGE),
    ],
)
def test_server_rejects_invalid_first_frame(payload: bytes | str, expected: linklab.ErrorCode) -> None:
    error, close_code = run(raw_handshake(server_config(), payload))
    assert error.code is expected
    assert error.message is None
    assert close_code == 1002


def test_server_handshake_timeout_sends_error_and_closes() -> None:
    async def scenario() -> None:
        config = server_config(handshake_timeout_s=0.01)

        async def handler(connection: ServerConnection) -> None:
            await _perform_server_handshake(connection, config)

        server = await _serve_websocket(config, handler)
        try:
            async with connect(
                f"ws://127.0.0.1:{config.port}",
                subprotocols=[_SUBPROTOCOL],
                compression=None,
                proxy=None,
            ) as connection:
                reply = await connection.recv()
                assert type(reply) is bytes
                message = linklab.decode_message(
                    reply,
                    direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                    limits=linklab.ConnectionLimits(),
                )
                assert isinstance(message, linklab.ErrorEvent)
                assert message.code is linklab.ErrorCode.HANDSHAKE_TIMEOUT
                await connection.wait_closed()
                assert connection.close_code == 1002
        finally:
            await close_server(server)

    run(scenario())


def test_required_subprotocol_is_enforced_before_application_handshake() -> None:
    async def scenario() -> None:
        config = server_config()

        async def handler(connection: ServerConnection) -> None:
            pytest.fail(f"handler unexpectedly called for {connection}")

        server = await _serve_websocket(config, handler)
        try:
            with pytest.raises(InvalidStatus):
                await connect(f"ws://127.0.0.1:{config.port}", compression=None, proxy=None)
        finally:
            await close_server(server)

    run(scenario())


def test_fragmented_binary_hello_is_accepted() -> None:
    async def scenario() -> None:
        config = server_config()

        async def handler(connection: ServerConnection) -> None:
            await _perform_server_handshake(connection, config)

        server = await _serve_websocket(config, handler)
        try:
            async with connect(
                f"ws://127.0.0.1:{config.port}",
                subprotocols=[_SUBPROTOCOL],
                compression=None,
                proxy=None,
            ) as connection:
                hello = linklab.encode_message(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
                await connection.send([hello[:3], hello[3:]])
                reply = await connection.recv()
                assert type(reply) is bytes
                assert isinstance(
                    linklab.decode_message(
                        reply,
                        direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                        limits=linklab.ConnectionLimits(),
                    ),
                    linklab.ServerHello,
                )
        finally:
            await close_server(server)

    run(scenario())


def test_client_accepts_connection_error_in_place_of_server_hello() -> None:
    async def scenario() -> None:
        config = server_config()
        error = linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, linklab.ErrorCode.FORMAT_MISMATCH, True)

        async def handler(connection: ServerConnection) -> None:
            await connection.recv()
            await connection.send(linklab.encode_message(error))
            await connection.wait_closed()

        server = await _serve_websocket(config, handler)
        try:
            with pytest.raises(_HandshakeRejected) as raised:
                await _open_client_websocket(client_config(config.port))
            assert raised.value.error == error
        finally:
            await close_server(server)

    run(scenario())


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("text", linklab.ErrorCode.MALFORMED_MESSAGE),
        (b"\xc1", linklab.ErrorCode.MALFORMED_MESSAGE),
        (msgpack.packb({"type": "future.message"}, use_bin_type=True), linklab.ErrorCode.UNKNOWN_MESSAGE),
        (
            linklab.encode_message(linklab.ConversationStartedEvent(linklab.ConversationId(2), "wake_word")),
            linklab.ErrorCode.PROTOCOL_STATE,
        ),
        (b"x" * 16_385, linklab.ErrorCode.MESSAGE_TOO_LARGE),
    ],
)
def test_loopback_post_handshake_fatal_error_is_last_message(
    payload: bytes | str,
    expected: linklab.ErrorCode,
) -> None:
    async def scenario() -> None:
        config = server_config(limits=linklab.ConnectionLimits(max_message_bytes=16_384, max_output_audio_frames=4_800))
        cores: list[_TransportCore] = []

        async def handler(connection: ServerConnection) -> None:
            handshake = await _perform_server_handshake(connection, config)
            assert handshake is not None
            core = _TransportCore(
                connection,
                role=linklab.EndpointRole.SERVER,
                limits=handshake.server_hello.limits,
                data_capacity=8,
                control_capacity=8,
                close_timeout_s=config.close_timeout_s,
                ping_interval_s=config.ping_interval_s,
                ping_timeout_s=config.ping_timeout_s,
                on_transition=lambda _message, _result: None,
                on_transport_loss=lambda _error: None,
            )
            cores.append(core)
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
            await core.wait_closed()

        server = await _serve_websocket(config, handler)
        try:
            async with connect(
                f"ws://127.0.0.1:{config.port}",
                subprotocols=[_SUBPROTOCOL],
                compression=None,
                ping_interval=None,
                proxy=None,
            ) as connection:
                await connection.send(linklab.encode_message(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,))))
                hello = await connection.recv()
                assert type(hello) is bytes
                await connection.send(payload)
                reply = await connection.recv()
                assert type(reply) is bytes
                error = linklab.decode_message(
                    reply,
                    direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                    limits=linklab.ConnectionLimits(),
                )
                assert isinstance(error, linklab.ErrorEvent)
                assert error.code is expected
                await connection.wait_closed()
                assert connection.close_code == 1002
        finally:
            await close_server(server)

        assert cores and cores[0].outcome is not None
        assert not cores[0].outcome.reconnect_eligible

    run(scenario())


def test_selected_limits_use_selected_rate_bounds() -> None:
    configured = linklab.ConnectionLimits(max_output_audio_frames=4_800)
    assert _selected_limits(configured, PCM_16K).max_output_audio_frames == 1_600
    assert _selected_limits(configured, PCM_24K).max_output_audio_frames == 2_400
    assert _selected_limits(configured, PCM_48K).max_output_audio_frames == 4_800

    invalid = primitive_client_hello()
    invalid["output_formats"] = [
        {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1},
        {"encoding": "pcm_s16le", "sample_rate_hz": 24_000, "channels": 1},
    ]
    with pytest.raises(_InvalidHello) as raised:
        _decode_first_message(
            cast(bytes, msgpack.packb(invalid, use_bin_type=True)),
            linklab.MessageDirection.CLIENT_TO_SERVER,
        )
    assert raised.value.code is linklab.ErrorCode.FORMAT_MISMATCH


def test_websocket_options_and_ssl_context_are_wired_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConnection:
        subprotocol = _SUBPROTOCOL
        state = State.OPEN

        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self.close_codes: list[int] = []

        async def send(self, data: bytes) -> None:
            self.sent.append(data)

        async def recv(self) -> bytes:
            return linklab.encode_message(
                linklab.ServerHello(1, CAPABILITIES, PCM_16K, linklab.ConnectionLimits(max_output_audio_frames=1_600))
            )

        async def close(self, code: int = 1000) -> None:
            self.close_codes.append(code)
            self.state = State.CLOSED

    client_calls: list[tuple[str, dict[str, Any]]] = []
    fake_connection = FakeConnection()

    async def fake_connect(uri: str, **kwargs: Any) -> Any:
        client_calls.append((uri, kwargs))
        return fake_connection

    server_calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    fake_server = cast(Server, object())

    async def fake_serve(handler: Any, host: Any, port: Any, **kwargs: Any) -> Server:
        server_calls.append((handler, host, port, kwargs))
        return fake_server

    monkeypatch.setattr("lumivox_linklab._handshake.connect", fake_connect)
    monkeypatch.setattr("lumivox_linklab._handshake.serve", fake_serve)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client = linklab.ClientConfig(
        "wss://voice.local",
        (PCM_16K,),
        ssl_context=context,
        connect_timeout_s=3,
        handshake_timeout_s=2,
        close_timeout_s=4,
        ping_interval_s=5,
        ping_timeout_s=6,
        websocket_max_queue=7,
        websocket_write_limit=8_192,
    )
    handshake = run(_open_client_websocket(client))
    assert cast(object, handshake.connection) is fake_connection
    assert client_calls == [
        (
            "wss://voice.local",
            {
                "subprotocols": [_SUBPROTOCOL],
                "compression": None,
                "open_timeout": 3,
                "ping_interval": None,
                "ping_timeout": None,
                "close_timeout": 4,
                "max_size": 262_144,
                "max_queue": 7,
                "write_limit": 8_192,
                "ssl": context,
                "proxy": None,
            },
        )
    ]

    async def handler(_connection: ServerConnection) -> None:
        return None

    server_config_value = server_config(
        ssl_context=context,
        close_timeout_s=4,
        ping_interval_s=5,
        ping_timeout_s=6,
        websocket_max_queue=7,
        websocket_write_limit=8_192,
    )
    assert run(_serve_websocket(server_config_value, handler)) is fake_server
    assert server_calls[0][1:3] == (server_config_value.host, server_config_value.port)
    assert server_calls[0][3] == {
        "subprotocols": [_SUBPROTOCOL],
        "select_subprotocol": server_calls[0][3]["select_subprotocol"],
        "compression": None,
        "open_timeout": 10.0,
        "ping_interval": None,
        "ping_timeout": None,
        "close_timeout": 4,
        "max_size": 262_144,
        "max_queue": 7,
        "write_limit": 8_192,
        "ssl": context,
    }
    assert client.ssl_context is context
    assert server_config_value.ssl_context is context
