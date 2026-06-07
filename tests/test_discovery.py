import socket
import asyncio
from typing import Any
from collections.abc import Coroutine

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._discovery import _endpoint_uri, _usable_address, _service_candidates

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class NullCallbacks:
    async def on_connection_state(self, _event: linklab.ConnectionStateEvent) -> None: ...

    async def on_conversation_state(self, _event: linklab.StateEvent) -> None: ...

    async def on_transcript_update(self, _event: linklab.TranscriptUpdateEvent) -> None: ...

    async def on_transcript_final(self, _event: linklab.TranscriptFinalEvent) -> None: ...

    async def on_response_started(self, _event: linklab.ResponseStartedEvent) -> None: ...

    async def on_response_text_delta(self, _event: linklab.ResponseTextDeltaEvent) -> None: ...

    async def on_response_text_final(self, _event: linklab.ResponseTextFinalEvent) -> None: ...

    async def on_response_ended(self, _event: linklab.ResponseEndedEvent) -> None: ...

    async def on_response_cancelled(self, _event: linklab.ResponseCancelledEvent) -> None: ...

    async def on_output_started(self, _event: linklab.OutputStartedEvent) -> None: ...

    async def on_output_audio(self, _event: linklab.OutputAudioEvent) -> None: ...

    async def on_output_ended(self, _event: linklab.OutputEndedEvent) -> None: ...

    async def on_conversation_ended(self, _event: linklab.ConversationEndedEvent) -> None: ...

    async def on_error(self, _event: linklab.ErrorEvent) -> None: ...


class NullHandler:
    async def on_conversation_started(
        self, _session: linklab.ServerSession, _event: linklab.ConversationStartedEvent
    ) -> None: ...

    async def on_input_started(self, _session: linklab.ServerSession, _event: linklab.InputStartedEvent) -> None: ...

    async def on_input_audio(self, _session: linklab.ServerSession, _event: linklab.InputAudioEvent) -> None: ...

    async def on_input_aborted(self, _session: linklab.ServerSession, _event: linklab.InputAbortedEvent) -> None: ...

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        _event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None: ...

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        _event: linklab.ConversationCancelledEvent,
    ) -> None: ...


class FakeBrowser:
    def __init__(self, candidates: tuple[str, ...]) -> None:
        self.candidates = candidates
        self.revision = 0
        self.updated = asyncio.Event()
        self.closed = False
        self.resolve_count = 0

    async def resolve(self, timeout_s: float) -> tuple[str, ...]:
        self.resolve_count += 1
        if not self.candidates:
            async with asyncio.timeout(timeout_s):
                await self.updated.wait()
        return self.candidates

    async def wait_for_update(self, revision: int) -> None:
        while self.revision == revision and not self.closed:
            self.updated.clear()
            await self.updated.wait()

    async def close(self) -> None:
        self.closed = True
        self.updated.set()

    def replace(self, candidates: tuple[str, ...]) -> None:
        self.candidates = candidates
        self.revision += 1
        self.updated.set()


def test_discovery_reaches_normal_handshake_and_closes_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        port = unused_port()
        server = linklab.VoiceServer(
            linklab.ServerConfig(port, (PCM_16K,)),
            lambda _session: NullHandler(),
            object(),
        )
        await server.serve()
        browser = FakeBrowser((f"ws://127.0.0.1:{port}",))
        opened: list[str] = []

        async def open_browser(service_id: str) -> FakeBrowser:
            opened.append(service_id)
            return browser

        monkeypatch.setattr("lumivox_linklab._client._open_service_browser", open_browser)
        client = linklab.VoiceClient(
            linklab.ClientConfig(None, (PCM_16K,), discovery_service_id="production"),
            NullCallbacks(),
            object(),
        )
        await client.connect()
        assert client.connection_state is linklab.ConnectionState.READY
        assert opened == ["production"]
        assert browser.resolve_count == 1

        browser.replace(())
        await asyncio.sleep(0)
        assert client.connection_state is linklab.ConnectionState.READY

        await client.close()
        await server.close()
        assert browser.closed

    run(scenario())


def test_direct_uri_never_creates_discovery_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        port = unused_port()
        server = linklab.VoiceServer(
            linklab.ServerConfig(port, (PCM_16K,)),
            lambda _session: NullHandler(),
            object(),
        )
        await server.serve()

        async def forbidden(_service_id: str) -> FakeBrowser:
            raise AssertionError("direct URI must not create discovery resources")

        monkeypatch.setattr("lumivox_linklab._client._open_service_browser", forbidden)
        client = linklab.VoiceClient(
            linklab.ClientConfig(
                f"ws://127.0.0.1:{port}",
                (PCM_16K,),
                discovery_service_id="production",
            ),
            NullCallbacks(),
            object(),
        )
        await client.connect()
        await client.close()
        await server.close()

    run(scenario())


def test_discovery_update_wakes_reconnect_delay() -> None:
    async def scenario() -> None:
        client = linklab.VoiceClient(
            linklab.ClientConfig(None, (PCM_16K,), discovery_service_id="production"),
            NullCallbacks(),
            object(),
        )
        browser = FakeBrowser(("ws://192.0.2.1:9000",))
        client._discovery = browser
        client._reconnect_discovery_revision = browser.revision
        waiting = asyncio.create_task(client._wait_reconnect_delay(10))
        await asyncio.sleep(0)
        browser.replace(("ws://192.0.2.2:9000",))
        assert await asyncio.wait_for(waiting, 1)
        await browser.close()

    run(scenario())


def test_initial_discovery_timeout_closes_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        browser = FakeBrowser(())

        async def open_browser(_service_id: str) -> FakeBrowser:
            return browser

        monkeypatch.setattr("lumivox_linklab._client._open_service_browser", open_browser)
        client = linklab.VoiceClient(
            linklab.ClientConfig(
                None,
                (PCM_16K,),
                discovery_service_id="production",
                discovery_timeout_s=0.01,
            ),
            NullCallbacks(),
            object(),
        )
        with pytest.raises(TimeoutError):
            await client.connect()
        assert browser.closed
        await client.wait_closed()

    run(scenario())


def test_server_advertisement_lifecycle_and_registration_rollback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Advertiser:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        calls: list[tuple[str, str, int, bool]] = []
        advertiser = Advertiser()

        async def register(service_id: str, host: str, port: int, *, secure: bool) -> Advertiser:
            calls.append((service_id, host, port, secure))
            return advertiser

        monkeypatch.setattr("lumivox_linklab._server._register_service", register)
        port = unused_port()
        server = linklab.VoiceServer(
            linklab.ServerConfig(port, (PCM_16K,), host="0.0.0.0", discovery_service_id="production"),
            lambda _session: NullHandler(),
            object(),
        )
        await server.serve()
        assert calls == [("production", "0.0.0.0", port, False)]
        await server.close()
        assert advertiser.closed

        failed_port = unused_port()

        async def conflict(_service_id: str, _host: str, _port: int, *, secure: bool) -> Advertiser:
            del secure
            raise RuntimeError("name conflict")

        monkeypatch.setattr("lumivox_linklab._server._register_service", conflict)
        failed = linklab.VoiceServer(
            linklab.ServerConfig(
                failed_port,
                (PCM_16K,),
                host="0.0.0.0",
                discovery_service_id="production",
            ),
            lambda _session: NullHandler(),
            object(),
        )
        with pytest.raises(RuntimeError, match="name conflict"):
            await failed.serve()
        assert failed._listener is None
        with pytest.raises(OSError):
            await asyncio.open_connection("127.0.0.1", failed_port)

    run(scenario())


def test_address_filter_and_uri_formatting() -> None:
    assert not _usable_address("127.0.0.1")
    assert not _usable_address("::1")
    assert not _usable_address("0.0.0.0")
    assert not _usable_address("224.0.0.1")
    assert _usable_address("192.0.2.1")
    assert _endpoint_uri("ws", "192.0.2.1", 9000) == "ws://192.0.2.1:9000"
    assert _endpoint_uri("wss", "fe80::1%eth0", 443) == "wss://[fe80::1%25eth0]:443"


def test_service_candidate_profile_is_exact_and_replaces_the_address_set() -> None:
    addresses = ["127.0.0.1", "192.0.2.1", "192.0.2.1", "2001:db8::1"]
    assert _service_candidates(
        {b"protocol": b"lumivox.voice.v1", b"scheme": b"ws"},
        addresses,
        9000,
    ) == ("ws://192.0.2.1:9000", "ws://[2001:db8::1]:9000")
    for properties in (
        {b"protocol": b"other", b"scheme": b"ws"},
        {b"protocol": b"lumivox.voice.v1", b"scheme": b"http"},
        {b"protocol": b"lumivox.voice.v1"},
    ):
        assert _service_candidates(properties, addresses, 9000) == ()
    assert _service_candidates({b"protocol": b"lumivox.voice.v1", b"scheme": b"ws"}, addresses, 0) == ()
