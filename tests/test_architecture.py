import asyncio
from typing import Any, cast
from collections.abc import Awaitable, Coroutine

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._transport import _TransportCore

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


class FakeCallbacks:
    async def on_connection_state(self, _event: linklab.ConnectionStateEvent) -> None:
        pass

    async def on_conversation_state(self, _event: linklab.StateEvent) -> None:
        pass

    async def on_transcript_update(self, _event: linklab.TranscriptUpdateEvent) -> None:
        pass

    async def on_transcript_final(self, _event: linklab.TranscriptFinalEvent) -> None:
        pass

    async def on_response_started(self, _event: linklab.ResponseStartedEvent) -> None:
        pass

    async def on_response_text_delta(self, _event: linklab.ResponseTextDeltaEvent) -> None:
        pass

    async def on_response_text_final(self, _event: linklab.ResponseTextFinalEvent) -> None:
        pass

    async def on_response_ended(self, _event: linklab.ResponseEndedEvent) -> None:
        pass

    async def on_response_cancelled(self, _event: linklab.ResponseCancelledEvent) -> None:
        pass

    async def on_output_started(self, _event: linklab.OutputStartedEvent) -> None:
        pass

    async def on_output_audio(self, _event: linklab.OutputAudioEvent) -> None:
        pass

    async def on_output_ended(self, _event: linklab.OutputEndedEvent) -> None:
        pass

    async def on_conversation_ended(self, _event: linklab.ConversationEndedEvent) -> None:
        pass

    async def on_error(self, _event: linklab.ErrorEvent) -> None:
        pass


class FakeTransport:
    async def recv(self) -> bytes:
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def send(self, _data: bytes) -> None:
        pass

    async def ping(self) -> Awaitable[float]:
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        future.set_result(0.0)
        return future

    async def close(self, _code: int) -> None:
        pass


def test_facades_route_wire_transitions_through_protocol_validator() -> None:
    async def scenario() -> None:
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,)),
            FakeCallbacks(),
            object(),
        )
        server = linklab.VoiceServer(
            linklab.ServerConfig(1, (PCM_16K,)),
            cast(Any, lambda _session: None),
            object(),
        )
        core = _TransportCore(
            FakeTransport(),
            role=linklab.EndpointRole.SERVER,
            limits=linklab.ConnectionLimits(max_output_audio_frames=1_600),
            data_capacity=1,
            control_capacity=1,
            close_timeout_s=0.1,
            on_transition=lambda _message, _result: None,
            on_transport_loss=lambda _error: None,
        )

        assert type(client._ingress._validator) is linklab.ProtocolValidator
        assert type(core.validator) is linklab.ProtocolValidator
        assert not hasattr(client, "_validator")
        assert not hasattr(server, "_validator")

    run(scenario())


def test_public_owners_bind_to_creation_loop_and_sync_handoff_uses_no_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="running event loop"):
        linklab.VoiceClient(linklab.ClientConfig("ws://unused", (PCM_16K,)), FakeCallbacks(), object())
    with pytest.raises(RuntimeError, match="running event loop"):
        linklab.VoiceServer(linklab.ServerConfig(1, (PCM_16K,)), cast(Any, lambda _session: None), object())

    owners: list[linklab.VoiceClient | linklab.VoiceServer] = []

    async def scenario() -> None:
        def forbidden_to_thread(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("client handoff created a per-call worker task")

        monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)
        client = linklab.VoiceClient(
            linklab.ClientConfig("ws://unused", (PCM_16K,)),
            FakeCallbacks(),
            object(),
        )
        server = linklab.VoiceServer(
            linklab.ServerConfig(1, (PCM_16K,)),
            cast(Any, lambda _session: None),
            object(),
        )
        owners.extend((client, server))
        assert (
            client.submit_annotated_audio(linklab.AnnotatedAudio(b"\0\0", 0, False, False, False))
            is linklab.AudioSubmitResult.IGNORED_INACTIVE
        )
        assert client.abort_input(linklab.InputAbortReason.CAPTURE_FAILED) is False

    run(scenario())

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await cast(linklab.VoiceClient, owners[0]).wait_closed()
        with pytest.raises(RuntimeError, match="different event loop"):
            await cast(linklab.VoiceServer, owners[1]).wait_closed()

    run(wrong_loop())


def test_writer_currency_check_and_publication_have_no_scheduling_gap() -> None:
    class InstrumentedSession:
        def __init__(self) -> None:
            self.current = True
            self.competitor_ran = False
            self.published: list[linklab.Message] = []

        def _check_loop(self) -> None:
            pass

        def _require_writer_response(self, _writer: linklab.ResponseWriter) -> object:
            assert self.current
            asyncio.get_running_loop().call_soon(self._make_stale)
            return object()

        def _require_conversation_id(self) -> linklab.ConversationId:
            assert self.current
            return linklab.ConversationId(1)

        def _enqueue_control(self, messages: tuple[linklab.Message, ...]) -> None:
            assert self.current
            assert not self.competitor_ran
            self.published.extend(messages)

        def _make_stale(self) -> None:
            self.current = False
            self.competitor_ran = True

    async def scenario() -> None:
        session = InstrumentedSession()
        writer = linklab.ResponseWriter(cast(Any, session), linklab.ResponseId(1))
        cast(Any, session)._response_writer = writer
        await writer.send_text_delta("current")
        assert session.published == [
            linklab.ResponseTextDeltaEvent(linklab.ConversationId(1), linklab.ResponseId(1), 0, "current")
        ]
        await asyncio.sleep(0)
        assert session.competitor_ran

    run(scenario())
