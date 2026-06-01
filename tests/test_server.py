import ssl
import time
import socket
import asyncio
import inspect
from typing import Any, Self, cast, get_type_hints
from collections.abc import Coroutine

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.client import connect

import lumivox_linklab as linklab
from lumivox_linklab._server import _HandlerRaised, _HandlerQueueFull
from lumivox_linklab._handshake import _SUBPROTOCOL, _open_client_websocket
from lumivox_linklab._transport import _QueueLane

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def server_config(**changes: Any) -> linklab.ServerConfig:
    values: dict[str, Any] = {
        "port": unused_port(),
        "output_formats": (PCM_16K,),
    }
    values.update(changes)
    return linklab.ServerConfig(**values)


def client_config(port: int) -> linklab.ClientConfig:
    return linklab.ClientConfig(f"ws://127.0.0.1:{port}", (PCM_16K,))


class RecordingHandler:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationStartedEvent,
    ) -> None:
        self.events.append(event)

    async def on_input_started(self, _session: linklab.ServerSession, event: linklab.InputStartedEvent) -> None:
        self.events.append(event)

    async def on_input_audio(self, _session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
        self.events.append(event)

    async def on_input_aborted(self, _session: linklab.ServerSession, event: linklab.InputAbortedEvent) -> None:
        self.events.append(event)

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        self.events.append(event)

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationCancelledEvent,
    ) -> None:
        self.events.append(event)


async def wait_for_count(items: list[object], count: int) -> None:
    async with asyncio.timeout(1):
        while len(items) < count:
            await asyncio.sleep(0)


async def send_message(connection: Any, message: linklab.Message) -> None:
    await connection.send(linklab.encode_message(message))


async def receive_messages(handshake: Any, count: int) -> list[linklab.Message]:
    messages: list[linklab.Message] = []
    async with asyncio.timeout(1):
        for _ in range(count):
            frame = await handshake.connection.recv()
            assert type(frame) is bytes
            messages.append(
                linklab.decode_message(
                    frame,
                    direction=linklab.MessageDirection.SERVER_TO_CLIENT,
                    limits=handshake.server_hello.limits,
                )
            )
    return messages


def test_server_shutdown_closes_open_input_before_conversation_and_uses_1001() -> None:
    async def scenario() -> None:
        config = server_config(close_timeout_s=0.2)
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        assert await receive_messages(client, 1) == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING)
        ]

        closing = asyncio.create_task(server.close())
        assert await receive_messages(client, 3) == [
            linklab.InputClosedEvent(conversation_id, input_id, 0, linklab.InputCloseReason.FAILED),
            linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.CANCELLED),
        ]
        await closing
        await client.connection.wait_closed()
        assert client.connection.close_code == 1001
        assert not server._sessions

    run(scenario())


def test_server_binds_returns_and_isolates_sessions_and_handlers() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=2)
        sessions: list[linklab.ServerSession] = []
        handlers: list[RecordingHandler] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            handler = RecordingHandler()
            sessions.append(session)
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        first = await _open_client_websocket(client_config(config.port))
        second = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 2)

        assert sessions[0] is not sessions[1]
        assert handlers[0] is not handlers[1]
        assert all(not handler.events for handler in handlers)

        await asyncio.gather(first.connection.close(), second.connection.close())
        await asyncio.gather(server.close(), server.close())
        await server.wait_closed()

    run(scenario())


def test_connection_cap_rejects_with_1008_and_releases_slot() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=1)
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        first = await _open_client_websocket(client_config(config.port))

        rejected = await connect(
            f"ws://127.0.0.1:{config.port}",
            subprotocols=[_SUBPROTOCOL],
            compression=None,
            proxy=None,
        )
        await rejected.wait_closed()
        assert rejected.close_code == 1008
        assert len(sessions) == 1

        await first.connection.close()
        async with asyncio.timeout(1):
            while True:
                try:
                    admitted = await _open_client_websocket(client_config(config.port))
                except ConnectionClosed as error:
                    assert error.rcvd is not None and error.rcvd.code == 1008
                    await asyncio.sleep(0)
                    continue
                break
        assert len(sessions) == 2

        await admitted.connection.close()
        await server.close()

    run(scenario())


def test_factory_failure_is_connection_local_and_releases_slot() -> None:
    async def scenario() -> None:
        config = server_config(max_connections=2)
        sessions: list[linklab.ServerSession] = []
        handlers: list[RecordingHandler] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            if len(sessions) == 2:
                raise RuntimeError("factory failed")
            handler = RecordingHandler()
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        healthy = await _open_client_websocket(client_config(config.port))
        failed = await _open_client_websocket(client_config(config.port))
        await failed.connection.wait_closed()

        assert failed.connection.close_code == 1011
        assert healthy.connection.close_code is None
        assert len(handlers) == 1
        assert not handlers[0].events

        recovered = await _open_client_websocket(client_config(config.port))
        assert len(sessions) == 3
        assert len(handlers) == 2
        await asyncio.gather(healthy.connection.close(), recovered.connection.close())
        await server.close()

    run(scenario())


def test_serve_is_one_shot_and_context_manager_does_not_suppress() -> None:
    async def concurrent_serve() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        results = await asyncio.gather(server.serve(), server.serve(), return_exceptions=True)
        assert sum(result is None for result in results) == 1
        errors = [result for result in results if isinstance(result, RuntimeError)]
        assert len(errors) == 1
        with pytest.raises(RuntimeError, match="only be called once"):
            await server.serve()
        await server.close()

    async def context_manager() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        with pytest.raises(LookupError, match="body failed"):
            async with server as entered:
                assert entered is server
                raise LookupError("body failed")
        await server.wait_closed()
        with pytest.raises(OSError):
            await connect(f"ws://127.0.0.1:{config.port}", proxy=None)

    run(concurrent_serve())
    run(context_manager())


def test_voice_server_exact_async_surface_and_annotations() -> None:
    assert {name for name in vars(linklab.VoiceServer) if not name.startswith("_")} == {
        "serve",
        "close",
        "wait_closed",
    }
    assert inspect.signature(linklab.VoiceServer, eval_str=True) == inspect.Signature(
        parameters=(
            inspect.Parameter("config", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=linklab.ServerConfig),
            inspect.Parameter(
                "handler_factory",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=get_type_hints(linklab.VoiceServer.__init__)["handler_factory"],
            ),
            inspect.Parameter("logger", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=object),
        ),
        return_annotation=None,
    )
    for name in ("serve", "close", "wait_closed"):
        method = getattr(linklab.VoiceServer, name)
        assert inspect.iscoroutinefunction(method)
        assert get_type_hints(method)["return"] is type(None)
    assert get_type_hints(linklab.VoiceServer.__aenter__)["return"] is Self
    assert get_type_hints(linklab.VoiceServer.__aexit__)["return"] is bool


def test_server_close_is_bounded_when_handler_suppresses_cancellation() -> None:
    class StubbornHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.release = asyncio.Event()
            self.finished = asyncio.Event()

        async def on_conversation_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.ConversationStartedEvent,
        ) -> None:
            self.events.append(event)
            self.entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cancelled.set()
                await self.release.wait()
            finally:
                self.finished.set()

    async def scenario() -> None:
        config = server_config(close_timeout_s=0.01)
        handler = StubbornHandler()
        server = linklab.VoiceServer(config, lambda _session: handler, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await send_message(
            client.connection,
            linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word"),
        )
        await handler.entered.wait()

        started = time.monotonic()
        await asyncio.wait_for(server.close(), 0.2)
        assert time.monotonic() - started < 0.2
        assert handler.cancelled.is_set()

        handler.release.set()
        await handler.finished.wait()
        await client.connection.wait_closed()

    run(scenario())


def test_close_before_serve_is_idempotent_and_ssl_context_is_caller_owned() -> None:
    async def scenario() -> None:
        plain = linklab.VoiceServer(server_config(), lambda _session: RecordingHandler(), object())
        await asyncio.gather(plain.close(), plain.close())
        await plain.wait_closed()

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        config = server_config(ssl_context=context)
        server = linklab.VoiceServer(config, lambda _session: RecordingHandler(), object())
        await server.serve()
        assert config.ssl_context is context
        await server.close()
        await server.wait_closed()

    run(scenario())


def test_invalid_handler_result_closes_only_that_connection() -> None:
    async def scenario() -> None:
        config = server_config()
        server = linklab.VoiceServer(config, lambda _session: cast(Any, object()), object())
        await server.serve()
        handshake = await _open_client_websocket(client_config(config.port))
        await handshake.connection.wait_closed()
        assert handshake.connection.close_code == 1011
        await server.close()

    run(scenario())


@pytest.mark.parametrize("interrupted", [False, True])
def test_handler_dispatches_every_inbound_event_in_validated_order(interrupted: bool) -> None:
    class InspectingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.committed_during_audio: list[int] = []

        async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.committed_during_audio.append(session._core.validator._data.inputs[-1].committed_end_frame)
            self.events.append(event)

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = InspectingHandler()

        def factory(session: linklab.ServerSession) -> InspectingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        connection_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(connection_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(connection_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        source = bytearray(b"\x01\x02\x03\x04")
        await send_message(
            client.connection,
            linklab.InputAudioEvent(connection_id, input_id, 0, True, cast(bytes, source)),
        )
        source[:] = b"\x00\x00\x00\x00"
        await send_message(
            client.connection,
            linklab.InputAbortedEvent(connection_id, input_id, linklab.InputAbortReason.CAPTURE_FAILED),
        )
        await wait_for_count(handler.events, 4)

        session = sessions[0]
        assert cast(linklab.InputAudioEvent, handler.events[2]).audio == b"\x01\x02\x03\x04"
        assert handler.committed_during_audio == [2]
        committed = session._core.validator._data.inputs[-1].committed_end_frame
        assert committed == 2

        response_id = linklab.ResponseId(1)
        output_id = linklab.OutputId(1)
        session._core.enqueue_batch(
            (
                linklab.TranscriptFinalEvent(connection_id, input_id, ""),
                linklab.ResponseStartedEvent(connection_id, response_id, input_id, False),
                linklab.OutputStartedEvent(connection_id, response_id, output_id),
                linklab.OutputAudioEvent(connection_id, response_id, output_id, 0, b"\x00\x00"),
                linklab.OutputEndedEvent(connection_id, response_id, output_id, 1),
                linklab.ResponseEndedEvent(connection_id, response_id),
            ),
            lane=_QueueLane.CONTROL,
        )
        if interrupted:
            outcome: linklab.Message = linklab.PlaybackInterruptedEvent(
                connection_id,
                response_id,
                output_id,
                1,
                linklab.PlaybackPosition.EXACT,
                linklab.PlaybackInterruptReason.LOCAL_CANCEL,
            )
        else:
            outcome = linklab.PlaybackFinishedEvent(connection_id, response_id, output_id, 1)
        await send_message(client.connection, outcome)
        await send_message(
            client.connection,
            linklab.ConversationCancelledEvent(connection_id, linklab.ConversationCancelReason.USER),
        )
        await wait_for_count(handler.events, 6)

        assert [event.type for event in cast(list[linklab.Message], handler.events)] == [
            "conversation.start",
            "input.start",
            "input.audio",
            "input.abort",
            outcome.type,
            "conversation.cancel",
        ]
        await client.connection.close()
        await server.close()

    run(scenario())


def test_handler_queue_full_and_callback_failure_are_explicit_signals() -> None:
    class BlockingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def on_conversation_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.ConversationStartedEvent,
        ) -> None:
            self.events.append(event)

        async def on_input_audio(self, _session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.events.append(event)
            self.entered.set()
            await self.release.wait()

    async def scenario() -> None:
        config = server_config(input_queue_frames=2)
        sessions: list[linklab.ServerSession] = []
        handler = BlockingHandler()

        def factory(session: linklab.ServerSession) -> BlockingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0" * 4))
        await handler.entered.wait()
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 2, True, b"\0" * 4))
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 4, True, b"\0" * 4))

        session = sessions[0]
        signal = await asyncio.wait_for(session._handler_signals.get(), 1)
        assert isinstance(signal, _HandlerQueueFull)
        assert cast(linklab.InputAudioEvent, signal.event).start_frame == 4
        audio_snapshot, event_snapshot = session._events.snapshots()
        assert (audio_snapshot.capacity, audio_snapshot.occupancy, audio_snapshot.overflow_count) == (2, 0, 1)
        assert event_snapshot.overflow_count == 0

        assert await receive_messages(client, 5) == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.ErrorEvent(
                linklab.ErrorScope.INPUT,
                linklab.ErrorCode.INPUT_OVERFLOW,
                True,
                conversation_id,
                input_id,
            ),
            linklab.InputClosedEvent(conversation_id, input_id, 2, linklab.InputCloseReason.FAILED),
            linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
            linklab.StateEvent(conversation_id, 2, linklab.CoarseState.WAITING),
        ]

        handler.release.set()
        await asyncio.sleep(0)
        audio_events = [event for event in handler.events if isinstance(event, linklab.InputAudioEvent)]
        assert [event.start_frame for event in audio_events] == [0]
        next_input_id = linklab.InputId(2)
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, next_input_id, linklab.InputStartReason.SPEECH, 0),
        )
        await wait_for_count(handler.events, 4)
        assert handler.events[-1] == linklab.InputStartedEvent(
            conversation_id,
            next_input_id,
            linklab.InputStartReason.SPEECH,
            0,
        )
        await client.connection.close()
        await server.close()

    run(scenario())


def test_non_audio_handler_overflow_closes_only_affected_session() -> None:
    class BlockingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def on_conversation_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.ConversationStartedEvent,
        ) -> None:
            self.events.append(event)
            self.entered.set()
            await self.release.wait()

    async def scenario() -> None:
        config = server_config(max_connections=2, websocket_max_queue=1)
        handlers: list[RecordingHandler] = []

        def factory(_session: linklab.ServerSession) -> RecordingHandler:
            handler: RecordingHandler = BlockingHandler() if not handlers else RecordingHandler()
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        overloaded = await _open_client_websocket(client_config(config.port))
        healthy = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], handlers), 2)
        blocking = cast(BlockingHandler, handlers[0])
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)

        await send_message(overloaded.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await blocking.entered.wait()
        await send_message(
            overloaded.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(
            overloaded.connection,
            linklab.InputAbortedEvent(conversation_id, input_id, linklab.InputAbortReason.CAPTURE_FAILED),
        )
        await overloaded.connection.wait_closed()
        assert overloaded.connection.close_code == 1011

        await send_message(healthy.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await wait_for_count(handlers[1].events, 1)
        assert isinstance(handlers[1].events[0], linklab.ConversationStartedEvent)
        assert healthy.connection.close_code is None

        blocking.release.set()
        await healthy.connection.close()
        await server.close()

    run(scenario())


def test_conversation_handler_failure_emits_conversation_failure_sequence() -> None:
    class FailingHandler(RecordingHandler):
        async def on_conversation_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.ConversationStartedEvent,
        ) -> None:
            self.events.append(event)
            raise LookupError("conversation handler failed")

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> FailingHandler:
            sessions.append(session)
            return FailingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))

        assert await receive_messages(client, 2) == [
            linklab.ErrorEvent(
                linklab.ErrorScope.CONVERSATION,
                linklab.ErrorCode.CONVERSATION_FAILED,
                True,
                conversation_id,
            ),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.SERVER_FAILED),
        ]
        signal = await asyncio.wait_for(sessions[0]._handler_signals.get(), 1)
        assert isinstance(signal, _HandlerRaised)
        assert isinstance(signal.error, LookupError)
        assert client.connection.close_code is None
        await client.connection.close()
        await server.close()

    run(scenario())


@pytest.mark.parametrize("phase", ["started", "audio", "aborted"])
def test_input_handler_failure_emits_stt_failure_sequence(phase: str) -> None:
    class FailingHandler(RecordingHandler):
        async def on_input_started(
            self,
            _session: linklab.ServerSession,
            event: linklab.InputStartedEvent,
        ) -> None:
            self.events.append(event)
            if phase == "started":
                raise LookupError("input handler failed")

        async def on_input_audio(self, _session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.events.append(event)
            if phase == "audio":
                raise LookupError("input handler failed")

        async def on_input_aborted(
            self,
            _session: linklab.ServerSession,
            event: linklab.InputAbortedEvent,
        ) -> None:
            self.events.append(event)
            if phase == "aborted":
                raise LookupError("input handler failed")

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> FailingHandler:
            sessions.append(session)
            return FailingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        if phase != "started":
            await send_message(
                client.connection,
                linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0\0"),
            )
        if phase == "aborted":
            await send_message(
                client.connection,
                linklab.InputAbortedEvent(conversation_id, input_id, linklab.InputAbortReason.CAPTURE_FAILED),
            )

        signal = await asyncio.wait_for(sessions[0]._handler_signals.get(), 1)
        assert isinstance(signal, _HandlerRaised)
        messages = await receive_messages(client, 5)
        error_index = next(index for index, message in enumerate(messages) if isinstance(message, linklab.ErrorEvent))
        assert messages[error_index] == linklab.ErrorEvent(
            linklab.ErrorScope.INPUT,
            linklab.ErrorCode.STT_FAILED,
            True,
            conversation_id,
            input_id,
        )
        waiting_revision = 3 if phase == "aborted" else 2
        assert messages[-1] == linklab.StateEvent(conversation_id, waiting_revision, linklab.CoarseState.WAITING)
        if phase == "aborted":
            assert not any(isinstance(message, linklab.InputClosedEvent) for message in messages)
        else:
            closed = next(message for message in messages if isinstance(message, linklab.InputClosedEvent))
            assert closed.accepted_end_frame == (1 if phase == "audio" else 0)
            assert closed.reason is linklab.InputCloseReason.FAILED
        await client.connection.close()
        await server.close()

    run(scenario())


def test_playback_handler_failure_invalidates_writer_and_ends_playback_failed() -> None:
    class FailingHandler(RecordingHandler):
        async def on_playback_outcome(
            self,
            _session: linklab.ServerSession,
            event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
        ) -> None:
            self.events.append(event)
            raise LookupError("playback handler failed")

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> FailingHandler:
            sessions.append(session)
            return FailingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        session = sessions[0]
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        async with asyncio.timeout(1):
            while session._core.validator.state.input_id != input_id:
                await asyncio.sleep(0)
        await session.close_input(input_id, linklab.InputCloseReason.NO_SPEECH)
        await session.finalize_transcript(input_id, "")
        response = await session.start_response(input_id)
        output = await response.start_output()
        await output.send_audio(b"\0\0")
        await output.finish()
        await send_message(
            client.connection,
            linklab.PlaybackFinishedEvent(conversation_id, response.response_id, output.output_id, 1),
        )

        signal = await asyncio.wait_for(session._handler_signals.get(), 1)
        assert isinstance(signal, _HandlerRaised)
        messages = await receive_messages(client, 12)
        assert messages[-3:] == [
            linklab.ErrorEvent(
                linklab.ErrorScope.RESPONSE,
                linklab.ErrorCode.PLAYBACK_FAILED,
                True,
                conversation_id,
                response_id=response.response_id,
            ),
            linklab.ResponseCancelledEvent(
                conversation_id,
                response.response_id,
                linklab.ResponseCancelReason.PLAYBACK_FAILED,
            ),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.PLAYBACK_FAILED),
        ]
        with pytest.raises(linklab.WriterClosed):
            await response.finish()
        await client.connection.close()
        await server.close()

    run(scenario())


def test_handler_failure_does_not_overwrite_work_published_during_callback() -> None:
    class PublishingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.response: linklab.ResponseWriter | None = None

        async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.events.append(event)
            await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
            await session.finalize_transcript(event.input_id, "speech")
            self.response = await session.start_response(event.input_id)
            raise LookupError("late input callback failure")

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = PublishingHandler()

        def factory(session: linklab.ServerSession) -> PublishingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0\0"))

        signal = await asyncio.wait_for(sessions[0]._handler_signals.get(), 1)
        assert isinstance(signal, _HandlerRaised)
        messages = await receive_messages(client, 6)
        assert not any(isinstance(message, linklab.ErrorEvent) for message in messages)
        assert isinstance(messages[-2], linklab.ResponseStartedEvent)
        assert messages[-1] == linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING)
        assert handler.response is not None
        await handler.response.send_text_delta("still current")
        assert await receive_messages(client, 1) == [
            linklab.ResponseTextDeltaEvent(conversation_id, handler.response.response_id, 0, "still current")
        ]
        await client.connection.close()
        await server.close()

    run(scenario())


def test_internal_dispatch_failure_closes_only_affected_session_1011(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        config = server_config(max_connections=2)
        sessions: list[linklab.ServerSession] = []
        handlers: list[RecordingHandler] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            handler = RecordingHandler()
            handlers.append(handler)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        failed = await _open_client_websocket(client_config(config.port))
        healthy = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 2)

        def fail_prepare(_event: linklab.Message) -> bool:
            raise RuntimeError("internal dispatch failed")

        monkeypatch.setattr(sessions[0], "_prepare_handler_event", fail_prepare)
        conversation_id = linklab.ConversationId(1)
        await send_message(failed.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await failed.connection.wait_closed()
        assert failed.connection.close_code == 1011

        await send_message(healthy.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await wait_for_count(handlers[1].events, 1)
        assert healthy.connection.close_code is None
        await healthy.connection.close()
        await server.close()

    run(scenario())


def test_endpoint_late_pcm_and_identical_terminal_events_are_not_dispatched() -> None:
    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = RecordingHandler()

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0" * 4))
        await wait_for_count(handler.events, 3)

        sessions[0]._core.enqueue_batch(
            (linklab.InputClosedEvent(conversation_id, input_id, 2, linklab.InputCloseReason.ENDPOINT),),
            lane=_QueueLane.CONTROL,
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 2, False, b"\0" * 4))
        cancel = linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.USER)
        await send_message(client.connection, cancel)
        await send_message(client.connection, cancel)
        await wait_for_count(handler.events, 4)
        await asyncio.sleep(0.01)

        assert [event.type for event in cast(list[linklab.Message], handler.events)] == [
            "conversation.start",
            "input.start",
            "input.audio",
            "conversation.cancel",
        ]
        await client.connection.close()
        await server.close()

    run(scenario())


def test_server_session_rejects_use_from_another_event_loop() -> None:
    retained: list[linklab.ServerSession] = []

    async def create_session() -> None:
        config = server_config()

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            retained.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], retained), 1)
        await client.connection.close()
        await server.close()

    async def wrong_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            await retained[0]._close_dispatcher()
        with pytest.raises(RuntimeError, match="different event loop"):
            await retained[0].end_conversation(linklab.ConversationEndReason.CANCELLED)

    run(create_session())
    run(wrong_loop())


def test_session_closes_at_callback_boundary_and_orders_transcripts_and_state() -> None:
    class ClosingHandler(RecordingHandler):
        def __init__(self) -> None:
            super().__init__()
            self.closed = asyncio.Event()

        async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
            self.events.append(event)
            await session.update_transcript(event.input_id, 1, "hello", "en")
            before = session._core.validator._data
            with pytest.raises(linklab.ProtocolViolation, match="revision"):
                await session.update_transcript(event.input_id, 3, "skipped")
            assert session._core.validator._data == before
            await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
            await session.finalize_transcript(event.input_id, "hello", "en")
            self.closed.set()

    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []
        handler = ClosingHandler()

        def factory(session: linklab.ServerSession) -> ClosingHandler:
            sessions.append(session)
            return handler

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 0, True, b"\0" * 4))
        await send_message(client.connection, linklab.InputAudioEvent(conversation_id, input_id, 2, True, b"\0" * 4))
        await handler.closed.wait()

        messages = await receive_messages(client, 5)
        assert messages == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.TranscriptUpdateEvent(conversation_id, input_id, 1, "hello", "en"),
            linklab.InputClosedEvent(conversation_id, input_id, 2, linklab.InputCloseReason.ENDPOINT),
            linklab.StateEvent(conversation_id, 2, linklab.CoarseState.PROCESSING),
            linklab.TranscriptFinalEvent(conversation_id, input_id, "hello", "en"),
        ]
        await asyncio.sleep(0)
        assert [event.type for event in cast(list[linklab.Message], handler.events)] == [
            "conversation.start",
            "input.start",
            "input.audio",
        ]

        session = sessions[0]
        await session.end_conversation(linklab.ConversationEndReason.COMPLETED)
        assert await receive_messages(client, 1) == [
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.COMPLETED)
        ]
        with pytest.raises(linklab.ProtocolViolation, match="open conversation"):
            await session.finalize_transcript(input_id, "late")

        await client.connection.close()
        await server.close()

    run(scenario())


def test_input_fail_after_close_emits_exact_recovery_batch_and_rejects_repeat() -> None:
    async def scenario() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        session = sessions[0]
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        async with asyncio.timeout(1):
            while session._core.validator.state.input_id != input_id:
                await asyncio.sleep(0)

        before = session._core.validator._data
        with pytest.raises(linklab.ProtocolViolation, match="no input"):
            await session.close_input(linklab.InputId(2), linklab.InputCloseReason.NO_SPEECH)
        assert session._core.validator._data == before
        await session.close_input(input_id, linklab.InputCloseReason.NO_SPEECH)
        await session.fail(linklab.ErrorScope.INPUT, linklab.ErrorCode.STT_FAILED, "stt unavailable")
        messages = await receive_messages(client, 6)
        assert messages == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.InputClosedEvent(conversation_id, input_id, 0, linklab.InputCloseReason.NO_SPEECH),
            linklab.StateEvent(conversation_id, 2, linklab.CoarseState.PROCESSING),
            linklab.ErrorEvent(
                linklab.ErrorScope.INPUT,
                linklab.ErrorCode.STT_FAILED,
                True,
                conversation_id,
                input_id,
                message="stt unavailable",
            ),
            linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.WAITING),
        ]

        before = session._core.validator._data
        with pytest.raises(linklab.ProtocolViolation, match="recoverable input"):
            await session.fail(linklab.ErrorScope.INPUT, linklab.ErrorCode.STT_FAILED)
        assert session._core.validator._data == before
        with pytest.raises(linklab.ProtocolViolation, match="live response"):
            await session.fail(linklab.ErrorScope.RESPONSE, linklab.ErrorCode.GENERATION_FAILED)

        await client.connection.close()
        await server.close()

    run(scenario())


def test_end_open_input_and_client_cancel_emit_terminal_batches() -> None:
    async def server_end() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        async with asyncio.timeout(1):
            while sessions[0]._core.validator.state.input_id != input_id:
                await asyncio.sleep(0)
        await sessions[0].end_conversation(linklab.ConversationEndReason.CANCELLED)
        assert await receive_messages(client, 4) == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.InputClosedEvent(conversation_id, input_id, 0, linklab.InputCloseReason.FAILED),
            linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.CANCELLED),
        ]
        await client.connection.close()
        await server.close()

    async def client_cancel() -> None:
        config = server_config()
        handler = RecordingHandler()
        server = linklab.VoiceServer(config, lambda _session: handler, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        await send_message(
            client.connection,
            linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.USER),
        )
        assert await receive_messages(client, 2) == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.CANCELLED),
        ]
        await wait_for_count(handler.events, 3)
        assert isinstance(handler.events[-1], linklab.ConversationCancelledEvent)
        await client.connection.close()
        await server.close()

    run(server_end())
    run(client_cancel())


def test_session_fail_resolves_connection_conversation_and_response_scopes() -> None:
    async def conversation_failure() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        async with asyncio.timeout(1):
            while sessions[0]._core.validator.state.input_id != input_id:
                await asyncio.sleep(0)
        await sessions[0].fail(
            linklab.ErrorScope.CONVERSATION,
            linklab.ErrorCode.CONVERSATION_FAILED,
            "pipeline failed",
        )
        assert await receive_messages(client, 5) == [
            linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),
            linklab.ErrorEvent(
                linklab.ErrorScope.CONVERSATION,
                linklab.ErrorCode.CONVERSATION_FAILED,
                True,
                conversation_id,
                message="pipeline failed",
            ),
            linklab.InputClosedEvent(conversation_id, input_id, 0, linklab.InputCloseReason.FAILED),
            linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
            linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.SERVER_FAILED),
        ]
        await client.connection.close()
        await server.close()

    async def response_failure() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        session = sessions[0]
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        response_id = linklab.ResponseId(1)
        await send_message(client.connection, linklab.ConversationStartedEvent(conversation_id, "wake_word"))
        await send_message(
            client.connection,
            linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0),
        )
        async with asyncio.timeout(1):
            while session._core.validator.state.input_id != input_id:
                await asyncio.sleep(0)
        await session.close_input(input_id, linklab.InputCloseReason.NO_SPEECH)
        await session.finalize_transcript(input_id, "")
        session._core.enqueue_batch(
            (linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False),),
            lane=_QueueLane.CONTROL,
        )
        await session.fail(linklab.ErrorScope.RESPONSE, linklab.ErrorCode.GENERATION_FAILED)
        messages = await receive_messages(client, 8)
        assert messages[-3:] == [
            linklab.ErrorEvent(
                linklab.ErrorScope.RESPONSE,
                linklab.ErrorCode.GENERATION_FAILED,
                True,
                conversation_id,
                response_id=response_id,
            ),
            linklab.ResponseCancelledEvent(
                conversation_id,
                response_id,
                linklab.ResponseCancelReason.GENERATION_FAILED,
            ),
            linklab.StateEvent(conversation_id, 3, linklab.CoarseState.WAITING),
        ]
        await client.connection.close()
        await server.close()

    async def connection_failure() -> None:
        config = server_config()
        sessions: list[linklab.ServerSession] = []

        def factory(session: linklab.ServerSession) -> RecordingHandler:
            sessions.append(session)
            return RecordingHandler()

        server = linklab.VoiceServer(config, factory, object())
        await server.serve()
        client = await _open_client_websocket(client_config(config.port))
        await wait_for_count(cast(list[object], sessions), 1)
        await sessions[0].fail(linklab.ErrorScope.CONNECTION, linklab.ErrorCode.PROTOCOL_STATE, "local failure")
        assert await receive_messages(client, 1) == [
            linklab.ErrorEvent(
                linklab.ErrorScope.CONNECTION,
                linklab.ErrorCode.PROTOCOL_STATE,
                True,
                message="local failure",
            )
        ]
        await client.connection.wait_closed()
        assert client.connection.close_code == 1002
        await server.close()

    run(conversation_failure())
    run(response_failure())
    run(connection_failure())
