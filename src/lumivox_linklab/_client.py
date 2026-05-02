from __future__ import annotations

import asyncio
from typing import Self, Protocol, runtime_checkable
from contextlib import suppress
from dataclasses import dataclass

from ._enums import (
    EndpointRole,
    ConnectionState,
    InputAbortReason,
    PlaybackPosition,
    AudioSubmitResult,
    PlaybackInterruptReason,
    ConversationCancelReason,
)
from ._config import AudioFormat, ClientConfig
from ._errors import QueueOverflow, ConnectionClosed
from ._values import OutputId, AnnotatedAudio
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
    InputClosedEvent,
    OutputAudioEvent,
    OutputEndedEvent,
    OutputStartedEvent,
    ResponseEndedEvent,
    ConnectionStateEvent,
    ResponseStartedEvent,
    TranscriptFinalEvent,
    TranscriptUpdateEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ResponseTextDeltaEvent,
    ResponseTextFinalEvent,
)
from ._protocol import _TransitionResult
from ._handshake import _open_client_websocket
from ._transport import _QueueLane, _OutboundBatch, _TransportCore
from ._client_ingress import _IngressLane, _IngressBatch, _ClientAudioIngress

_CONTROL_CAPACITY = 16


@runtime_checkable
class ClientCallbacks(Protocol):
    async def on_connection_state(self, event: ConnectionStateEvent) -> None: ...

    async def on_conversation_state(self, event: StateEvent) -> None: ...

    async def on_transcript_update(self, event: TranscriptUpdateEvent) -> None: ...

    async def on_transcript_final(self, event: TranscriptFinalEvent) -> None: ...

    async def on_response_started(self, event: ResponseStartedEvent) -> None: ...

    async def on_response_text_delta(self, event: ResponseTextDeltaEvent) -> None: ...

    async def on_response_text_final(self, event: ResponseTextFinalEvent) -> None: ...

    async def on_response_ended(self, event: ResponseEndedEvent) -> None: ...

    async def on_response_cancelled(self, event: ResponseCancelledEvent) -> None: ...

    async def on_output_started(self, event: OutputStartedEvent) -> None: ...

    async def on_output_audio(self, event: OutputAudioEvent) -> None: ...

    async def on_output_ended(self, event: OutputEndedEvent) -> None: ...

    async def on_conversation_ended(self, event: ConversationEndedEvent) -> None: ...

    async def on_error(self, event: ErrorEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class _InboundEvent:
    message: Message
    transition: _TransitionResult


class VoiceClient:
    def __init__(self, config: ClientConfig, callbacks: ClientCallbacks, logger: object) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("VoiceClient must be created in a running event loop") from error
        if not isinstance(config, ClientConfig):
            raise TypeError("config must be ClientConfig")
        if not isinstance(callbacks, ClientCallbacks):
            raise TypeError("callbacks must implement ClientCallbacks")
        self._config = config
        self._callbacks = callbacks
        self._logger = logger
        self._wake = asyncio.Event()
        self._closed = asyncio.Event()
        self._closed.set()
        self._events: asyncio.Queue[_InboundEvent] = asyncio.Queue(maxsize=config.websocket_max_queue)
        self._ingress = _ClientAudioIngress(
            config,
            notify=self._notify_handoff,
            control_capacity=_CONTROL_CAPACITY,
        )
        self._core: _TransportCore | None = None
        self._connect_task: asyncio.Task[object] | None = None
        self._handoff_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._connect_started = False
        self._output_format: AudioFormat | None = None
        self._transport_loss: BaseException | None = None

    @property
    def connection_state(self) -> ConnectionState:
        return self._ingress.state.connection_state

    @property
    def output_format(self) -> AudioFormat | None:
        return self._output_format

    async def connect(self) -> None:
        self._check_loop()
        if self._connect_started:
            raise RuntimeError("VoiceClient.connect() may only be called once")
        self._connect_started = True
        self._close_task = None
        self._closed.clear()
        self._ingress.transport_connected()
        current = asyncio.current_task()
        assert current is not None
        self._connect_task = current
        core: _TransportCore | None = None
        try:
            handshake = await _open_client_websocket(self._config)
            core = _TransportCore(
                handshake.connection,
                role=EndpointRole.CLIENT,
                limits=handshake.server_hello.limits,
                data_capacity=self._config.input_queue_frames,
                control_capacity=_CONTROL_CAPACITY,
                close_timeout_s=self._config.close_timeout_s,
                on_transition=self._accept_inbound,
                on_transport_loss=self._record_transport_loss,
                ping_interval_s=self._config.ping_interval_s,
                ping_timeout_s=self._config.ping_timeout_s,
                on_outbound_space=self._wake.set,
                occupancy_unit="frames",
            )
            self._core = core
            self._ingress.complete_handshake(handshake.client_hello, handshake.server_hello)
            self._output_format = handshake.server_hello.output_format
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
            self._handoff_task = self._loop.create_task(self._run_handoff(), name="linklab-client-handoff")
            self._monitor_task = self._loop.create_task(self._monitor_transport(), name="linklab-client-monitor")
            self._wake.set()
        except BaseException:
            if core is not None:
                with suppress(Exception):
                    await core.close(1011)
            self._output_format = None
            self._ingress.stop()
            self._closed.set()
            raise
        finally:
            self._connect_task = None

    async def close(self) -> None:
        self._check_loop()
        if self._close_task is None:
            self._close_task = self._loop.create_task(self._close(), name="linklab-client-close")
        await asyncio.shield(self._close_task)

    async def wait_closed(self) -> None:
        self._check_loop()
        await self._closed.wait()

    def submit_annotated_audio(self, audio: AnnotatedAudio) -> AudioSubmitResult:
        return self._ingress.submit(audio)

    def abort_input(self, reason: InputAbortReason) -> bool:
        return self._ingress.abort_input(reason)

    def cancel_conversation(self, reason: ConversationCancelReason) -> bool:
        return self._ingress.cancel_conversation(reason)

    def playback_finished(self, output_id: OutputId, played_frames: int) -> bool:
        return self._ingress.playback_finished(output_id, played_frames)

    def playback_interrupted(
        self,
        output_id: OutputId,
        played_frames: int,
        position: PlaybackPosition,
        reason: PlaybackInterruptReason,
    ) -> bool:
        return self._ingress.playback_interrupted(output_id, played_frames, position, reason)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> bool:
        await self.close()
        return False

    async def _close(self) -> None:
        connect_task = self._connect_task
        if self.connection_state is ConnectionState.HANDSHAKING and connect_task is not None:
            self._ingress.begin_close()
            connect_task.cancel()
            with suppress(asyncio.CancelledError):
                await connect_task
            await self._closed.wait()
            return

        core = self._core
        if core is None:
            return
        if self.connection_state in (ConnectionState.HANDSHAKING, ConnectionState.READY):
            self._ingress.begin_close()
        await core.close(1000)
        monitor = self._monitor_task
        if monitor is not None and monitor is not asyncio.current_task():
            await monitor

    async def _run_handoff(self) -> None:
        core = self._core
        assert core is not None
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                failure = self._ingress.failure or self._ingress.notification_error
                if failure is not None:
                    await core.close(1011)
                    return
                while (batch := self._ingress.next_batch()) is not None:
                    lane = _QueueLane.DATA if batch.lane is _IngressLane.DATA else _QueueLane.CONTROL
                    weight = batch.audio_frames if batch.lane is _IngressLane.DATA else 1
                    try:
                        core.enqueue_batch(batch.messages, lane=lane, weight=weight, tag=batch)
                    except QueueOverflow:
                        break
                    except ConnectionClosed:
                        return
                    self._ingress.acknowledge(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            await core.close(1011)

    async def _monitor_transport(self) -> None:
        core = self._core
        assert core is not None
        try:
            await core.wait_closed()
        finally:
            handoff = self._handoff_task
            if handoff is not None and handoff is not asyncio.current_task() and not handoff.done():
                handoff.cancel()
                with suppress(asyncio.CancelledError):
                    await handoff
            self._output_format = None
            self._ingress.stop()
            self._closed.set()

    def _accept_inbound(self, message: Message, _transport_transition: _TransitionResult) -> None:
        transition = self._ingress.accept_inbound(message)
        core = self._core
        assert core is not None
        if isinstance(message, InputClosedEvent):
            core.discard_queued(lambda batch: self._is_unsent_input_audio(batch, message))
        try:
            self._events.put_nowait(_InboundEvent(message, transition))
        except asyncio.QueueFull as error:
            raise QueueOverflow("client inbound event handoff capacity exceeded") from error

    @staticmethod
    def _is_unsent_input_audio(batch: _OutboundBatch, closed: InputClosedEvent) -> bool:
        tag = batch.tag
        return isinstance(tag, _IngressBatch) and tag.lane is _IngressLane.DATA and tag.input_id == closed.input_id

    def _record_transport_loss(self, error: BaseException) -> None:
        self._transport_loss = error

    def _notify_handoff(self) -> None:
        self._loop.call_soon_threadsafe(self._wake.set)

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("VoiceClient operation requires its event loop") from error
        if loop is not self._loop:
            raise RuntimeError("VoiceClient is bound to a different event loop")
