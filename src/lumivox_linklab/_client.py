from __future__ import annotations

import asyncio
from enum import StrEnum
from random import random as _random
from typing import Self, Literal, Protocol, runtime_checkable
from threading import Lock
from contextlib import suppress
from collections import deque
from dataclasses import replace, dataclass
from collections.abc import Callable

from lumivox_core.logger import Logger
from websockets.exceptions import ConnectionClosed as WebSocketConnectionClosed

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
from ._discovery import _ServiceBrowser, _open_service_browser
from ._handshake import _ClientHandshake, _open_client_websocket
from ._transport import _QueueLane, _OutboundBatch, _TransportCore
from ._observability import _Observer, _QueueSnapshot, _lifecycle_boundary
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
    message: Message | ConnectionStateEvent
    transition: _TransitionResult | None = None


class _CallbackPath(StrEnum):
    EVENT = "event"
    OUTPUT = "output"
    CONTROL = "control"


class _CallbackQueue:
    def __init__(
        self,
        event_capacity: int,
        output_capacity_frames: int,
        notify: Callable[[], None],
        clock: Callable[[], float],
    ) -> None:
        self._event_capacity = event_capacity
        self._output_capacity_frames = output_capacity_frames
        self._notify = notify
        self._clock = clock
        self._lock = Lock()
        self._items: deque[tuple[_InboundEvent, _CallbackPath, int, float]] = deque()
        self._event_occupancy = 0
        self._output_occupancy_frames = 0
        self._control_occupancy = 0
        self._overflows = {path: 0 for path in _CallbackPath}
        self._closed = False

    def put_nowait(self, item: _InboundEvent, path: _CallbackPath, frames: int = 0) -> bool:
        with self._lock:
            if self._closed:
                return False
            if path is _CallbackPath.OUTPUT:
                if self._output_occupancy_frames + frames > self._output_capacity_frames:
                    self._overflows[path] += 1
                    return False
                self._output_occupancy_frames += frames
            elif path is _CallbackPath.CONTROL:
                if self._control_occupancy >= _CONTROL_CAPACITY:
                    self._overflows[path] += 1
                    return False
                self._control_occupancy += 1
            else:
                if self._event_occupancy >= self._event_capacity:
                    self._overflows[path] += 1
                    return False
                self._event_occupancy += 1
            self._items.append((item, path, frames, self._clock()))
        self._notify()
        return True

    def get_nowait(self) -> tuple[_InboundEvent, _CallbackPath] | None:
        with self._lock:
            if not self._items:
                return None
            item, path, frames, _ = self._items.popleft()
            if path is _CallbackPath.OUTPUT:
                self._output_occupancy_frames -= frames
            elif path is _CallbackPath.CONTROL:
                self._control_occupancy -= 1
            else:
                self._event_occupancy -= 1
            return item, path

    def discard(self, predicate: Callable[[_InboundEvent], bool]) -> int:
        with self._lock:
            discarded = 0
            retained: deque[tuple[_InboundEvent, _CallbackPath, int, float]] = deque()
            while self._items:
                item, path, frames, enqueued_at = self._items.popleft()
                if predicate(item):
                    discarded += 1
                    if path is _CallbackPath.OUTPUT:
                        self._output_occupancy_frames -= frames
                    elif path is _CallbackPath.CONTROL:
                        self._control_occupancy -= 1
                    else:
                        self._event_occupancy -= 1
                else:
                    retained.append((item, path, frames, enqueued_at))
            self._items = retained
            return discarded

    def snapshots(self) -> tuple[_QueueSnapshot, _QueueSnapshot, _QueueSnapshot]:
        with self._lock:
            now = self._clock()

            def snapshot(path: _CallbackPath, capacity: int, occupancy: int, unit: str) -> _QueueSnapshot:
                oldest = next((item for item in self._items if item[1] is path), None)
                residence = 0.0 if oldest is None else max(0.0, (now - oldest[3]) * 1_000)
                return _QueueSnapshot(
                    f"client.callbacks.{path.value}",
                    capacity,
                    occupancy,
                    self._overflows[path],
                    residence,
                    unit,
                )

            return (
                snapshot(_CallbackPath.EVENT, self._event_capacity, self._event_occupancy, "events"),
                snapshot(
                    _CallbackPath.OUTPUT,
                    self._output_capacity_frames,
                    self._output_occupancy_frames,
                    "frames",
                ),
                snapshot(_CallbackPath.CONTROL, _CONTROL_CAPACITY, self._control_occupancy, "events"),
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._notify()

    def set_output_capacity(self, frames: int) -> None:
        with self._lock:
            if self._output_occupancy_frames:
                raise RuntimeError("callback output capacity cannot change after enqueue")
            self._output_capacity_frames = frames

    @property
    def closed_and_empty(self) -> bool:
        with self._lock:
            return self._closed and not self._items

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)


class VoiceClient:
    def __init__(self, config: ClientConfig, callbacks: ClientCallbacks, logger: Logger) -> None:
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
        self._callback_wake = asyncio.Event()
        self._closed = asyncio.Event()
        self._closed.set()
        self._conversation_terminal = asyncio.Event()
        maximum_output_frames = max(format_.sample_rate_hz for format_ in config.output_formats)
        playback_capacity = max(1, maximum_output_frames * config.playback_queue_ms // 1_000)
        self._events = _CallbackQueue(
            config.websocket_max_queue,
            playback_capacity,
            self._notify_callbacks,
            self._loop.time,
        )
        self._ingress = _ClientAudioIngress(
            config,
            notify=self._notify_handoff,
            control_capacity=_CONTROL_CAPACITY,
            clock=self._loop.time,
        )
        self._core: _TransportCore | None = None
        self._connect_task: asyncio.Task[object] | None = None
        self._handoff_task: asyncio.Task[None] | None = None
        self._callback_task: asyncio.Task[None] | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._reconnect_stop = asyncio.Event()
        self._discovery: _ServiceBrowser | None = None
        self._reconnect_discovery_revision = 0
        self._connect_started = False
        self._output_format: AudioFormat | None = None
        self._transport_loss: BaseException | None = None
        self._local_failure: BaseException | None = None
        self._connection_serial = 0
        self._observer: _Observer | None = None

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
        self._reconnect_stop.clear()
        self._closed.clear()
        self._start_callback_dispatcher()
        current = asyncio.current_task()
        assert current is not None
        self._connect_task = current
        try:
            await self._establish_connection()
            self._monitor_task = self._loop.create_task(self._monitor_connections(), name="linklab-client-monitor")
        except BaseException:
            await self._finish()
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
        response_id = self._ingress.state.response_id
        result = self._ingress.submit(audio)
        if result is AudioSubmitResult.ACCEPTED and response_id is not None:
            discarded = self._events.discard(
                lambda item: not isinstance(item.message, ConnectionStateEvent)
                and not self._ingress.is_current(item.message)
            )
            if discarded:
                self._observe_callback_queues("stale_output_discard")
        self._schedule_ingress_observation()
        return result

    def abort_input(self, reason: InputAbortReason) -> bool:
        accepted = self._ingress.abort_input(reason)
        self._schedule_ingress_observation()
        return accepted

    def cancel_conversation(self, reason: ConversationCancelReason) -> bool:
        accepted = self._ingress.cancel_conversation(reason)
        self._schedule_ingress_observation()
        return accepted

    def playback_finished(self, output_id: OutputId, played_frames: int) -> bool:
        accepted = self._ingress.playback_finished(output_id, played_frames)
        self._schedule_ingress_observation()
        return accepted

    def playback_interrupted(
        self,
        output_id: OutputId,
        played_frames: int,
        position: PlaybackPosition,
        reason: PlaybackInterruptReason,
    ) -> bool:
        accepted = self._ingress.playback_interrupted(output_id, played_frames, position, reason)
        self._schedule_ingress_observation()
        return accepted

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> Literal[False]:
        await self.close()
        return False

    async def _close(self) -> None:
        if self._observer is not None:
            self._observer.lifecycle("client_shutdown_started")
        self._reconnect_stop.set()
        connect_task = self._connect_task
        if self.connection_state is ConnectionState.HANDSHAKING and connect_task is not None:
            self._ingress.begin_close()
            self._queue_connection_state(ConnectionState.CLOSING)
            connect_task.cancel()
            with suppress(asyncio.CancelledError):
                await connect_task
            await self._closed.wait()
            return

        core = self._core
        if core is None:
            monitor = self._monitor_task
            if monitor is not None and monitor is not asyncio.current_task():
                if self.connection_state is ConnectionState.HANDSHAKING:
                    self._ingress.begin_close()
                    self._queue_connection_state(ConnectionState.CLOSING)
                    monitor.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor
            return
        if core.outcome is not None:
            await core.wait_closed()
            monitor = self._monitor_task
            if monitor is not None and monitor is not asyncio.current_task():
                await monitor
            return
        shutdown_messages = self._ingress.prepare_shutdown()
        if self.connection_state in (ConnectionState.HANDSHAKING, ConnectionState.READY):
            self._queue_connection_state(ConnectionState.CLOSING)
        if shutdown_messages:
            if self._events.discard(lambda item: isinstance(item.message, OutputAudioEvent)):
                self._observe_callback_queues("shutdown_output_discard")
            self._conversation_terminal.clear()
            core.seal_orderly(shutdown_messages)
            try:
                async with asyncio.timeout(self._config.close_timeout_s / 2):
                    await self._conversation_terminal.wait()
            except TimeoutError:
                pass
        self._ingress.begin_close()
        await core.close(1000, drain=bool(shutdown_messages))
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
                    except QueueOverflow as error:
                        if lane is _QueueLane.CONTROL:
                            core._request_close(1011, drain=False, loss=error)
                            return
                        break
                    except ConnectionClosed:
                        return
                    self._ingress.acknowledge(batch)
                    self._observe_ingress_queues()
        except asyncio.CancelledError:
            raise
        except Exception:
            await core.close(1011)

    async def _monitor_connections(self) -> None:
        backoff = self._config.reconnect_initial_s
        try:
            while True:
                core = self._core
                assert core is not None
                await core.wait_closed()
                outcome = core.outcome
                if self._discovery is not None:
                    self._reconnect_discovery_revision = self._discovery.revision
                await self._cleanup_connection(core)
                self._queue_connection_state(ConnectionState.DISCONNECTED, self._transport_reason())

                if (
                    self._reconnect_stop.is_set()
                    or self._local_failure is not None
                    or not self._config.reconnect
                    or outcome is None
                    or not outcome.reconnect_eligible
                ):
                    if self._observer is not None:
                        self._observer.lifecycle(
                            "reconnect_decision",
                            eligible=False,
                            close_code=None if outcome is None else outcome.close_code,
                        )
                    return

                while True:
                    delay = _random() * backoff
                    if self._observer is not None:
                        self._observer.lifecycle("reconnect_scheduled", delay_s=delay, maximum_delay_s=backoff)
                    if not await self._wait_reconnect_delay(delay):
                        return
                    if self._reconnect_stop.is_set():
                        return
                    backoff = min(backoff * 2, self._config.reconnect_max_s)
                    try:
                        if self._observer is not None:
                            self._observer.lifecycle("reconnect_attempt_started")
                        await self._establish_connection()
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        self._transport_loss = error
                        if self._observer is not None:
                            self._observer.failure("reconnect_attempt_failed", error)
                        if not _reconnect_failure_eligible(error):
                            return
                        if self._discovery is not None:
                            self._reconnect_discovery_revision = self._discovery.revision
                        continue
                    backoff = self._config.reconnect_initial_s
                    if self._observer is not None:
                        self._observer.lifecycle("reconnect_succeeded")
                    break
        finally:
            await self._finish()

    async def _establish_connection(self) -> None:
        self._transport_loss = None
        self._connection_serial += 1
        self._observer = _Observer(
            self._logger,
            endpoint_role=EndpointRole.CLIENT.value,
            connection_id=f"client-{self._connection_serial}",
            clock=self._loop.time,
        )
        self._ingress.transport_connected()
        self._observer.lifecycle("handshake_started")
        self._queue_connection_state(ConnectionState.HANDSHAKING)
        core: _TransportCore | None = None
        try:
            handshake = await self._open_connection()
            self._observer.lifecycle(
                "handshake_completed",
                output_sample_rate_hz=handshake.server_hello.output_format.sample_rate_hz,
            )
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
                observer=self._observer,
            )
            self._core = core
            self._ingress.complete_handshake(handshake.client_hello, handshake.server_hello)
            self._output_format = handshake.server_hello.output_format
            playback_capacity = max(
                1,
                handshake.server_hello.output_format.sample_rate_hz * self._config.playback_queue_ms // 1_000,
            )
            self._events.set_output_capacity(playback_capacity)
            core.start_handshaken(handshake.client_hello, handshake.server_hello)
            self._handoff_task = self._loop.create_task(self._run_handoff(), name="linklab-client-handoff")
            self._wake.set()
            self._queue_connection_state(ConnectionState.READY)
            self._observe_all_queues("ready")
            if self._local_failure is not None:
                core._request_close(1011, drain=False, loss=self._local_failure)
        except BaseException as error:
            self._observer.failure("handshake_failed", error)
            self._transport_loss = error
            if core is not None:
                with suppress(Exception):
                    await core.close(1011)
            if self._core is core:
                self._core = None
            self._output_format = None
            self._ingress.stop()
            self._observe_all_queues("connection_establishment_failed")
            self._queue_connection_state(ConnectionState.DISCONNECTED, self._transport_reason())
            raise

    async def _cleanup_connection(self, core: _TransportCore) -> None:
        handoff = self._handoff_task
        if handoff is not None and handoff is not asyncio.current_task() and not handoff.done():
            handoff.cancel()
            with suppress(asyncio.CancelledError):
                await handoff
        self._handoff_task = None
        if self._events.discard(lambda item: not isinstance(item.message, ConnectionStateEvent)):
            self._observe_callback_queues("connection_cleanup")
        self._output_format = None
        self._ingress.stop()
        self._observe_ingress_queues("connection_cleanup")
        self._conversation_terminal.set()
        if self._core is core:
            self._core = None

    async def _wait_reconnect_delay(self, delay: float) -> bool:
        discovery = self._discovery
        if discovery is None:
            try:
                async with asyncio.timeout(delay):
                    await self._reconnect_stop.wait()
            except TimeoutError:
                return True
            return False

        stop = self._loop.create_task(self._reconnect_stop.wait())
        update = self._loop.create_task(discovery.wait_for_update(self._reconnect_discovery_revision))
        try:
            done, _ = await asyncio.wait((stop, update), timeout=delay, return_when=asyncio.FIRST_COMPLETED)
            return not (stop in done and stop.result())
        finally:
            for task in (stop, update):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stop, update, return_exceptions=True)

    async def _open_connection(self) -> _ClientHandshake:
        if self._config.uri is not None:
            return await _open_client_websocket(self._config)
        service_id = self._config.discovery_service_id
        assert service_id is not None
        if self._discovery is None:
            if self._observer is not None:
                self._observer.lifecycle("discovery_browser_started", service_id=service_id)
            self._discovery = await _open_service_browser(service_id)
        if self._observer is not None:
            self._observer.lifecycle("discovery_resolution_started", service_id=service_id)
        candidates = await self._discovery.resolve(self._config.discovery_timeout_s)
        if self._observer is not None:
            self._observer.lifecycle("discovery_resolution_completed", candidate_count=len(candidates))
        last_error: BaseException | None = None
        async with asyncio.timeout(self._config.connect_timeout_s):
            for candidate_index, uri in enumerate(candidates):
                try:
                    if self._observer is not None:
                        self._observer.lifecycle("discovery_candidate_attempted", candidate_index=candidate_index)
                    return await _open_client_websocket(replace(self._config, uri=uri))
                except asyncio.CancelledError:
                    raise
                except (OSError, TimeoutError) as error:
                    last_error = error
        if last_error is not None:
            raise last_error
        raise OSError("discovery returned no usable endpoints")

    async def _finish(self) -> None:
        self._observe_all_queues("client_finish")
        core = self._core
        if core is not None:
            await self._cleanup_connection(core)
            if self.connection_state is not ConnectionState.DISCONNECTED:
                self._queue_connection_state(ConnectionState.DISCONNECTED, self._transport_reason())
        await self._stop_callback_dispatcher()
        discovery = self._discovery
        self._discovery = None
        if discovery is not None:
            await discovery.close()
            if self._observer is not None:
                self._observer.lifecycle("discovery_browser_closed")
        if self._observer is not None:
            self._observer.lifecycle("client_shutdown_completed")
        self._closed.set()

    def _accept_inbound(self, message: Message, _transport_transition: _TransitionResult) -> None:
        transition = self._ingress.accept_inbound(message)
        core = self._core
        assert core is not None
        if isinstance(message, InputClosedEvent):
            core.discard_queued(lambda batch: self._is_unsent_input_audio(batch, message))
        if transition.dispatch:
            self._queue_callback(message, transition)
        if isinstance(message, ConversationEndedEvent):
            self._conversation_terminal.set()

    @staticmethod
    def _is_unsent_input_audio(batch: _OutboundBatch, closed: InputClosedEvent) -> bool:
        tag = batch.tag
        return isinstance(tag, _IngressBatch) and tag.lane is _IngressLane.DATA and tag.input_id == closed.input_id

    def _record_transport_loss(self, error: BaseException) -> None:
        self._transport_loss = error

    def _start_callback_dispatcher(self) -> None:
        if self._callback_task is None:
            self._callback_task = self._loop.create_task(
                self._run_callbacks(),
                name="linklab-client-callbacks",
            )

    async def _stop_callback_dispatcher(self) -> None:
        task = self._callback_task
        if task is None:
            return
        self._events.close()
        if task is not asyncio.current_task():
            try:
                async with asyncio.timeout(self._config.close_timeout_s):
                    await task
            except TimeoutError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                self._events.discard(lambda _item: True)
        self._observe_callback_queues("callback_dispatcher_closed")
        self._callback_task = None

    async def _run_callbacks(self) -> None:
        while True:
            queued = self._events.get_nowait()
            if queued is None:
                if self._events.closed_and_empty:
                    return
                self._callback_wake.clear()
                queued = self._events.get_nowait()
                if queued is None:
                    await self._callback_wake.wait()
                    continue
            item, path = queued
            self._observe_callback_queues()
            message = item.message
            if not isinstance(message, ConnectionStateEvent):
                if item.transition is not None and not item.transition.dispatch:
                    if self._observer is not None:
                        self._observer.disposition(message, "idempotent_ignore")
                    continue
                if not self._ingress.is_current(message):
                    if self._observer is not None:
                        self._observer.disposition(message, "stale_callback_discard")
                    continue
            try:
                await self._dispatch_callback(message)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if self._observer is not None and not isinstance(message, ConnectionStateEvent):
                    self._observer.callback_failure(message, error)
                self._handle_callback_failure(path, message, error)
            else:
                if self._observer is not None and not isinstance(message, ConnectionStateEvent):
                    self._observer.message(message, direction="inbound", phase="callback_dispatched")

    def _queue_callback(self, message: Message, transition: _TransitionResult) -> None:
        path = self._callback_path(message)
        frames = len(message.audio) // 2 if isinstance(message, OutputAudioEvent) else 0
        if not self._events.put_nowait(_InboundEvent(message, transition), path, frames):
            self._observe_callback_queues(_lifecycle_boundary(message))
            self._handle_callback_saturation(path, message)
        else:
            boundary = _lifecycle_boundary(message)
            self._observe_callback_queues(boundary)
            if boundary is not None:
                self._observe_ingress_queues(boundary)
                if self._core is not None:
                    self._core.observe_queues(boundary)

    def _queue_connection_state(self, state: ConnectionState, reason: str | None = None) -> None:
        event = ConnectionStateEvent(state, reason)
        if self._observer is not None:
            fields: dict[str, object] = {"state": state.value}
            if reason is not None:
                fields["reason_type"] = reason
            self._observer.lifecycle("connection_state_changed", **fields)
        if not self._events.put_nowait(_InboundEvent(event), _CallbackPath.CONTROL):
            core = self._core
            if core is not None:
                core._request_close(1011, drain=False, loss=QueueOverflow("client callback control capacity exceeded"))
        self._observe_callback_queues()

    def _handle_callback_saturation(self, path: _CallbackPath, message: Message) -> None:
        if path is _CallbackPath.OUTPUT:
            if not isinstance(message, OutputAudioEvent):
                self._request_overload_close("client output callback capacity exceeded")
                return
            self._events.discard(
                lambda item: isinstance(item.message, (OutputStartedEvent, OutputAudioEvent, OutputEndedEvent))
                and item.message.response_id == message.response_id
            )
            self._observe_callback_queues("output_overflow_discard")
            if not self._ingress.playback_interrupted(
                message.output_id,
                0,
                PlaybackPosition.ESTIMATED,
                PlaybackInterruptReason.OVERFLOW,
            ):
                self._request_overload_close("playback overflow accounting capacity exhausted")
            return

        if path is _CallbackPath.EVENT and self._ingress.state.conversation_id is not None:
            if not self._ingress.cancel_conversation(ConversationCancelReason.CLIENT_FAILED):
                self._request_overload_close("client callback cancellation capacity exhausted")
            return
        self._request_overload_close("client callback capacity exceeded")

    def _handle_callback_failure(
        self,
        path: _CallbackPath,
        message: Message | ConnectionStateEvent,
        error: Exception,
    ) -> None:
        if not isinstance(message, ConnectionStateEvent) and not self._ingress.is_current(message):
            return
        if path is _CallbackPath.OUTPUT:
            assert isinstance(message, (OutputStartedEvent, OutputAudioEvent, OutputEndedEvent))
            self._events.discard(
                lambda item: isinstance(item.message, (OutputStartedEvent, OutputAudioEvent, OutputEndedEvent))
                and item.message.response_id == message.response_id
            )
            self._observe_callback_queues("callback_failure_discard")
            if self._ingress.playback_interrupted(
                message.output_id,
                0,
                PlaybackPosition.ESTIMATED,
                PlaybackInterruptReason.PLAYBACK_FAILED,
            ):
                return
        elif self._ingress.state.conversation_id is not None:
            if self._ingress.cancel_conversation(ConversationCancelReason.CLIENT_FAILED):
                return
        self._request_implementation_close(error)

    def _request_overload_close(self, message: str) -> None:
        core = self._core
        if core is not None:
            core._request_close(1011, drain=False, loss=QueueOverflow(message))

    def _request_implementation_close(self, error: BaseException) -> None:
        if self._local_failure is None:
            self._local_failure = error
        self._reconnect_stop.set()
        core = self._core
        if core is not None:
            core._request_close(1011, drain=False, loss=self._local_failure)

    @staticmethod
    def _callback_path(message: Message) -> _CallbackPath:
        if isinstance(message, (OutputStartedEvent, OutputAudioEvent, OutputEndedEvent)):
            return _CallbackPath.OUTPUT
        if isinstance(message, (ResponseCancelledEvent, ConversationEndedEvent, ErrorEvent)):
            return _CallbackPath.CONTROL
        return _CallbackPath.EVENT

    async def _dispatch_callback(self, event: Message | ConnectionStateEvent) -> None:
        if isinstance(event, ConnectionStateEvent):
            await self._callbacks.on_connection_state(event)
        elif isinstance(event, StateEvent):
            await self._callbacks.on_conversation_state(event)
        elif isinstance(event, TranscriptUpdateEvent):
            await self._callbacks.on_transcript_update(event)
        elif isinstance(event, TranscriptFinalEvent):
            await self._callbacks.on_transcript_final(event)
        elif isinstance(event, ResponseStartedEvent):
            await self._callbacks.on_response_started(event)
        elif isinstance(event, ResponseTextDeltaEvent):
            await self._callbacks.on_response_text_delta(event)
        elif isinstance(event, ResponseTextFinalEvent):
            await self._callbacks.on_response_text_final(event)
        elif isinstance(event, ResponseEndedEvent):
            await self._callbacks.on_response_ended(event)
        elif isinstance(event, ResponseCancelledEvent):
            await self._callbacks.on_response_cancelled(event)
        elif isinstance(event, OutputStartedEvent):
            await self._callbacks.on_output_started(event)
        elif isinstance(event, OutputAudioEvent):
            await self._callbacks.on_output_audio(event)
        elif isinstance(event, OutputEndedEvent):
            await self._callbacks.on_output_ended(event)
        elif isinstance(event, ConversationEndedEvent):
            await self._callbacks.on_conversation_ended(event)
        elif isinstance(event, ErrorEvent):
            await self._callbacks.on_error(event)

    def _transport_reason(self) -> str | None:
        return None if self._transport_loss is None else type(self._transport_loss).__name__

    def _notify_handoff(self) -> None:
        self._loop.call_soon_threadsafe(self._wake.set)

    def _notify_callbacks(self) -> None:
        self._loop.call_soon_threadsafe(self._callback_wake.set)

    def _schedule_ingress_observation(self) -> None:
        self._loop.call_soon_threadsafe(self._observe_ingress_queues)

    def _observe_ingress_queues(self, boundary: str | None = None) -> None:
        if self._observer is not None:
            self._observer.queue_snapshots(self._ingress.snapshots(), boundary=boundary)

    def _observe_callback_queues(self, boundary: str | None = None) -> None:
        if self._observer is not None:
            self._observer.queue_snapshots(self._events.snapshots(), boundary=boundary)

    def _observe_all_queues(self, boundary: str) -> None:
        observer = self._observer
        if observer is None:
            return
        observer.queue_snapshots(self._ingress.snapshots(), boundary=boundary)
        observer.queue_snapshots(self._events.snapshots(), boundary=boundary)
        if self._core is not None:
            self._core.observe_queues(boundary)

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("VoiceClient operation requires its event loop") from error
        if loop is not self._loop:
            raise RuntimeError("VoiceClient is bound to a different event loop")


def _reconnect_failure_eligible(error: BaseException) -> bool:
    if isinstance(error, WebSocketConnectionClosed):
        received = error.rcvd
        sent = error.sent
        code = received.code if received is not None else None if sent is None else sent.code
        return code in (None, 1001, 1006)
    return isinstance(error, (OSError, TimeoutError))
