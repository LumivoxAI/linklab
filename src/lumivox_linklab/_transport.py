from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Protocol
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable

from ._codec import decode_message, _encode_message_with_limits
from ._enums import EndpointRole, ConnectionState, MessageDirection
from ._config import ConnectionLimits
from ._errors import CodecError, QueueOverflow, ConnectionClosed
from ._messages import Message
from ._protocol import ProtocolValidator, _TransitionResult


class _FrameTransport(Protocol):
    async def recv(self) -> bytes | str: ...

    async def send(self, data: bytes) -> None: ...

    async def close(self, code: int) -> None: ...


class _QueueLane(StrEnum):
    DATA = "data"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class _QueueSnapshot:
    identity: str
    capacity: int
    occupancy: int
    overflow_count: int
    oldest_residence_ms: float
    occupancy_unit: str


@dataclass(frozen=True, slots=True)
class _OutboundBatch:
    frames: tuple[bytes, ...]
    lane: _QueueLane
    weight: int
    enqueued_at: float


class _BoundedBatchQueue:
    def __init__(
        self,
        *,
        identity: str,
        data_capacity: int,
        control_capacity: int,
        occupancy_unit: str,
        clock: Callable[[], float],
    ) -> None:
        if not identity:
            raise ValueError("identity must not be empty")
        if type(data_capacity) is not int or data_capacity <= 0:
            raise ValueError("data_capacity must be a positive integer")
        if type(control_capacity) is not int or control_capacity <= 0:
            raise ValueError("control_capacity must be a positive integer")
        if not occupancy_unit:
            raise ValueError("occupancy_unit must not be empty")
        self._identity = identity
        self._capacities = {
            _QueueLane.DATA: data_capacity,
            _QueueLane.CONTROL: control_capacity,
        }
        self._occupancy = {_QueueLane.DATA: 0, _QueueLane.CONTROL: 0}
        self._overflows = {_QueueLane.DATA: 0, _QueueLane.CONTROL: 0}
        self._occupancy_unit = occupancy_unit
        self._clock = clock
        self._items: deque[_OutboundBatch] = deque()
        self._available = asyncio.Event()
        self._closed = False

    def put_nowait(self, frames: tuple[bytes, ...], *, lane: _QueueLane, weight: int) -> None:
        if self._closed:
            raise ConnectionClosed("transport queue is closed")
        if not isinstance(lane, _QueueLane):
            raise TypeError("lane must be _QueueLane")
        if not frames or not all(type(frame) is bytes for frame in frames):
            raise ValueError("frames must be a nonempty tuple of bytes")
        if type(weight) is not int or weight <= 0:
            raise ValueError("weight must be a positive integer")
        if self._occupancy[lane] + weight > self._capacities[lane]:
            self._overflows[lane] += 1
            raise QueueOverflow(f"{self._identity} {lane.value} capacity exceeded")
        self._items.append(_OutboundBatch(frames, lane, weight, self._clock()))
        self._occupancy[lane] += weight
        self._available.set()

    async def get(self) -> _OutboundBatch:
        while not self._items:
            if self._closed:
                raise ConnectionClosed("transport queue is closed")
            self._available.clear()
            await self._available.wait()
        item = self._items.popleft()
        self._occupancy[item.lane] -= item.weight
        if not self._items:
            self._available.clear()
        return item

    def discard(self, predicate: Callable[[_OutboundBatch], bool]) -> tuple[_OutboundBatch, ...]:
        discarded: list[_OutboundBatch] = []
        retained: deque[_OutboundBatch] = deque()
        while self._items:
            item = self._items.popleft()
            if predicate(item):
                discarded.append(item)
                self._occupancy[item.lane] -= item.weight
            else:
                retained.append(item)
        self._items = retained
        if not self._items:
            self._available.clear()
        return tuple(discarded)

    def close(self, *, discard: bool) -> None:
        if self._closed:
            return
        self._closed = True
        if discard:
            self.discard(lambda _item: True)
        self._available.set()

    def snapshot(self, lane: _QueueLane) -> _QueueSnapshot:
        oldest = next((item for item in self._items if item.lane is lane), None)
        residence = 0.0 if oldest is None else max(0.0, (self._clock() - oldest.enqueued_at) * 1_000)
        return _QueueSnapshot(
            identity=f"{self._identity}.{lane.value}",
            capacity=self._capacities[lane],
            occupancy=self._occupancy[lane],
            overflow_count=self._overflows[lane],
            oldest_residence_ms=residence,
            occupancy_unit=self._occupancy_unit,
        )


type _TransitionHook = Callable[[Message, _TransitionResult], None]
type _LossHook = Callable[[BaseException], None]


class _TransportCore:
    def __init__(
        self,
        transport: _FrameTransport,
        *,
        role: EndpointRole,
        limits: ConnectionLimits,
        data_capacity: int,
        control_capacity: int,
        close_timeout_s: float,
        on_transition: _TransitionHook,
        on_transport_loss: _LossHook,
        occupancy_unit: str = "units",
        clock: Callable[[], float] | None = None,
    ) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("transport core must be created in a running event loop") from error
        if not isinstance(role, EndpointRole):
            raise TypeError("role must be EndpointRole")
        if not isinstance(limits, ConnectionLimits):
            raise TypeError("limits must be ConnectionLimits")
        if not isinstance(close_timeout_s, int | float) or close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")
        self._transport = transport
        self._role = role
        self._limits = limits
        self._close_timeout_s = float(close_timeout_s)
        self._on_transition = on_transition
        self._on_transport_loss = on_transport_loss
        self._validator = ProtocolValidator(role)
        self._queue = _BoundedBatchQueue(
            identity="outbound",
            data_capacity=data_capacity,
            control_capacity=control_capacity,
            occupancy_unit=occupancy_unit,
            clock=clock or self._loop.time,
        )
        self._reader_task: asyncio.Task[None] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = asyncio.Event()
        self._loss: BaseException | None = None

    @property
    def validator(self) -> ProtocolValidator:
        return self._validator

    def start(self) -> None:
        self._check_loop()
        if self._reader_task is not None:
            raise RuntimeError("transport core is already started")
        self._validator.transport_connected()
        self._reader_task = self._loop.create_task(self._reader(), name="linklab-transport-reader")
        self._writer_task = self._loop.create_task(self._writer(), name="linklab-transport-writer")

    def set_limits(self, limits: ConnectionLimits) -> None:
        self._check_loop()
        if not isinstance(limits, ConnectionLimits):
            raise TypeError("limits must be ConnectionLimits")
        if self._closing:
            raise ConnectionClosed("transport core is closing")
        self._limits = limits

    def enqueue_batch(
        self,
        messages: tuple[Message, ...],
        *,
        lane: _QueueLane,
        weight: int | None = None,
    ) -> None:
        self._check_loop()
        if self._reader_task is None:
            raise RuntimeError("transport core is not started")
        if self._closing:
            raise ConnectionClosed("transport core is closing")
        if not messages:
            raise ValueError("messages must not be empty")

        data = self._validator._data
        wire_messages: list[Message] = []
        for message in messages:
            data, result = self._validator._transition(data, message)
            wire_messages.append(message)
            wire_messages.extend(result.outbound)
        frames = tuple(_encode_message_with_limits(message, self._limits) for message in wire_messages)
        self._queue.put_nowait(frames, lane=lane, weight=len(frames) if weight is None else weight)
        self._validator._data = data

    def snapshots(self) -> tuple[_QueueSnapshot, _QueueSnapshot]:
        self._check_loop()
        return (self._queue.snapshot(_QueueLane.DATA), self._queue.snapshot(_QueueLane.CONTROL))

    def discard_queued(self, predicate: Callable[[_OutboundBatch], bool]) -> tuple[_OutboundBatch, ...]:
        self._check_loop()
        return self._queue.discard(predicate)

    async def close(self, code: int = 1000, *, drain: bool = False) -> None:
        self._check_loop()
        task = self._request_close(code, drain=drain)
        await asyncio.shield(task)

    async def wait_closed(self) -> None:
        self._check_loop()
        await self._closed.wait()

    def _check_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("transport core operation requires its event loop") from error
        if loop is not self._loop:
            raise RuntimeError("transport core is bound to a different event loop")

    async def _reader(self) -> None:
        try:
            while True:
                frame = await self._transport.recv()
                if type(frame) is not bytes:
                    raise CodecError("WebSocket application messages must be binary")
                message = decode_message(frame, direction=self._inbound_direction(), limits=self._limits)
                data, result = self._validator._transition(self._validator._data, message)
                if result.outbound:
                    encoded = tuple(_encode_message_with_limits(item, self._limits) for item in result.outbound)
                    self._queue.put_nowait(encoded, lane=_QueueLane.CONTROL, weight=len(encoded))
                self._validator._data = data
                self._on_transition(message, result)
                if result.close_code is not None:
                    self._request_close(result.close_code, drain=True)
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._request_close(1002, drain=False, loss=error)

    async def _writer(self) -> None:
        try:
            while True:
                batch = await self._queue.get()
                for frame in batch.frames:
                    await self._transport.send(frame)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            return
        except BaseException as error:
            self._request_close(1011, drain=False, loss=error)

    def _request_close(
        self,
        code: int,
        *,
        drain: bool,
        loss: BaseException | None = None,
    ) -> asyncio.Task[None]:
        if self._close_task is not None:
            return self._close_task
        self._closing = True
        self._loss = loss
        if self._validator.state.connection_state in (ConnectionState.HANDSHAKING, ConnectionState.READY):
            self._validator.begin_close()
        self._close_task = self._loop.create_task(
            self._shutdown(code, drain=drain),
            name="linklab-transport-close",
        )
        return self._close_task

    async def _shutdown(self, code: int, *, drain: bool) -> None:
        current = asyncio.current_task()
        reader = self._reader_task
        writer = self._writer_task
        try:
            if reader is not None and reader is not current and not reader.done():
                reader.cancel()
            self._queue.close(discard=not drain)
            if writer is not None and writer is not current and not writer.done():
                if drain:
                    try:
                        async with asyncio.timeout(self._close_timeout_s):
                            await writer
                    except TimeoutError:
                        writer.cancel()
                else:
                    writer.cancel()
            await self._join_task(reader, current)
            await self._join_task(writer, current)
            try:
                async with asyncio.timeout(self._close_timeout_s):
                    await self._transport.close(code)
            except TimeoutError:
                pass
            except Exception as error:
                if self._loss is None:
                    self._loss = error
        finally:
            if self._validator.state.connection_state is not ConnectionState.DISCONNECTED:
                self._validator.transport_disconnected()
            if self._loss is not None:
                try:
                    self._on_transport_loss(self._loss)
                except Exception:
                    pass
            self._closed.set()

    async def _join_task(self, task: asyncio.Task[None] | None, current: asyncio.Task[Any] | None) -> None:
        if task is None or task is current:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, ConnectionClosed):
            pass

    def _inbound_direction(self) -> MessageDirection:
        if self._role is EndpointRole.CLIENT:
            return MessageDirection.SERVER_TO_CLIENT
        return MessageDirection.CLIENT_TO_SERVER
