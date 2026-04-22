import asyncio
from typing import Any
from collections.abc import Callable, Coroutine

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._transport import (
    _QueueLane,
    _OutboundBatch,
    _TransportCore,
    _BoundedBatchQueue,
)

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
LIMITS = linklab.ConnectionLimits(max_output_audio_frames=1_600)
CLIENT_HELLO = linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,))
SERVER_HELLO = linklab.ServerHello(1, CAPABILITIES, PCM_16K, LIMITS)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coroutine, debug=True)


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


class FakeTransport:
    def __init__(
        self,
        *,
        block_send: bool = False,
        block_close: bool = False,
        close_error: Exception | None = None,
    ) -> None:
        self.inbound: asyncio.Queue[bytes | str] = asyncio.Queue()
        self.sent: list[bytes] = []
        self.recv_tasks: set[asyncio.Task[object] | None] = set()
        self.send_tasks: set[asyncio.Task[object] | None] = set()
        self.close_codes: list[int] = []
        self.close_error = close_error
        self.send_started = asyncio.Event()
        self.send_gate = asyncio.Event()
        self.close_gate = asyncio.Event()
        if not block_send:
            self.send_gate.set()
        if not block_close:
            self.close_gate.set()

    async def recv(self) -> bytes | str:
        self.recv_tasks.add(asyncio.current_task())
        return await self.inbound.get()

    async def send(self, data: bytes) -> None:
        self.send_tasks.add(asyncio.current_task())
        self.send_started.set()
        await self.send_gate.wait()
        self.sent.append(data)

    async def close(self, code: int) -> None:
        self.close_codes.append(code)
        await self.close_gate.wait()
        if self.close_error is not None:
            raise self.close_error


def make_queue(clock: Callable[[], float]) -> _BoundedBatchQueue:
    return _BoundedBatchQueue(
        identity="test",
        data_capacity=3,
        control_capacity=1,
        occupancy_unit="frames",
        clock=clock,
    )


def test_bounded_queue_has_independent_capacity_and_deterministic_snapshots() -> None:
    async def scenario() -> None:
        clock = Clock()
        queue = make_queue(clock)
        queue.put_nowait((b"one",), lane=_QueueLane.DATA, weight=3)
        queue.put_nowait((b"stop",), lane=_QueueLane.CONTROL, weight=1)

        with pytest.raises(linklab.QueueOverflow):
            queue.put_nowait((b"overflow",), lane=_QueueLane.DATA, weight=1)

        clock.now += 0.25
        data, control = queue.snapshot(_QueueLane.DATA), queue.snapshot(_QueueLane.CONTROL)
        assert data.capacity == 3
        assert data.occupancy == 3
        assert data.overflow_count == 1
        assert data.oldest_residence_ms == 250.0
        assert data.occupancy_unit == "frames"
        assert control.occupancy == 1
        assert control.overflow_count == 0

        first = await queue.get()
        second = await queue.get()
        assert (first.frames, second.frames) == ((b"one",), (b"stop",))
        assert queue.snapshot(_QueueLane.DATA).oldest_residence_ms == 0.0

    run(scenario())


def test_bounded_queue_rejects_whole_batch_and_wakes_waiter_when_closed() -> None:
    async def scenario() -> None:
        queue = make_queue(asyncio.get_running_loop().time)
        queue.put_nowait((b"a", b"b"), lane=_QueueLane.DATA, weight=2)
        before = queue.snapshot(_QueueLane.DATA)

        with pytest.raises(linklab.QueueOverflow):
            queue.put_nowait((b"c", b"d"), lane=_QueueLane.DATA, weight=2)
        assert queue.snapshot(_QueueLane.DATA).occupancy == before.occupancy

        discarded = queue.discard(lambda item: b"a" in item.frames)
        assert len(discarded) == 1
        assert queue.snapshot(_QueueLane.DATA).occupancy == 0
        waiter = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        queue.close(discard=False)
        with pytest.raises(linklab.ConnectionClosed):
            await waiter
        with pytest.raises(linklab.ConnectionClosed):
            queue.put_nowait((b"late",), lane=_QueueLane.CONTROL, weight=1)

    run(scenario())


def test_queue_discard_preserves_other_batches_and_fifo() -> None:
    async def scenario() -> None:
        queue = make_queue(asyncio.get_running_loop().time)
        queue.put_nowait((b"data",), lane=_QueueLane.DATA, weight=1)
        queue.put_nowait((b"control",), lane=_QueueLane.CONTROL, weight=1)
        queue.put_nowait((b"other",), lane=_QueueLane.DATA, weight=1)

        discarded = queue.discard(lambda item: item.frames == (b"data",))

        assert tuple(item.frames for item in discarded) == ((b"data",),)
        assert (await queue.get()).frames == (b"control",)
        assert (await queue.get()).frames == (b"other",)

    run(scenario())


def make_core(
    transport: FakeTransport,
    transitions: list[linklab.Message],
    losses: list[BaseException],
    *,
    data_capacity: int = 8,
    close_timeout_s: float = 0.1,
) -> _TransportCore:
    return _TransportCore(
        transport,
        role=linklab.EndpointRole.CLIENT,
        limits=LIMITS,
        data_capacity=data_capacity,
        control_capacity=8,
        close_timeout_s=close_timeout_s,
        on_transition=lambda message, _result: transitions.append(message),
        on_transport_loss=losses.append,
        occupancy_unit="messages",
    )


def test_core_has_one_reader_writer_atomic_batches_and_reader_independence() -> None:
    async def scenario() -> None:
        transport = FakeTransport(block_send=True)
        transitions: list[linklab.Message] = []
        losses: list[BaseException] = []
        core = make_core(transport, transitions, losses)
        core.start()
        core.enqueue_batch((CLIENT_HELLO,), lane=_QueueLane.DATA)
        await transport.send_started.wait()

        await transport.inbound.put(linklab.encode_message(SERVER_HELLO))
        async with asyncio.timeout(1):
            while not transitions:
                await asyncio.sleep(0)
        assert transitions == [SERVER_HELLO]

        first = linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word")
        second = linklab.InputStartedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputStartReason.ACTIVATION,
            0,
        )
        core.enqueue_batch((first, second), lane=_QueueLane.DATA)
        transport.send_gate.set()
        async with asyncio.timeout(1):
            while len(transport.sent) != 3:
                await asyncio.sleep(0)

        assert transport.sent == [
            linklab.encode_message(CLIENT_HELLO),
            linklab.encode_message(first),
            linklab.encode_message(second),
        ]
        assert len(transport.recv_tasks) == 1
        assert len(transport.send_tasks) == 1
        await core.close()
        assert losses == []

    run(scenario())


def test_enqueue_overflow_does_not_mutate_validator() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        core = make_core(transport, [], [], data_capacity=1)
        core.start()
        before = core.validator._data

        with pytest.raises(linklab.QueueOverflow):
            core.enqueue_batch((CLIENT_HELLO, SERVER_HELLO), lane=_QueueLane.DATA, weight=2)

        assert core.validator._data == before
        assert core.snapshots()[0].occupancy == 0
        await core.close()

    run(scenario())


def test_text_frame_reports_loss_once_and_closes() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        losses: list[BaseException] = []
        core = make_core(transport, [], losses)
        core.start()
        await transport.inbound.put("not binary")
        await core.wait_closed()

        assert len(losses) == 1
        assert isinstance(losses[0], linklab.CodecError)
        assert transport.close_codes == [1002]
        assert core.validator.state.connection_state is linklab.ConnectionState.DISCONNECTED

    run(scenario())


def test_concurrent_close_is_bounded_idempotent_and_leaves_no_tasks() -> None:
    async def scenario() -> None:
        transport = FakeTransport(block_send=True, block_close=True)
        core = make_core(transport, [], [], close_timeout_s=0.01)
        core.start()
        core.enqueue_batch((CLIENT_HELLO,), lane=_QueueLane.DATA)
        await transport.send_started.wait()

        await asyncio.gather(core.close(), core.close(), core.close())

        assert transport.close_codes == [1000]
        assert core.validator.state.connection_state is linklab.ConnectionState.DISCONNECTED
        live_names = {
            task.get_name() for task in asyncio.all_tasks() if not task.done() and task is not asyncio.current_task()
        }
        assert not {name for name in live_names if name.startswith("linklab-transport-")}

    run(scenario())


def test_close_transport_failure_is_reported_without_breaking_close() -> None:
    async def scenario() -> None:
        error = OSError("close failed")
        transport = FakeTransport(close_error=error)
        losses: list[BaseException] = []
        core = make_core(transport, [], losses)
        core.start()

        await core.close()

        assert losses == [error]
        assert core.validator.state.connection_state is linklab.ConnectionState.DISCONNECTED

    run(scenario())


def test_core_enforces_loop_affinity() -> None:
    holder: list[_TransportCore] = []

    async def create() -> None:
        holder.append(make_core(FakeTransport(), [], []))

    run(create())

    async def use_from_another_loop() -> None:
        with pytest.raises(RuntimeError, match="different event loop"):
            holder[0].snapshots()

    run(use_from_another_loop())


def test_queue_snapshot_is_immutable() -> None:
    async def scenario() -> None:
        snapshot = make_queue(asyncio.get_running_loop().time).snapshot(_QueueLane.DATA)
        with pytest.raises((AttributeError, TypeError)):
            setattr(snapshot, "occupancy", 1)

    run(scenario())


def test_discard_predicate_receives_complete_atomic_batch() -> None:
    async def scenario() -> None:
        queue = make_queue(asyncio.get_running_loop().time)
        observed: list[_OutboundBatch] = []
        queue.put_nowait((b"one", b"two"), lane=_QueueLane.DATA, weight=2)

        def discard(batch: _OutboundBatch) -> bool:
            observed.append(batch)
            return True

        queue.discard(discard)
        assert observed[0].frames == (b"one", b"two")

    run(scenario())
