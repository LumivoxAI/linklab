import asyncio
from typing import Any
from collections.abc import Callable, Coroutine

import pytest
import msgpack  # type: ignore[import-untyped]

import lumivox_linklab as linklab
from tests.helpers import RecordingLogger
from lumivox_linklab._transport import (
    _QueueLane,
    _OutboundBatch,
    _TransportCore,
    _BoundedBatchQueue,
)
from lumivox_linklab._observability import _Observer

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
        block_pong: bool = False,
        close_error: Exception | None = None,
    ) -> None:
        self.inbound: asyncio.Queue[bytes | str | BaseException] = asyncio.Queue()
        self.sent: list[bytes] = []
        self.recv_tasks: set[asyncio.Task[object] | None] = set()
        self.send_tasks: set[asyncio.Task[object] | None] = set()
        self.close_codes: list[int] = []
        self.close_error = close_error
        self.send_started = asyncio.Event()
        self.ping_started = asyncio.Event()
        self.send_gate = asyncio.Event()
        self.close_gate = asyncio.Event()
        self.pong_gate = asyncio.Event()
        if not block_send:
            self.send_gate.set()
        if not block_close:
            self.close_gate.set()
        if not block_pong:
            self.pong_gate.set()

    async def recv(self) -> bytes | str:
        self.recv_tasks.add(asyncio.current_task())
        value = await self.inbound.get()
        if isinstance(value, BaseException):
            raise value
        return value

    async def send(self, data: bytes) -> None:
        self.send_tasks.add(asyncio.current_task())
        self.send_started.set()
        await self.send_gate.wait()
        self.sent.append(data)

    async def ping(self) -> asyncio.Future[float]:
        self.ping_started.set()
        future: asyncio.Future[float] = asyncio.get_running_loop().create_future()
        if self.pong_gate.is_set():
            future.set_result(0.001)
        return future

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
    role: linklab.EndpointRole = linklab.EndpointRole.CLIENT,
    limits: linklab.ConnectionLimits = LIMITS,
    data_capacity: int = 8,
    close_timeout_s: float = 0.1,
    ping_interval_s: float = 20.0,
    ping_timeout_s: float = 20.0,
    on_rtt: Callable[[float], None] | None = None,
    observer: _Observer | None = None,
) -> _TransportCore:
    return _TransportCore(
        transport,
        role=role,
        limits=limits,
        data_capacity=data_capacity,
        control_capacity=8,
        close_timeout_s=close_timeout_s,
        on_transition=lambda message, _result: transitions.append(message),
        on_transport_loss=losses.append,
        ping_interval_s=ping_interval_s,
        ping_timeout_s=ping_timeout_s,
        on_rtt=on_rtt,
        occupancy_unit="messages",
        observer=observer,
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


def test_validator_generated_messages_share_queued_and_sent_observation_path() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        logger = RecordingLogger()
        observer = _Observer(logger, endpoint_role="server", connection_id="server-1", clock=lambda: 10.0)
        core = make_core(
            transport,
            [],
            [],
            role=linklab.EndpointRole.SERVER,
            observer=observer,
        )
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        conversation_id = linklab.ConversationId(1)
        input_id = linklab.InputId(1)
        response_id = linklab.ResponseId(1)

        inbound = (
            linklab.ConversationStartedEvent(conversation_id, "wake_word"),
            linklab.InputStartedEvent(
                conversation_id,
                input_id,
                linklab.InputStartReason.ACTIVATION,
                0,
            ),
        )
        for message in inbound:
            await transport.inbound.put(linklab.encode_message(message))
            await asyncio.sleep(0)
        core.enqueue_batch(
            (linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING),),
            lane=_QueueLane.CONTROL,
        )
        abort = linklab.InputAbortedEvent(conversation_id, input_id, linklab.InputAbortReason.CAPTURE_FAILED)
        await transport.inbound.put(linklab.encode_message(abort))
        await asyncio.sleep(0)
        core.enqueue_batch(
            (
                linklab.StateEvent(conversation_id, 2, linklab.CoarseState.PROCESSING),
                linklab.TranscriptFinalEvent(conversation_id, input_id, ""),
                linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False),
                linklab.StateEvent(conversation_id, 3, linklab.CoarseState.RESPONDING),
            ),
            lane=_QueueLane.CONTROL,
        )
        barge_in = linklab.InputStartedEvent(
            conversation_id,
            linklab.InputId(2),
            linklab.InputStartReason.BARGE_IN,
            1,
            response_id,
        )
        await transport.inbound.put(linklab.encode_message(barge_in))

        async with asyncio.timeout(1):
            while not any(
                event == "protocol_message"
                and fields.get("message_type") == "ResponseCancelledEvent"
                and fields.get("phase") == "sent"
                for _, event, fields in logger.records
            ):
                await asyncio.sleep(0)

        phases = {
            fields["phase"]
            for _, event, fields in logger.records
            if event == "protocol_message"
            and fields.get("message_type") == "ResponseCancelledEvent"
            and fields.get("reason") == "barge_in"
        }
        assert phases == {"queued", "sent"}
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


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"\xc1", linklab.ErrorCode.MALFORMED_MESSAGE),
        (
            msgpack.packb({"type": "future.message"}, use_bin_type=True),
            linklab.ErrorCode.UNKNOWN_MESSAGE,
        ),
        (
            linklab.encode_message(linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING)),
            linklab.ErrorCode.PROTOCOL_STATE,
        ),
        (b"x" * 16_385, linklab.ErrorCode.MESSAGE_TOO_LARGE),
    ],
)
def test_server_fatal_categories_send_safe_error_then_close(payload: bytes, expected: linklab.ErrorCode) -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        losses: list[BaseException] = []
        limits = linklab.ConnectionLimits(max_message_bytes=16_384, max_output_audio_frames=1_600)
        core = make_core(
            transport,
            [],
            losses,
            role=linklab.EndpointRole.SERVER,
            limits=limits,
        )
        core.start()
        await transport.inbound.put(payload)
        await core.wait_closed()

        assert len(transport.sent) == 1
        error = linklab.decode_message(
            transport.sent[0],
            direction=linklab.MessageDirection.SERVER_TO_CLIENT,
            limits=limits,
        )
        assert error == linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, expected, True)
        assert transport.close_codes == [1002]
        assert core.outcome is not None and not core.outcome.reconnect_eligible
        assert len(losses) == 1

    run(scenario())


def test_fatal_seals_queue_and_error_is_last_application_message() -> None:
    async def scenario() -> None:
        transport = FakeTransport(block_send=True)
        core = make_core(transport, [], [], role=linklab.EndpointRole.SERVER)
        core.start()
        core.enqueue_batch((CLIENT_HELLO, SERVER_HELLO), lane=_QueueLane.DATA)
        await transport.send_started.wait()

        await transport.inbound.put(msgpack.packb({"type": "future.message"}, use_bin_type=True))
        async with asyncio.timeout(1):
            while core.outcome is None:
                await asyncio.sleep(0)
        with pytest.raises(linklab.ConnectionClosed):
            core.enqueue_batch((CLIENT_HELLO,), lane=_QueueLane.DATA)
        transport.send_gate.set()
        await core.wait_closed()

        assert len(transport.sent) == 2
        assert transport.sent[0] == linklab.encode_message(CLIENT_HELLO)
        error = linklab.decode_message(
            transport.sent[1],
            direction=linklab.MessageDirection.SERVER_TO_CLIENT,
            limits=LIMITS,
        )
        assert isinstance(error, linklab.ErrorEvent)
        assert error.code is linklab.ErrorCode.UNKNOWN_MESSAGE

    run(scenario())


def test_peer_connection_error_discards_later_queued_application_messages() -> None:
    async def scenario() -> None:
        transport = FakeTransport(block_send=True)
        core = make_core(transport, [], [])
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        conversation = linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word")
        input_started = linklab.InputStartedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputStartReason.ACTIVATION,
            0,
        )
        core.enqueue_batch((conversation, input_started), lane=_QueueLane.DATA)
        await transport.send_started.wait()

        peer_error = linklab.ErrorEvent(
            linklab.ErrorScope.CONNECTION,
            linklab.ErrorCode.PROTOCOL_STATE,
            True,
        )
        await transport.inbound.put(linklab.encode_message(peer_error))
        transport.send_gate.set()
        await core.wait_closed()

        assert transport.sent == [linklab.encode_message(conversation)]
        assert transport.close_codes == [1002]

    run(scenario())


def test_fatal_blocked_writer_still_attempts_close_within_timeout() -> None:
    async def scenario() -> None:
        transport = FakeTransport(block_send=True)
        core = make_core(
            transport,
            [],
            [],
            role=linklab.EndpointRole.SERVER,
            close_timeout_s=0.02,
        )
        core.start()
        core.enqueue_batch((CLIENT_HELLO,), lane=_QueueLane.DATA)
        await transport.send_started.wait()

        await transport.inbound.put("text")
        async with asyncio.timeout(0.1):
            await core.wait_closed()

        assert transport.close_codes == [1002]

    run(scenario())


def test_completed_handshake_seeds_ready_state_and_negotiated_limits() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        core = make_core(transport, [], [])
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)

        assert core.validator.state.connection_state is linklab.ConnectionState.READY
        core.enqueue_batch(
            (linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word"),),
            lane=_QueueLane.DATA,
        )
        async with asyncio.timeout(1):
            while not transport.sent:
                await asyncio.sleep(0)
        assert len(transport.sent) == 1
        await core.close()

    run(scenario())


def test_keepalive_observes_rtt_and_peer_timeout_is_fatal() -> None:
    async def scenario() -> None:
        observed: list[float] = []
        responsive = FakeTransport()
        core = make_core(
            responsive,
            [],
            [],
            ping_interval_s=0.001,
            ping_timeout_s=0.01,
            on_rtt=observed.append,
        )
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        async with asyncio.timeout(1):
            while not observed:
                await asyncio.sleep(0)
        assert observed[0] == 0.001
        await core.close()

        silent = FakeTransport(block_pong=True)
        server_core = make_core(
            silent,
            [],
            [],
            role=linklab.EndpointRole.SERVER,
            ping_interval_s=0.001,
            ping_timeout_s=0.001,
        )
        server_core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        await server_core.wait_closed()
        error = linklab.decode_message(
            silent.sent[0],
            direction=linklab.MessageDirection.SERVER_TO_CLIENT,
            limits=LIMITS,
        )
        assert isinstance(error, linklab.ErrorEvent)
        assert error.code is linklab.ErrorCode.PEER_UNRESPONSIVE
        assert silent.close_codes == [1002]

    run(scenario())


def test_transport_loss_classification_and_protocol_close_reconnect_policy() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        core = make_core(transport, [], [])
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        await transport.inbound.put(OSError("connection lost"))
        await core.wait_closed()
        assert transport.close_codes == []
        assert core.outcome is not None
        assert core.outcome.abnormal
        assert core.outcome.reconnect_eligible

        for code in (1002, 1008):
            closed_transport = FakeTransport()
            closed_core = make_core(closed_transport, [], [])
            closed_core.start()
            await closed_core.close(code)
            assert closed_core.outcome is not None
            assert not closed_core.outcome.reconnect_eligible

    run(scenario())


def test_local_id_exhaustion_has_no_partial_start_and_closes_fatally() -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        core = make_core(transport, [], [], role=linklab.EndpointRole.SERVER)
        core.start_handshaken(CLIENT_HELLO, SERVER_HELLO)
        before = core.validator.state

        core.fail_connection(linklab.ErrorCode.ID_EXHAUSTED)
        await core.wait_closed()

        assert before.conversation_id is None
        assert core.validator.state.connection_state is linklab.ConnectionState.DISCONNECTED
        assert len(transport.sent) == 1
        error = linklab.decode_message(
            transport.sent[0],
            direction=linklab.MessageDirection.SERVER_TO_CLIENT,
            limits=LIMITS,
        )
        assert isinstance(error, linklab.ErrorEvent)
        assert error.code is linklab.ErrorCode.ID_EXHAUSTED
        assert transport.close_codes == [1002]

    run(scenario())


def test_fatal_error_encoding_failure_falls_back_to_close(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        transport = FakeTransport()
        core = make_core(transport, [], [], role=linklab.EndpointRole.SERVER)
        core.start()

        def fail_encoding(_message: linklab.Message, _limits: linklab.ConnectionLimits) -> bytes:
            raise linklab.CodecError("injected encoding failure")

        monkeypatch.setattr("lumivox_linklab._transport._encode_message_with_limits", fail_encoding)
        await transport.inbound.put("text")
        await core.wait_closed()

        assert transport.sent == []
        assert transport.close_codes == [1002]

    run(scenario())
