from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._client_ingress import _IngressLane, _IngressBatch, _ClientAudioIngress

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def pcm(frames: int, value: int = 1) -> bytes:
    return bytes([value]) * frames * 2


def annotated(
    frames: int,
    *,
    generation: int = 0,
    discontinuity: bool = False,
    speech: bool = False,
    activated: bool = False,
    wake_word: str | None = None,
    value: int = 1,
) -> linklab.AnnotatedAudio:
    return linklab.AnnotatedAudio(
        pcm(frames, value),
        generation,
        discontinuity,
        speech,
        activated,
        wake_word,
    )


def make_ingress(
    *,
    capacity: int = 16_000,
    pre_roll: int = 8_000,
    max_chunk: int = 1_600,
    max_input_frames: int = 1_920_000,
    notify: Callable[[], None] | None = None,
    control_capacity: int = 16,
) -> _ClientAudioIngress:
    config = linklab.ClientConfig(
        "ws://localhost:8765",
        (PCM_16K,),
        input_queue_frames=capacity,
        waiting_pre_roll_frames=pre_roll,
    )
    limits = linklab.ConnectionLimits(
        max_input_audio_frames=max_chunk,
        max_output_audio_frames=1_600,
        max_input_frames=max_input_frames,
    )
    ingress = _ClientAudioIngress(config, notify=notify, control_capacity=control_capacity)
    ingress.start(
        linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)),
        linklab.ServerHello(1, CAPABILITIES, PCM_16K, limits),
    )
    return ingress


def take_all(ingress: _ClientAudioIngress) -> tuple[_IngressBatch, ...]:
    batches: list[_IngressBatch] = []
    while (batch := ingress.next_batch()) is not None:
        batches.append(batch)
        ingress.acknowledge(batch)
    return tuple(batches)


def audio_messages(batch: _IngressBatch) -> tuple[linklab.InputAudioEvent, ...]:
    return tuple(message for message in batch.messages if isinstance(message, linklab.InputAudioEvent))


def finish_input_and_response(
    ingress: _ClientAudioIngress,
    input_id: linklab.InputId,
    accepted_frames: int,
    *,
    response_id: linklab.ResponseId = linklab.ResponseId(1),
    finish_response: bool = True,
) -> None:
    conversation_id = linklab.ConversationId(1)
    ingress.accept_inbound(
        linklab.InputClosedEvent(
            conversation_id,
            input_id,
            accepted_frames,
            linklab.InputCloseReason.ENDPOINT,
        )
    )
    ingress.accept_inbound(linklab.TranscriptFinalEvent(conversation_id, input_id, "hello"))
    ingress.accept_inbound(linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False))
    if finish_response:
        ingress.accept_inbound(linklab.ResponseEndedEvent(conversation_id, response_id))


def test_disconnected_and_inactive_audio_are_ignored() -> None:
    config = linklab.ClientConfig("ws://localhost:8765", (PCM_16K,))
    disconnected = _ClientAudioIngress(config)
    assert disconnected.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.IGNORED_INACTIVE

    ingress = make_ingress()
    assert ingress.submit(annotated(10)) is linklab.AudioSubmitResult.IGNORED_INACTIVE
    assert ingress.next_batch() is None


def test_activation_is_atomic_copied_split_and_notified_once() -> None:
    notifications: list[None] = []
    ingress = make_ingress(max_chunk=160, notify=lambda: notifications.append(None))
    source = bytearray(pcm(321, 7))
    item = linklab.AnnotatedAudio(source, 3, False, True, True, "lumivox")

    assert ingress.submit(item) is linklab.AudioSubmitResult.ACCEPTED
    source[:] = pcm(321, 9)

    batch = ingress.next_batch()
    assert batch is not None
    assert batch.lane is _IngressLane.DATA
    assert batch.audio_frames == 321
    assert batch.messages[:2] == (
        linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word", "lumivox"),
        linklab.InputStartedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputStartReason.ACTIVATION,
            3,
        ),
    )
    chunks = audio_messages(batch)
    assert tuple((chunk.start_frame, len(chunk.audio) // 2) for chunk in chunks) == (
        (0, 160),
        (160, 160),
        (320, 1),
    )
    assert b"".join(chunk.audio for chunk in chunks) == pcm(321, 7)
    assert notifications == [None]


def test_first_activation_capacity_failure_emits_nothing_and_consumes_no_ids() -> None:
    ingress = make_ingress(capacity=10)

    assert ingress.submit(annotated(11, activated=True)) is linklab.AudioSubmitResult.OVERFLOW
    assert ingress.next_batch() is None
    assert ingress.snapshot.overflow_count == 1

    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    batch = ingress.next_batch()
    assert batch is not None
    assert batch.messages[0] == linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word")
    assert batch.messages[1] == linklab.InputStartedEvent(
        linklab.ConversationId(1), linklab.InputId(1), linklab.InputStartReason.ACTIVATION, 0
    )


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        b"x",
        memoryview(bytearray(8))[::2],
        object(),
    ],
)
def test_invalid_audio_buffers_raise_value_error_without_state(audio: object) -> None:
    ingress = make_ingress()
    item = linklab.AnnotatedAudio(audio, 0, False, False, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ingress.submit(item)
    assert ingress.next_batch() is None
    assert ingress.state.conversation_id is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation", True),
        ("generation", -1),
        ("discontinuity", 1),
        ("speech", 1),
        ("activated", 1),
        ("wake_word", ""),
    ],
)
def test_invalid_annotation_metadata_raises_value_error(field: str, value: object) -> None:
    values: dict[str, object] = {
        "audio": pcm(1),
        "generation": 0,
        "discontinuity": False,
        "speech": False,
        "activated": True,
        "wake_word": None,
    }
    values[field] = value
    item = linklab.AnnotatedAudio(**values)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_ingress().submit(item)


def test_open_input_preserves_submission_and_speech_boundaries() -> None:
    ingress = make_ingress(max_chunk=160)
    assert ingress.submit(annotated(10, activated=True, speech=False, value=1)) is linklab.AudioSubmitResult.ACCEPTED
    assert ingress.submit(annotated(161, activated=True, speech=True, value=2)) is linklab.AudioSubmitResult.ACCEPTED

    first, second = take_all(ingress)
    assert [(event.start_frame, event.speech, len(event.audio) // 2) for event in audio_messages(first)] == [
        (0, False, 10)
    ]
    assert [(event.start_frame, event.speech, len(event.audio) // 2) for event in audio_messages(second)] == [
        (10, True, 160),
        (170, True, 1),
    ]


def test_aggregate_limit_rejection_is_a_value_error_without_partial_commit() -> None:
    ingress = make_ingress(capacity=20_000, max_input_frames=16_000)
    assert ingress.submit(annotated(15_999, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    before = ingress.state
    with pytest.raises(ValueError, match="max_input_frames"):
        ingress.submit(annotated(2, activated=True))

    assert ingress.state == before
    assert ingress.snapshot.occupancy_frames == 15_999


def test_open_input_overflow_rejects_range_and_queues_one_abort() -> None:
    ingress = make_ingress(capacity=10)
    assert ingress.submit(annotated(6, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    assert ingress.submit(annotated(5, activated=True)) is linklab.AudioSubmitResult.OVERFLOW
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.CLOSED_INPUT

    data, control = take_all(ingress)
    assert data.audio_frames == 6
    assert control.lane is _IngressLane.CONTROL
    assert control.messages == (
        linklab.InputAbortedEvent(linklab.ConversationId(1), linklab.InputId(1), linklab.InputAbortReason.OVERFLOW),
    )


def test_capture_failure_aborts_exactly_once_and_rejects_later_pcm() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    assert ingress.abort_input(linklab.InputAbortReason.CAPTURE_FAILED) is True
    assert ingress.abort_input(linklab.InputAbortReason.CAPTURE_FAILED) is False
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.CLOSED_INPUT

    batches = take_all(ingress)
    aborts = [
        message for batch in batches for message in batch.messages if isinstance(message, linklab.InputAbortedEvent)
    ]
    assert aborts == [
        linklab.InputAbortedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputAbortReason.CAPTURE_FAILED,
        )
    ]


def test_conversation_cancel_is_thread_safe_and_enqueued_once() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(ingress.cancel_conversation, linklab.ConversationCancelReason.USER).result() is True
        assert executor.submit(ingress.cancel_conversation, linklab.ConversationCancelReason.USER).result() is False

    batches = take_all(ingress)
    assert batches[-1].messages == (
        linklab.ConversationCancelledEvent(linklab.ConversationId(1), linklab.ConversationCancelReason.USER),
    )


def test_playback_accounting_uses_known_output_and_accepts_one_outcome() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)
    finish_input_and_response(ingress, linklab.InputId(1), 10, finish_response=False)
    ingress.accept_inbound(
        linklab.OutputStartedEvent(linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1))
    )
    ingress.accept_inbound(
        linklab.OutputAudioEvent(
            linklab.ConversationId(1),
            linklab.ResponseId(1),
            linklab.OutputId(1),
            0,
            pcm(4),
        )
    )
    ingress.accept_inbound(
        linklab.OutputEndedEvent(
            linklab.ConversationId(1),
            linklab.ResponseId(1),
            linklab.OutputId(1),
            4,
        )
    )

    assert ingress.playback_finished(linklab.OutputId(2), 4) is False
    assert ingress.playback_finished(linklab.OutputId(1), 3) is False
    assert ingress.playback_finished(linklab.OutputId(1), 4) is True
    assert ingress.playback_finished(linklab.OutputId(1), 4) is False

    batch = ingress.next_batch()
    assert batch is not None
    assert batch.messages == (
        linklab.PlaybackFinishedEvent(
            linklab.ConversationId(1),
            linklab.ResponseId(1),
            linklab.OutputId(1),
            4,
        ),
    )


def test_server_close_discards_unsent_audio_and_returns_closed_input() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    first = ingress.next_batch()
    assert first is not None
    ingress.acknowledge(first)
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    ingress.accept_inbound(
        linklab.InputClosedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            10,
            linklab.InputCloseReason.ENDPOINT,
        )
    )

    assert ingress.next_batch() is None
    assert ingress.snapshot.occupancy_frames == 0
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.CLOSED_INPUT


def test_waiting_silence_is_bounded_and_evicted_at_exact_frame_boundary() -> None:
    ingress = make_ingress(pre_roll=5, max_chunk=160)
    assert ingress.submit(annotated(2, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)
    finish_input_and_response(ingress, linklab.InputId(1), 2)

    assert ingress.submit(annotated(4, activated=True, value=3)) is linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
    assert ingress.submit(annotated(4, activated=True, value=4)) is linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
    assert ingress.submit(annotated(2, activated=True, speech=True, value=5)) is linklab.AudioSubmitResult.ACCEPTED

    batch = ingress.next_batch()
    assert batch is not None
    events = audio_messages(batch)
    assert [(event.start_frame, len(event.audio) // 2, event.speech) for event in events] == [
        (0, 1, False),
        (1, 4, False),
        (5, 2, True),
    ]
    assert events[0].audio == pcm(1, 3)
    assert events[1].audio == pcm(4, 4)


def test_generation_boundary_aborts_old_input_and_preserves_chunk_for_future_input() -> None:
    ingress = make_ingress(pre_roll=20)
    assert ingress.submit(annotated(5, generation=1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)

    boundary = ingress.submit(annotated(4, generation=2, discontinuity=True, activated=True, value=6))
    assert boundary is linklab.AudioSubmitResult.CLOSED_INPUT
    abort_batch = ingress.next_batch()
    assert abort_batch is not None
    assert abort_batch.messages == (
        linklab.InputAbortedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputAbortReason.DISCONTINUITY,
        ),
    )
    ingress.acknowledge(abort_batch)

    ingress.accept_inbound(linklab.TranscriptFinalEvent(linklab.ConversationId(1), linklab.InputId(1), ""))
    ingress.accept_inbound(
        linklab.ResponseStartedEvent(linklab.ConversationId(1), linklab.ResponseId(1), linklab.InputId(1), False)
    )
    ingress.accept_inbound(linklab.ResponseEndedEvent(linklab.ConversationId(1), linklab.ResponseId(1)))

    assert (
        ingress.submit(annotated(2, generation=2, activated=True, speech=True, value=7))
        is linklab.AudioSubmitResult.ACCEPTED
    )
    batch = ingress.next_batch()
    assert batch is not None
    events = audio_messages(batch)
    assert [(event.start_frame, len(event.audio) // 2, event.speech) for event in events] == [
        (0, 4, False),
        (4, 2, True),
    ]
    assert events[0].audio == pcm(4, 6)


def test_generation_change_without_discontinuity_is_still_a_boundary() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(2, generation=1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)

    assert (
        ingress.submit(annotated(2, generation=2, discontinuity=False, activated=True))
        is linklab.AudioSubmitResult.CLOSED_INPUT
    )
    batch = ingress.next_batch()
    assert batch is not None
    assert batch.messages == (
        linklab.InputAbortedEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            linklab.InputAbortReason.DISCONTINUITY,
        ),
    )


def test_barge_in_starts_immediately_before_server_acknowledgement() -> None:
    ingress = make_ingress(pre_roll=10)
    assert ingress.submit(annotated(2, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)
    finish_input_and_response(ingress, linklab.InputId(1), 2, finish_response=False)

    assert ingress.submit(annotated(3, activated=True, value=3)) is linklab.AudioSubmitResult.IGNORED_WAITING_SILENCE
    assert ingress.submit(annotated(2, activated=True, speech=True, value=4)) is linklab.AudioSubmitResult.ACCEPTED

    batch = ingress.next_batch()
    assert batch is not None
    assert batch.messages[0] == linklab.InputStartedEvent(
        linklab.ConversationId(1),
        linklab.InputId(2),
        linklab.InputStartReason.BARGE_IN,
        0,
        linklab.ResponseId(1),
    )
    assert [(event.speech, len(event.audio) // 2) for event in audio_messages(batch)] == [
        (False, 3),
        (True, 2),
    ]
    assert ingress.state.input_id == linklab.InputId(2)
    assert ingress.state.response_id is None


def test_stop_flushes_pending_work_and_resets_connection() -> None:
    ingress = make_ingress()
    assert ingress.submit(annotated(10, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    ingress.stop()

    assert ingress.next_batch() is None
    assert ingress.snapshot.occupancy_frames == 0
    assert ingress.state.connection_state is linklab.ConnectionState.DISCONNECTED
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.IGNORED_INACTIVE


def test_notifier_failure_is_retained_for_loop_failure_handling() -> None:
    calls = 0

    def fail_notify() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("loop closed")

    ingress = make_ingress(notify=fail_notify)
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    assert isinstance(ingress.notification_error, RuntimeError)

    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    assert calls == 2


def test_reserved_control_exhaustion_sets_unrecoverable_failure() -> None:
    ingress = make_ingress(capacity=1, control_capacity=1)
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    # Simulate another reserved control producer until task 13a owns the shared handoff.
    with ingress._lock:
        ingress._control_occupancy = 1
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.OVERFLOW
    assert isinstance(ingress.failure, RuntimeError)


def test_threaded_submission_is_serialized_without_duplicate_ranges() -> None:
    ingress = make_ingress(capacity=1_000)
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED
    take_all(ingress)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(
            executor.map(lambda value: ingress.submit(annotated(10, activated=True, value=value)), range(8))
        )

    assert results == (linklab.AudioSubmitResult.ACCEPTED,) * 8
    batches = take_all(ingress)
    events = [event for batch in batches for event in audio_messages(batch)]
    assert sorted(event.start_frame for event in events) == list(range(1, 81, 10))
    assert all(len(event.audio) == 20 for event in events)
    assert ingress.snapshot.occupancy_frames == 0


def test_threaded_overflow_has_one_terminal_abort() -> None:
    ingress = make_ingress(capacity=51)
    assert ingress.submit(annotated(1, activated=True)) is linklab.AudioSubmitResult.ACCEPTED

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = tuple(executor.map(lambda _: ingress.submit(annotated(10, activated=True)), range(10)))

    assert results.count(linklab.AudioSubmitResult.ACCEPTED) == 5
    assert results.count(linklab.AudioSubmitResult.OVERFLOW) == 1
    assert results.count(linklab.AudioSubmitResult.CLOSED_INPUT) == 4
    aborts = [
        message
        for batch in take_all(ingress)
        for message in batch.messages
        if isinstance(message, linklab.InputAbortedEvent)
    ]
    assert len(aborts) == 1
