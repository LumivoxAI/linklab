from collections.abc import Callable

import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)


def ready_validator(
    role: linklab.EndpointRole = linklab.EndpointRole.SERVER,
    *,
    limits: linklab.ConnectionLimits | None = None,
) -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(role)
    validator.transport_connected()
    validator.accept(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
    validator.accept(
        linklab.ServerHello(
            1,
            CAPABILITIES,
            PCM_16K,
            limits or linklab.ConnectionLimits(max_output_audio_frames=1_600),
        )
    )
    validator.accept(linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    return validator


def start_input(validator: linklab.ProtocolValidator, *, generation: int = 0) -> None:
    validator.accept(
        linklab.InputStartedEvent(
            CONVERSATION_ID,
            INPUT_ID,
            linklab.InputStartReason.ACTIVATION,
            generation,
        )
    )


def audio(start_frame: int, frames: int, *, speech: bool = True) -> linklab.InputAudioEvent:
    return linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, start_frame, speech, b"\0\0" * frames)


def test_activation_input_tracks_generation_and_is_current() -> None:
    validator = ready_validator()
    start_input(validator, generation=4_294_967_295)

    assert validator.state.input_id == INPUT_ID
    assert validator._data.inputs[-1].start.generation == 4_294_967_295


@pytest.mark.parametrize("reason", [linklab.InputStartReason.SPEECH, linklab.InputStartReason.BARGE_IN])
def test_first_input_must_be_activation(reason: linklab.InputStartReason) -> None:
    validator = ready_validator()
    before = validator._data
    event = linklab.InputStartedEvent(
        CONVERSATION_ID,
        INPUT_ID,
        reason,
        0,
        linklab.ResponseId(1) if reason is linklab.InputStartReason.BARGE_IN else None,
    )

    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(event)
    assert validator._data == before


def test_only_one_input_is_open_and_ids_are_monotonic() -> None:
    validator = ready_validator()
    start_input(validator)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(2),
                linklab.InputStartReason.SPEECH,
                0,
            )
        )
    assert validator._data == before


@pytest.mark.parametrize("start_frame", [1, 3, 4])
def test_input_audio_gap_overlap_reorder_and_duplicate_are_atomic(start_frame: int) -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 2))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="contiguous"):
        validator.accept(audio(start_frame, 1))
    assert validator._data == before


def test_audio_honors_negotiated_per_message_limit() -> None:
    limits = linklab.ConnectionLimits(max_input_audio_frames=160, max_output_audio_frames=1_600)
    validator = ready_validator(limits=limits)
    start_input(validator)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="message limit"):
        validator.accept(audio(0, 161))
    assert validator._data == before


def test_received_pcm_does_not_advance_callback_committed_boundary() -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 4))

    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.FAILED))
    assert validator.tombstones[-1].terminal_state == "closed"


def test_committed_boundary_is_monotonic_received_and_immutable() -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 2))
    validator.accept(audio(2, 2))
    validator._commit_input_audio(INPUT_ID, 2)
    validator._commit_input_audio(INPUT_ID, 4)

    for invalid in (-1, 3, 5):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator._commit_input_audio(INPUT_ID, invalid)
        assert validator._data == before

    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 4, linklab.InputCloseReason.ENDPOINT))
    with pytest.raises(linklab.ProtocolViolation, match="immutable"):
        validator._commit_input_audio(INPUT_ID, 4)


def test_committing_exact_aggregate_limit_plans_max_duration_close() -> None:
    limits = linklab.ConnectionLimits(max_input_frames=16_000, max_output_audio_frames=1_600)
    validator = ready_validator(limits=limits)
    start_input(validator)
    outbound: tuple[linklab.Message, ...] = ()
    for start in range(0, 16_000, 1_600):
        validator.accept(audio(start, 1_600))
        outbound = validator._commit_input_audio(INPUT_ID, start + 1_600)

    assert outbound == (
        linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 16_000, linklab.InputCloseReason.MAX_DURATION),
    )
    assert validator.tombstones[-1].terminal_state == "closed"


def test_server_close_requires_exact_committed_boundary_atomically() -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 2))
    validator.accept(audio(2, 2))
    validator._commit_input_audio(INPUT_ID, 2)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="committed"):
        validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 4, linklab.InputCloseReason.ENDPOINT))
    assert validator._data == before


def test_endpoint_race_discards_late_contiguous_pcm_but_rejects_malformed_range() -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 2))
    validator.accept(audio(2, 2))
    validator._commit_input_audio(INPUT_ID, 2)
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.ENDPOINT))

    data, result = validator._transition(validator._data, audio(4, 2, speech=False))
    assert result.dispatch is False
    validator._data = data
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="contiguous"):
        validator.accept(audio(5, 1))
    assert validator._data == before


def test_late_pcm_still_honors_negotiated_message_and_aggregate_limits() -> None:
    message_limits = linklab.ConnectionLimits(max_input_audio_frames=160, max_output_audio_frames=1_600)
    validator = ready_validator(role=linklab.EndpointRole.CLIENT, limits=message_limits)
    start_input(validator)
    validator.accept(audio(0, 2))
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.ENDPOINT))
    with pytest.raises(linklab.ProtocolViolation, match="message limit"):
        validator.accept(audio(2, 161))

    aggregate_limits = linklab.ConnectionLimits(max_input_frames=16_000, max_output_audio_frames=1_600)
    validator = ready_validator(role=linklab.EndpointRole.CLIENT, limits=aggregate_limits)
    start_input(validator)
    for start in range(0, 14_400, 1_600):
        validator.accept(audio(start, 1_600))
    validator.accept(audio(14_400, 1_599))
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 15_999, linklab.InputCloseReason.ENDPOINT))
    with pytest.raises(linklab.ProtocolViolation, match="aggregate limit"):
        validator.accept(audio(15_999, 2))


@pytest.mark.parametrize("reason", list(linklab.InputAbortReason))
def test_input_abort_reasons_and_identical_duplicate(reason: linklab.InputAbortReason) -> None:
    validator = ready_validator()
    start_input(validator)
    event = linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, reason)

    validator.accept(event)
    validator.accept(event)
    assert validator.tombstones[-1].terminal_state == "aborted"

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="conflicting"):
        validator.accept(
            linklab.InputAbortedEvent(
                CONVERSATION_ID,
                INPUT_ID,
                linklab.InputAbortReason.OVERFLOW
                if reason is not linklab.InputAbortReason.OVERFLOW
                else linklab.InputAbortReason.CAPTURE_FAILED,
            )
        )
    assert validator._data == before


@pytest.mark.parametrize("reason", list(linklab.InputCloseReason))
def test_input_close_reasons_and_identical_duplicate(reason: linklab.InputCloseReason) -> None:
    validator = ready_validator(role=linklab.EndpointRole.CLIENT)
    start_input(validator)
    validator.accept(audio(0, 2))
    event = linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, reason)

    validator.accept(event)
    validator.accept(event)
    assert validator.tombstones[-1].terminal_state == "closed"


def test_client_rejects_close_inside_a_received_chunk() -> None:
    validator = ready_validator(role=linklab.EndpointRole.CLIENT)
    start_input(validator)
    validator.accept(audio(0, 4))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="received audio chunk"):
        validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.ENDPOINT))
    assert validator._data == before


def test_transcript_updates_require_open_input_and_exact_revisions() -> None:
    validator = ready_validator()
    start_input(validator)
    first = linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, 1, "hello", "en")
    validator.accept(first)
    validator.accept(first)
    validator.accept(linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, 2, "hello world", "en"))

    for revision in (2, 4):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, revision, "conflict"))
        assert validator._data == before

    validator.accept(linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.CAPTURE_FAILED))
    with pytest.raises(linklab.ProtocolViolation, match="only while input is open"):
        validator.accept(linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, 3, "late"))


@pytest.mark.parametrize(
    ("terminal", "empty_allowed"),
    [
        (linklab.InputCloseReason.ENDPOINT, False),
        (linklab.InputCloseReason.MAX_DURATION, False),
        (linklab.InputCloseReason.NO_SPEECH, True),
        (linklab.InputCloseReason.FAILED, True),
    ],
)
def test_final_transcript_empty_context_and_response_prerequisite(
    terminal: linklab.InputCloseReason, empty_allowed: bool
) -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, terminal))
    final = linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "")

    if not empty_allowed:
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation, match="empty"):
            validator.accept(final)
        assert validator._data == before
        final = linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "final")

    assert validator._response_input_ready(INPUT_ID) is False
    validator.accept(final)
    validator.accept(final)
    assert validator._response_input_ready(INPUT_ID) is True

    with pytest.raises(linklab.ProtocolViolation, match="conflicting"):
        validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "different"))


def test_abort_allows_empty_final_but_open_input_does_not_allow_any_final() -> None:
    validator = ready_validator()
    start_input(validator)
    final = linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "")
    with pytest.raises(linklab.ProtocolViolation, match="terminal input"):
        validator.accept(final)

    validator.accept(linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.DISCONTINUITY))
    validator.accept(final)
    assert validator._response_input_ready(INPUT_ID)


def test_normal_conversation_end_requires_final_transcript() -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.DISCONTINUITY))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="final transcript"):
        validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))
    assert validator._data == before


@pytest.mark.parametrize(
    "code",
    [
        linklab.ErrorCode.INPUT_DISCONTINUITY,
        linklab.ErrorCode.INPUT_OVERFLOW,
        linklab.ErrorCode.INPUT_TOO_LONG,
        linklab.ErrorCode.CAPTURE_FAILED,
        linklab.ErrorCode.STT_FAILED,
        linklab.ErrorCode.PROCESSING_TIMEOUT,
    ],
)
def test_input_error_plan_closes_at_committed_boundary_and_finalizes(code: linklab.ErrorCode) -> None:
    validator = ready_validator()
    start_input(validator)
    validator.accept(audio(0, 2))
    validator.accept(audio(2, 2))
    validator._commit_input_audio(INPUT_ID, 2)
    error = linklab.ErrorEvent(linklab.ErrorScope.INPUT, code, True, CONVERSATION_ID, INPUT_ID)

    data, result = validator._transition(validator._data, error)
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.INPUT,)
    assert result.outbound == (
        linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.FAILED),
        linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""),
    )
    validator._data = data
    assert validator._response_input_ready(INPUT_ID)


def test_client_input_error_waits_for_and_dispatches_wire_terminals() -> None:
    validator = ready_validator(role=linklab.EndpointRole.CLIENT)
    start_input(validator)
    validator.accept(audio(0, 2))
    error = linklab.ErrorEvent(
        linklab.ErrorScope.INPUT,
        linklab.ErrorCode.STT_FAILED,
        True,
        CONVERSATION_ID,
        INPUT_ID,
    )

    data, error_result = validator._transition(validator._data, error)
    assert error_result.outbound == ()
    assert error_result.dispatch is True
    validator._data = data
    with pytest.raises(linklab.ProtocolViolation, match="only while input is open"):
        validator.accept(linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, 1, "late"))
    with pytest.raises(linklab.ProtocolViolation, match="failed input"):
        validator.accept(audio(2, 1))
    with pytest.raises(linklab.ProtocolViolation, match="failed input"):
        validator.accept(linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.CAPTURE_FAILED))

    close = linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.FAILED)
    data, close_result = validator._transition(validator._data, close)
    assert close_result.dispatch is True
    validator._data = data

    final = linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "")
    data, final_result = validator._transition(validator._data, final)
    assert final_result.dispatch is True
    validator._data = data
    assert validator._response_input_ready(INPUT_ID)


def test_failure_after_max_duration_allows_empty_final_and_waiting_recovery() -> None:
    validator = ready_validator(role=linklab.EndpointRole.CLIENT)
    start_input(validator)
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.LISTENING))
    close = linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.MAX_DURATION)
    validator.accept(close)
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.PROCESSING))
    validator.accept(
        linklab.ErrorEvent(
            linklab.ErrorScope.INPUT,
            linklab.ErrorCode.PROCESSING_TIMEOUT,
            True,
            CONVERSATION_ID,
            INPUT_ID,
        )
    )
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 3, linklab.CoarseState.WAITING))

    assert validator.state.input_id is None
    validator.accept(
        linklab.InputStartedEvent(
            CONVERSATION_ID,
            linklab.InputId(2),
            linklab.InputStartReason.SPEECH,
            0,
        )
    )
    assert validator.state.input_id == linklab.InputId(2)


def test_client_aggregate_crossing_is_rejected_without_server_error_plan() -> None:
    limits = linklab.ConnectionLimits(max_input_frames=16_000, max_output_audio_frames=1_600)
    validator = ready_validator(role=linklab.EndpointRole.CLIENT, limits=limits)
    start_input(validator)
    for start in range(0, 14_400, 1_600):
        validator.accept(audio(start, 1_600))
    validator.accept(audio(14_400, 1_599))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="aggregate limit"):
        validator.accept(audio(15_999, 2))
    assert validator._data == before


def test_aggregate_frame_crossing_rejects_whole_chunk_with_input_error_plan() -> None:
    limits = linklab.ConnectionLimits(max_input_frames=16_000, max_output_audio_frames=1_600)
    validator = ready_validator(limits=limits)
    start_input(validator)
    for start in range(0, 14_400, 1_600):
        validator.accept(audio(start, 1_600))
        validator._commit_input_audio(INPUT_ID, start + 1_600)
    validator.accept(audio(14_400, 1_599))
    validator._commit_input_audio(INPUT_ID, 15_999)

    data, result = validator._transition(validator._data, audio(15_999, 2))
    assert isinstance(result.outbound[0], linklab.ErrorEvent)
    assert result.outbound[0].code is linklab.ErrorCode.INPUT_TOO_LONG
    assert result.outbound[1:] == (
        linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 15_999, linklab.InputCloseReason.FAILED),
        linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""),
    )
    assert data.inputs[-1].received_end_frame == 15_999


@pytest.mark.parametrize(
    "terminal_action",
    [
        lambda validator: validator.accept(
            linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER)
        ),
        lambda validator: validator.accept(
            linklab.ErrorEvent(
                linklab.ErrorScope.CONVERSATION,
                linklab.ErrorCode.CONVERSATION_FAILED,
                True,
                CONVERSATION_ID,
            )
        ),
    ],
)
def test_conversation_terminal_action_terminates_open_input(
    terminal_action: Callable[[linklab.ProtocolValidator], None],
) -> None:
    validator = ready_validator()
    start_input(validator)
    terminal_action(validator)
    assert validator.tombstones[-1] == linklab.ProtocolTombstone(
        linklab.ProtocolObjectKind.INPUT,
        CONVERSATION_ID,
        INPUT_ID,
        None,
        None,
        "aborted",
    )
    assert validator._response_input_ready(INPUT_ID) is False
    with pytest.raises(linklab.ProtocolViolation, match="terminal conversation"):
        validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "final"))
    with pytest.raises(linklab.ProtocolViolation, match="terminal conversation"):
        validator.accept(
            linklab.ErrorEvent(
                linklab.ErrorScope.INPUT,
                linklab.ErrorCode.STT_FAILED,
                True,
                CONVERSATION_ID,
                INPUT_ID,
            )
        )


def test_state_must_match_current_input_lifecycle() -> None:
    validator = ready_validator()
    start_input(validator)
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="listening"):
        validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.WAITING))
    assert validator._data == before

    validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.LISTENING))
    validator.accept(linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.OVERFLOW))
    with pytest.raises(linklab.ProtocolViolation, match="processing"):
        validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.WAITING))
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.PROCESSING))


def test_late_contiguous_pcm_is_checked_after_conversation_end() -> None:
    validator = ready_validator(role=linklab.EndpointRole.CLIENT)
    start_input(validator)
    validator.accept(audio(0, 2))
    close = linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.NO_SPEECH)
    validator.accept(close)
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))

    validator.accept(close)
    validator.accept(audio(2, 1))
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="contiguous"):
        validator.accept(audio(2, 1))
    assert validator._data == before
