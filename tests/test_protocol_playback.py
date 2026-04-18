import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)
RESPONSE_ID = linklab.ResponseId(1)
OUTPUT_ID = linklab.OutputId(1)


def responding_validator(
    *,
    role: linklab.EndpointRole = linklab.EndpointRole.SERVER,
    end_conversation: bool = False,
    start_output: bool = True,
    end_output: bool = True,
    end_response: bool = False,
) -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(role)
    validator.transport_connected()
    validator.accept(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
    validator.accept(
        linklab.ServerHello(
            1,
            CAPABILITIES,
            PCM_16K,
            linklab.ConnectionLimits(max_output_audio_frames=1_600),
        )
    )
    validator.accept(linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.ACTIVATION, 0))
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.NO_SPEECH))
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, end_conversation))
    if start_output:
        validator.accept(linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID))
        validator.accept(linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\0\0" * 4))
        if end_output:
            validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4))
    if end_response:
        validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
    return validator


def finished(frames: int = 4) -> linklab.PlaybackFinishedEvent:
    return linklab.PlaybackFinishedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, frames)


def interrupted(reason: linklab.PlaybackInterruptReason, frames: int = 2) -> linklab.PlaybackInterruptedEvent:
    return linklab.PlaybackInterruptedEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        OUTPUT_ID,
        frames,
        linklab.PlaybackPosition.EXACT,
        reason,
    )


@pytest.mark.parametrize("outcome_before_response_end", [False, True])
@pytest.mark.parametrize("end_conversation", [False, True])
def test_finished_playback_completes_after_both_required_terminals(
    outcome_before_response_end: bool, end_conversation: bool
) -> None:
    validator = responding_validator(end_conversation=end_conversation)
    outcome = finished()

    if outcome_before_response_end:
        validator.accept(outcome)
        awaiting_response_end = validator.state
        assert awaiting_response_end.response_id == RESPONSE_ID
        validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
    else:
        validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
        awaiting_playback = validator.state
        assert awaiting_playback.response_id == RESPONSE_ID
        validator.accept(outcome)

    validator.accept(outcome)
    completed = validator.state
    assert (completed.response_id, completed.output_id) == (None, None)
    assert validator.tombstones[-1].terminal_state == "completed"
    if end_conversation:
        validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))
    else:
        validator.accept(
            linklab.InputStartedEvent(CONVERSATION_ID, linklab.InputId(2), linklab.InputStartReason.SPEECH, 0)
        )


def test_finished_requires_output_end_and_exact_total_atomically() -> None:
    validator = responding_validator(end_output=False)

    for outcome in (finished(), finished(3)):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(outcome)
        assert validator._data == before

    validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4))
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="output total"):
        validator.accept(finished(3))
    assert validator._data == before


@pytest.mark.parametrize(
    ("reason", "cancel_reason"),
    [
        (linklab.PlaybackInterruptReason.BARGE_IN, linklab.ResponseCancelReason.BARGE_IN),
        (linklab.PlaybackInterruptReason.LOCAL_CANCEL, linklab.ResponseCancelReason.LOCAL_CANCEL),
        (linklab.PlaybackInterruptReason.SHUTDOWN, linklab.ResponseCancelReason.SHUTDOWN),
    ],
)
@pytest.mark.parametrize("end_conversation", [False, True])
def test_non_error_interruption_cancels_response_and_classifies_outbound(
    reason: linklab.PlaybackInterruptReason,
    cancel_reason: linklab.ResponseCancelReason,
    end_conversation: bool,
) -> None:
    validator = responding_validator(end_conversation=end_conversation, end_output=False)
    outcome = interrupted(reason)

    data, result = validator._transition(validator._data, outcome)
    assert result.outbound == (linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, cancel_reason),)
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.RESPONSE,)
    assert result.error_code is None
    assert data.conversation is not None
    expected_end = (
        linklab.ConversationEndReason.CANCELLED
        if reason is linklab.PlaybackInterruptReason.SHUTDOWN
        or (reason is linklab.PlaybackInterruptReason.LOCAL_CANCEL and end_conversation)
        else None
    )
    assert data.conversation.expected_end_reason is expected_end
    validator._data = data

    late_text = linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "late")
    data, late_result = validator._transition(validator._data, late_text)
    assert late_result.dispatch is False
    validator._data = data
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="cancelled response"):
        validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
    assert validator._data == before


@pytest.mark.parametrize(
    "reason",
    [linklab.PlaybackInterruptReason.PLAYBACK_FAILED, linklab.PlaybackInterruptReason.OVERFLOW],
)
def test_playback_failures_are_stale_and_classified_for_error_plans(
    reason: linklab.PlaybackInterruptReason,
) -> None:
    validator = responding_validator(end_output=False)

    data, result = validator._transition(validator._data, interrupted(reason))
    assert result.error_code is linklab.ErrorCode.PLAYBACK_FAILED
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.RESPONSE,)
    assert result.outbound == ()
    validator._data = data

    late_audio = linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4, b"\0\0")
    _, late_result = validator._transition(validator._data, late_audio)
    assert late_result.dispatch is False


def test_interruption_frame_bound_conflict_and_unknown_ids_are_atomic() -> None:
    validator = responding_validator(end_output=False)
    invalid = (
        interrupted(linklab.PlaybackInterruptReason.LOCAL_CANCEL, 5),
        linklab.PlaybackInterruptedEvent(
            CONVERSATION_ID,
            linklab.ResponseId(2),
            OUTPUT_ID,
            0,
            linklab.PlaybackPosition.ESTIMATED,
            linklab.PlaybackInterruptReason.LOCAL_CANCEL,
        ),
        linklab.PlaybackInterruptedEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.OutputId(2),
            0,
            linklab.PlaybackPosition.ESTIMATED,
            linklab.PlaybackInterruptReason.LOCAL_CANCEL,
        ),
    )
    for outcome in invalid:
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(outcome)
        assert validator._data == before

    outcome = interrupted(linklab.PlaybackInterruptReason.LOCAL_CANCEL)
    validator.accept(outcome)
    validator.accept(outcome)
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="conflicting playback"):
        validator.accept(interrupted(linklab.PlaybackInterruptReason.SHUTDOWN))
    assert validator._data == before


def test_one_post_cancel_playback_outcome_is_observability_only() -> None:
    validator = responding_validator(end_output=False)
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.GENERATION_FAILED,
        )
    )
    outcome = interrupted(linklab.PlaybackInterruptReason.LOCAL_CANCEL)

    data, result = validator._transition(validator._data, outcome)
    assert result.dispatch is True
    assert result.outbound == ()
    validator._data = data
    validator.accept(outcome)
    with pytest.raises(linklab.ProtocolViolation, match="conflicting playback"):
        validator.accept(interrupted(linklab.PlaybackInterruptReason.SHUTDOWN))


@pytest.mark.parametrize("playback_first", [False, True])
def test_barge_in_atomically_stales_response_and_revokes_terminal_intent(playback_first: bool) -> None:
    validator = responding_validator(end_conversation=True, end_output=False)
    outcome = interrupted(linklab.PlaybackInterruptReason.BARGE_IN)
    barge = linklab.InputStartedEvent(
        CONVERSATION_ID,
        linklab.InputId(2),
        linklab.InputStartReason.BARGE_IN,
        0,
        RESPONSE_ID,
    )

    if playback_first:
        validator.accept(outcome)
        validator.accept(barge)
    else:
        data, result = validator._transition(validator._data, barge)
        assert result.outbound == (
            linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, linklab.ResponseCancelReason.BARGE_IN),
        )
        assert result.cancellation_targets == (linklab.ProtocolObjectKind.RESPONSE,)
        validator._data = data
        validator.accept(outcome)

    assert validator.state.input_id == linklab.InputId(2)
    assert validator.state.response_id is None
    assert validator._data.conversation is not None
    assert validator._data.conversation.expected_end_reason is None
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.LISTENING))

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="current response"):
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(3),
                linklab.InputStartReason.BARGE_IN,
                0,
                RESPONSE_ID,
            )
        )
    assert validator._data == before


def test_barge_in_cancels_response_awaiting_playback() -> None:
    validator = responding_validator(end_conversation=True, end_response=True)
    barge = linklab.InputStartedEvent(
        CONVERSATION_ID,
        linklab.InputId(2),
        linklab.InputStartReason.BARGE_IN,
        0,
        RESPONSE_ID,
    )

    data, result = validator._transition(validator._data, barge)
    assert result.outbound == (
        linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, linklab.ResponseCancelReason.BARGE_IN),
    )
    assert data.conversation is not None
    assert data.conversation.response_id is None
    assert data.conversation.input_id == linklab.InputId(2)
    assert data.conversation.expected_end_reason is None


def test_playback_first_barge_in_reference_expires_after_next_input() -> None:
    validator = responding_validator(end_output=False)
    validator.accept(interrupted(linklab.PlaybackInterruptReason.BARGE_IN))
    second_input = linklab.InputId(2)
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, second_input, linklab.InputStartReason.SPEECH, 0))
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, second_input, 0, linklab.InputCloseReason.NO_SPEECH))
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, second_input, ""))
    validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, linklab.ResponseId(2), second_input, False))
    validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, linklab.ResponseId(2)))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="current response"):
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(3),
                linklab.InputStartReason.BARGE_IN,
                0,
                RESPONSE_ID,
            )
        )
    assert validator._data == before


def test_completed_response_cannot_be_cancelled_or_corrupt_later_input() -> None:
    validator = responding_validator(end_response=True)
    validator.accept(finished())
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, linklab.InputId(2), linklab.InputStartReason.SPEECH, 0))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="completed response"):
        validator.accept(
            linklab.ResponseCancelledEvent(
                CONVERSATION_ID,
                RESPONSE_ID,
                linklab.ResponseCancelReason.LOCAL_CANCEL,
            )
        )
    assert validator._data == before
    assert validator.state.input_id == linklab.InputId(2)


def test_identical_response_cancellation_is_idempotent_after_conversation_end() -> None:
    validator = responding_validator(end_conversation=True, end_output=False)
    outcome = interrupted(linklab.PlaybackInterruptReason.LOCAL_CANCEL)
    validator.accept(outcome)
    validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.CANCELLED))
    cancel = linklab.ResponseCancelledEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        linklab.ResponseCancelReason.LOCAL_CANCEL,
    )

    validator.accept(cancel)


def test_late_output_start_is_stale_and_immediately_tombstoned() -> None:
    validator = responding_validator(start_output=False)
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.GENERATION_FAILED,
        )
    )

    data, result = validator._transition(
        validator._data,
        linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID),
    )
    assert result.dispatch is False
    assert data.tombstones[-1] == linklab.ProtocolTombstone(
        linklab.ProtocolObjectKind.OUTPUT,
        CONVERSATION_ID,
        None,
        RESPONSE_ID,
        OUTPUT_ID,
        "cancelled",
    )
    validator._data = data
    audio = linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\0\0" * 4)
    data, audio_result = validator._transition(validator._data, audio)
    assert audio_result.dispatch is False
    validator._data = data
    data, end_result = validator._transition(
        validator._data,
        linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4),
    )
    assert end_result.dispatch is False
    assert data.tombstones[-1].terminal_state == "cancelled"


def test_barge_in_rejects_noncurrent_response_reference_atomically() -> None:
    validator = responding_validator(end_output=False)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="known response"):
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(2),
                linklab.InputStartReason.BARGE_IN,
                0,
                linklab.ResponseId(2),
            )
        )
    assert validator._data == before
