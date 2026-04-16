from collections.abc import Callable

import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)
RESPONSE_ID = linklab.ResponseId(1)
OUTPUT_ID = linklab.OutputId(1)


def ready_validator() -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
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
    return validator


def start_response(
    validator: linklab.ProtocolValidator, *, end_conversation: bool = False
) -> linklab.ResponseStartedEvent:
    event = linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, end_conversation)
    validator.accept(event)
    return event


def start_output(validator: linklab.ProtocolValidator) -> None:
    validator.accept(linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID))


def output_audio(start_frame: int, frames: int) -> linklab.OutputAudioEvent:
    return linklab.OutputAudioEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        OUTPUT_ID,
        start_frame,
        b"\0\0" * frames,
    )


@pytest.mark.parametrize("end_conversation", [False, True])
def test_no_text_no_output_response_completion_honors_terminal_intent(end_conversation: bool) -> None:
    validator = ready_validator()
    start_response(validator, end_conversation=end_conversation)

    end = linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID)
    validator.accept(end)
    validator.accept(end)
    assert (validator.state.response_id, validator.state.input_id) == (None, None)
    assert validator.tombstones[-1] == linklab.ProtocolTombstone(
        linklab.ProtocolObjectKind.RESPONSE,
        CONVERSATION_ID,
        None,
        RESPONSE_ID,
        None,
        "completed",
    )

    if end_conversation:
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation, match="terminal end"):
            validator.accept(
                linklab.InputStartedEvent(
                    CONVERSATION_ID,
                    linklab.InputId(2),
                    linklab.InputStartReason.SPEECH,
                    0,
                )
            )
        assert validator._data == before
        validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))
    else:
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(2),
                linklab.InputStartReason.SPEECH,
                0,
            )
        )


def test_response_requires_current_finalized_terminal_input_and_one_response_per_input() -> None:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
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
    response = linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, False)

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(response)
    assert validator._data == before

    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.NO_SPEECH))
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(response)
    assert validator._data == before

    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(response)
    assert validator._response_input_ready(INPUT_ID) is False
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="current response"):
        validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, linklab.ResponseId(2), INPUT_ID, False))
    assert validator._data == before


@pytest.mark.parametrize(
    "event",
    [
        linklab.ResponseStartedEvent(CONVERSATION_ID, linklab.ResponseId(2), INPUT_ID, False),
        linklab.ResponseStartedEvent(linklab.ConversationId(2), RESPONSE_ID, INPUT_ID, False),
        linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, linklab.InputId(2), False),
    ],
)
def test_response_start_rejects_skipped_id_and_wrong_parents_atomically(
    event: linklab.ResponseStartedEvent,
) -> None:
    validator = ready_validator()
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(event)
    assert validator._data == before


def test_response_text_sequence_and_final_must_be_canonical() -> None:
    validator = ready_validator()
    start_response(validator)
    validator.accept(linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "hello "))
    validator.accept(linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 1, "world"))

    for event in (
        linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 1, "duplicate"),
        linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 3, "skipped"),
        linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "different"),
    ):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(event)
        assert validator._data == before

    final = linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "hello world")
    validator.accept(final)
    for event in (
        final,
        linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "conflict"),
        linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 2, "late"),
    ):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(event)
        assert validator._data == before


def test_nonempty_text_final_is_valid_without_deltas_but_not_after_response_end() -> None:
    validator = ready_validator()
    start_response(validator)
    validator.accept(linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "standalone"))
    validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="response end"):
        validator.accept(linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "late"))
    assert validator._data == before


def test_cancellation_after_output_response_end_does_not_make_late_text_legal() -> None:
    validator = ready_validator()
    start_response(validator)
    finish_response_with_output(validator)
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.LOCAL_CANCEL,
        )
    )

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="response end"):
        validator.accept(linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "late"))
    assert validator._data == before


def test_output_audio_is_contiguous_bounded_nonempty_and_finishes_before_response() -> None:
    validator = ready_validator()
    start_response(validator)
    start_output(validator)
    assert validator.state.output_id == OUTPUT_ID
    validator.accept(output_audio(0, 2))
    validator.accept(output_audio(2, 3))

    for event in (
        output_audio(4, 1),
        linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4),
        linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID),
    ):
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(event)
        assert validator._data == before

    validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 5))
    assert validator.tombstones[-1].terminal_state == "ended"
    validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
    assert validator.state.response_id == RESPONSE_ID
    assert validator.state.output_id == OUTPUT_ID


def test_empty_multiple_and_wrong_parent_output_are_rejected_atomically() -> None:
    validator = ready_validator()
    start_response(validator)
    start_output(validator)

    invalid_events: tuple[linklab.Message, ...] = (
        linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, linklab.OutputId(2)),
        linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, linklab.OutputId(2), 0, b"\0\0"),
        linklab.OutputAudioEvent(CONVERSATION_ID, linklab.ResponseId(2), OUTPUT_ID, 0, b"\0\0"),
        linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0),
    )
    for event in invalid_events:
        before = validator._data
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(event)
        assert validator._data == before


def test_output_honors_negotiated_per_message_limit() -> None:
    validator = ready_validator()
    start_response(validator)
    start_output(validator)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="message limit"):
        validator.accept(output_audio(0, 1_601))
    assert validator._data == before


CancellationSetup = Callable[[linklab.ProtocolValidator], None]


def add_partial_text(validator: linklab.ProtocolValidator) -> None:
    validator.accept(linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "partial"))


def add_output_audio(validator: linklab.ProtocolValidator) -> None:
    start_output(validator)
    validator.accept(output_audio(0, 2))


def finish_output(validator: linklab.ProtocolValidator) -> None:
    add_output_audio(validator)
    validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 2))


def finish_response_with_output(validator: linklab.ProtocolValidator) -> None:
    finish_output(validator)
    validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))


@pytest.mark.parametrize(
    "prepare",
    [
        lambda validator: None,
        add_partial_text,
        start_output,
        add_output_audio,
        finish_output,
        finish_response_with_output,
    ],
)
def test_response_can_be_cancelled_at_every_live_point(prepare: CancellationSetup) -> None:
    validator = ready_validator()
    start_response(validator)
    prepare(validator)
    cancel = linklab.ResponseCancelledEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        linklab.ResponseCancelReason.GENERATION_FAILED,
    )

    validator.accept(cancel)
    validator.accept(cancel)
    assert validator.state.response_id is None
    assert validator.state.output_id is None
    assert validator.tombstones[-1].terminal_state == "cancelled"


@pytest.mark.parametrize("end_conversation", [False, True])
def test_failure_cancellation_applies_response_terminal_intent(end_conversation: bool) -> None:
    validator = ready_validator()
    start_response(validator, end_conversation=end_conversation)
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.TTS_FAILED,
        )
    )

    if end_conversation:
        with pytest.raises(linklab.ProtocolViolation, match="server_failed"):
            validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))
        validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.SERVER_FAILED))
    else:
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(2),
                linklab.InputStartReason.SPEECH,
                0,
            )
        )


def test_conversation_cancel_requires_matching_response_cancellation_reason() -> None:
    validator = ready_validator()
    start_response(validator)
    validator.accept(linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER))
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="matching"):
        validator.accept(
            linklab.ResponseCancelledEvent(
                CONVERSATION_ID,
                RESPONSE_ID,
                linklab.ResponseCancelReason.GENERATION_FAILED,
            )
        )
    assert validator._data == before
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.CONVERSATION_CANCELLED,
        )
    )


def test_cancelled_response_discards_but_validates_late_text_and_output() -> None:
    validator = ready_validator()
    start_response(validator)
    validator.accept(linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "a"))
    start_output(validator)
    validator.accept(output_audio(0, 2))
    validator.accept(
        linklab.ResponseCancelledEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            linklab.ResponseCancelReason.GENERATION_FAILED,
        )
    )

    for event in (
        linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 1, "b"),
        output_audio(2, 2),
    ):
        data, result = validator._transition(validator._data, event)
        assert result.dispatch is False
        validator._data = data

    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="sequence"):
        validator.accept(linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 3, "gap"))
    assert validator._data == before
    with pytest.raises(linklab.ProtocolViolation, match="contiguous"):
        validator.accept(output_audio(3, 1))
    assert validator._data == before

    for terminal_event in (
        linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "ab"),
        linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 4),
    ):
        data, result = validator._transition(validator._data, terminal_event)
        assert result.dispatch is False
        validator._data = data


def test_state_projection_requires_responding_while_response_is_current() -> None:
    validator = ready_validator()
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.PROCESSING))
    start_response(validator)
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="responding"):
        validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.WAITING))
    assert validator._data == before
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.RESPONDING))
