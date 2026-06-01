import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
LIMITS = linklab.ConnectionLimits(max_output_audio_frames=1_600)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)
RESPONSE_ID = linklab.ResponseId(1)
OUTPUT_ID = linklab.OutputId(1)


def ready_validator() -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
    validator.transport_connected()
    validator.accept(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
    validator.accept(linklab.ServerHello(1, CAPABILITIES, PCM_16K, LIMITS))
    validator.accept(linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    return validator


def input_validator(*, closed: bool = False) -> linklab.ProtocolValidator:
    validator = ready_validator()
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.ACTIVATION, 0))
    validator.accept(linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, b"\0\0" * 2))
    if closed:
        validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 2, linklab.InputCloseReason.NO_SPEECH))
    return validator


def response_validator(*, output: bool = False) -> linklab.ProtocolValidator:
    validator = input_validator(closed=True)
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, False))
    if output:
        validator.accept(linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID))
        validator.accept(linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\0\0" * 2))
    return validator


def assert_atomic_rejection(validator: linklab.ProtocolValidator, event: linklab.Message) -> None:
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(event)
    assert validator._data == before


def test_complete_late_arrival_table_and_unknown_future_ids_is_atomic() -> None:
    # Identical terminal and cancellation events are idempotent; conflicts are fatal.
    validator = input_validator()
    abort = linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.CAPTURE_FAILED)
    validator.accept(abort)
    before = validator._data
    validator.accept(abort)
    assert validator._data == before
    assert_atomic_rejection(
        validator,
        linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.OVERFLOW),
    )

    # Endpoint-race PCM remains continuity-validated against a closed input and is discarded.
    validator = input_validator(closed=True)
    late_pcm = linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 2, False, b"\0\0")
    data, transition = validator._transition(validator._data, late_pcm)
    assert transition.dispatch is False
    validator._data = data
    assert_atomic_rejection(
        validator,
        linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 2, False, b"\0\0"),
    )

    # A cancelled output accepts one later accounting outcome, then only an identical repeat.
    validator = response_validator(output=True)
    cancel = linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, linklab.ResponseCancelReason.BARGE_IN)
    validator.accept(cancel)
    outcome = linklab.PlaybackInterruptedEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        OUTPUT_ID,
        1,
        linklab.PlaybackPosition.ESTIMATED,
        linklab.PlaybackInterruptReason.BARGE_IN,
    )
    validator.accept(outcome)
    after_outcome = validator._data
    validator.accept(outcome)
    assert validator._data == after_outcome
    assert_atomic_rejection(
        validator,
        linklab.PlaybackInterruptedEvent(
            CONVERSATION_ID,
            RESPONSE_ID,
            OUTPUT_ID,
            0,
            linklab.PlaybackPosition.EXACT,
            linklab.PlaybackInterruptReason.LOCAL_CANCEL,
        ),
    )

    # Stale text and output are fully validated but never dispatched.
    late_text = linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "late")
    data, transition = validator._transition(validator._data, late_text)
    assert transition.dispatch is False
    validator._data = data
    late_audio = linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 2, b"\0\0")
    data, transition = validator._transition(validator._data, late_audio)
    assert transition.dispatch is False
    validator._data = data

    # A transcript after response creation is always a state violation.
    validator = response_validator()
    assert_atomic_rejection(validator, linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "late"))

    # State after conversation end discards newer revisions but rejects a conflicting duplicate.
    validator = ready_validator()
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.WAITING))
    validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))
    before = validator._data
    validator.accept(linklab.StateEvent(CONVERSATION_ID, 2, linklab.CoarseState.LISTENING))
    assert validator._data == before
    assert_atomic_rejection(validator, linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.PROCESSING))

    # Terminal IDs cannot be reused, while unknown/future object IDs are also fatal.
    assert_atomic_rejection(validator, linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    validator.accept(linklab.ConversationStartedEvent(linklab.ConversationId(2), "wake_word"))
    assert_atomic_rejection(
        validator,
        linklab.InputAudioEvent(linklab.ConversationId(2), linklab.InputId(2), 0, True, b"\0\0"),
    )


def test_conversation_cancel_id_and_reason_idempotency_and_child_order() -> None:
    validator = input_validator()
    cancel = linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER)
    validator.accept(cancel)
    assert validator.tombstones == (
        linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.INPUT,
            CONVERSATION_ID,
            INPUT_ID,
            None,
            None,
            "aborted",
        ),
    )
    before = validator._data
    validator.accept(cancel)
    assert validator._data == before
    assert_atomic_rejection(
        validator,
        linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.CLIENT_FAILED),
    )
    assert_atomic_rejection(
        validator,
        linklab.ConversationCancelledEvent(linklab.ConversationId(2), linklab.ConversationCancelReason.USER),
    )

    end = linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.CANCELLED)
    validator.accept(end)
    ended = validator._data
    validator.accept(end)
    assert validator._data == ended
    assert [item.kind for item in validator.tombstones] == [
        linklab.ProtocolObjectKind.INPUT,
        linklab.ProtocolObjectKind.CONVERSATION,
    ]
    assert_atomic_rejection(
        validator,
        linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED),
    )


def test_every_protocol_object_tombstone_has_only_applicable_parent_and_object_ids() -> None:
    validator = response_validator(output=True)
    validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 2))
    validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID))
    validator.accept(linklab.PlaybackFinishedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 2))
    validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED))

    by_kind = {item.kind: item for item in validator.tombstones}
    assert by_kind == {
        linklab.ProtocolObjectKind.INPUT: linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.INPUT, CONVERSATION_ID, INPUT_ID, None, None, "closed"
        ),
        linklab.ProtocolObjectKind.OUTPUT: linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.OUTPUT, CONVERSATION_ID, None, RESPONSE_ID, OUTPUT_ID, "ended"
        ),
        linklab.ProtocolObjectKind.RESPONSE: linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.RESPONSE, CONVERSATION_ID, None, RESPONSE_ID, None, "completed"
        ),
        linklab.ProtocolObjectKind.CONVERSATION: linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.CONVERSATION, CONVERSATION_ID, None, None, None, "completed"
        ),
    }
