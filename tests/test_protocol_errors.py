import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)
RESPONSE_ID = linklab.ResponseId(1)
OUTPUT_ID = linklab.OutputId(1)

CONNECTION_CODES = (
    linklab.ErrorCode.MALFORMED_MESSAGE,
    linklab.ErrorCode.MESSAGE_TOO_LARGE,
    linklab.ErrorCode.UNKNOWN_MESSAGE,
    linklab.ErrorCode.UNSUPPORTED_VERSION,
    linklab.ErrorCode.CAPABILITY_MISMATCH,
    linklab.ErrorCode.FORMAT_MISMATCH,
    linklab.ErrorCode.HANDSHAKE_TIMEOUT,
    linklab.ErrorCode.PROTOCOL_STATE,
    linklab.ErrorCode.ID_EXHAUSTED,
    linklab.ErrorCode.PEER_UNRESPONSIVE,
)
INPUT_CODES = (
    linklab.ErrorCode.INPUT_DISCONTINUITY,
    linklab.ErrorCode.INPUT_OVERFLOW,
    linklab.ErrorCode.INPUT_TOO_LONG,
    linklab.ErrorCode.CAPTURE_FAILED,
    linklab.ErrorCode.STT_FAILED,
    linklab.ErrorCode.PROCESSING_TIMEOUT,
)
RESPONSE_CASES = (
    (linklab.ErrorCode.GENERATION_FAILED, linklab.ResponseCancelReason.GENERATION_FAILED),
    (linklab.ErrorCode.TTS_FAILED, linklab.ResponseCancelReason.TTS_FAILED),
    (linklab.ErrorCode.PLAYBACK_FAILED, linklab.ResponseCancelReason.PLAYBACK_FAILED),
    (linklab.ErrorCode.OUTPUT_OVERFLOW, linklab.ResponseCancelReason.OVERFLOW),
    (linklab.ErrorCode.RESPONSE_CANCELLED, linklab.ResponseCancelReason.CONVERSATION_CANCELLED),
)


def ready_validator(role: linklab.EndpointRole = linklab.EndpointRole.SERVER) -> linklab.ProtocolValidator:
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
    return validator


def input_validator(role: linklab.EndpointRole = linklab.EndpointRole.SERVER) -> linklab.ProtocolValidator:
    validator = ready_validator(role)
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.ACTIVATION, 0))
    return validator


def response_validator(
    *,
    role: linklab.EndpointRole = linklab.EndpointRole.SERVER,
    end_conversation: bool = False,
    output: bool = False,
) -> linklab.ProtocolValidator:
    validator = input_validator(role)
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.NO_SPEECH))
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, end_conversation))
    if output:
        validator.accept(linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID))
    return validator


@pytest.mark.parametrize("code", CONNECTION_CODES)
def test_connection_errors_request_protocol_close_without_application_terminals(code: linklab.ErrorCode) -> None:
    validator = ready_validator()
    error = linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, code, True)

    data, result = validator._transition(validator._data, error)

    assert data.connection_state is linklab.ConnectionState.CLOSING
    assert result.close_code == 1002
    assert result.outbound == ()
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.CONVERSATION,)


@pytest.mark.parametrize(
    ("code", "end_reason"),
    (
        (linklab.ErrorCode.CONVERSATION_FAILED, linklab.ConversationEndReason.SERVER_FAILED),
        (linklab.ErrorCode.IDLE_TIMEOUT, linklab.ConversationEndReason.IDLE_TIMEOUT),
    ),
)
def test_conversation_error_closes_input_then_finalizes_and_ends(
    code: linklab.ErrorCode,
    end_reason: linklab.ConversationEndReason,
) -> None:
    validator = input_validator()
    error = linklab.ErrorEvent(linklab.ErrorScope.CONVERSATION, code, True, CONVERSATION_ID)

    data, result = validator._transition(validator._data, error)

    assert result.outbound == (
        linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.FAILED),
        linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""),
        linklab.ConversationEndedEvent(CONVERSATION_ID, end_reason),
    )
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.CONVERSATION,)
    assert data.conversation is None


@pytest.mark.parametrize("code", INPUT_CODES)
def test_every_input_error_is_recoverable_after_exact_terminal_batch(code: linklab.ErrorCode) -> None:
    validator = input_validator()
    error = linklab.ErrorEvent(linklab.ErrorScope.INPUT, code, True, CONVERSATION_ID, INPUT_ID)

    data, result = validator._transition(validator._data, error)

    assert result.outbound == (
        linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.FAILED),
        linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""),
    )
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.INPUT,)
    validator._data = data
    assert validator._response_input_ready(INPUT_ID)


@pytest.mark.parametrize(("code", "cancel_reason"), RESPONSE_CASES)
@pytest.mark.parametrize("end_conversation", (False, True))
def test_response_error_mapping_terminal_intent_and_conversation_consequence(
    code: linklab.ErrorCode,
    cancel_reason: linklab.ResponseCancelReason,
    end_conversation: bool,
) -> None:
    validator = response_validator(
        end_conversation=end_conversation,
        output=code in (linklab.ErrorCode.PLAYBACK_FAILED, linklab.ErrorCode.OUTPUT_OVERFLOW),
    )
    error = linklab.ErrorEvent(
        linklab.ErrorScope.RESPONSE,
        code,
        True,
        CONVERSATION_ID,
        response_id=RESPONSE_ID,
    )

    data, result = validator._transition(validator._data, error)

    expected: tuple[linklab.Message, ...] = (
        linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, cancel_reason),
    )
    end_reason = None
    if code is linklab.ErrorCode.PLAYBACK_FAILED:
        end_reason = linklab.ConversationEndReason.PLAYBACK_FAILED
    elif code is linklab.ErrorCode.RESPONSE_CANCELLED:
        end_reason = linklab.ConversationEndReason.CANCELLED
    elif end_conversation:
        end_reason = linklab.ConversationEndReason.SERVER_FAILED
    if end_reason is not None:
        expected = (*expected, linklab.ConversationEndedEvent(CONVERSATION_ID, end_reason))

    assert result.outbound == expected
    assert result.cancellation_targets == (linklab.ProtocolObjectKind.RESPONSE,)
    assert (data.conversation is None) is (end_reason is not None)
    if end_reason is None:
        validator._data = data
        validator.accept(
            linklab.InputStartedEvent(
                CONVERSATION_ID,
                linklab.InputId(2),
                linklab.InputStartReason.SPEECH,
                0,
            )
        )


def test_client_validates_response_error_terminal_sequence_and_rejects_wrong_reason_atomically() -> None:
    validator = response_validator(role=linklab.EndpointRole.CLIENT, end_conversation=True)
    error = linklab.ErrorEvent(
        linklab.ErrorScope.RESPONSE,
        linklab.ErrorCode.TTS_FAILED,
        True,
        CONVERSATION_ID,
        response_id=RESPONSE_ID,
    )
    validator.accept(error)
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation, match="tts_failed"):
        validator.accept(
            linklab.ResponseCancelledEvent(
                CONVERSATION_ID,
                RESPONSE_ID,
                linklab.ResponseCancelReason.GENERATION_FAILED,
            )
        )
    assert validator._data == before

    validator.accept(
        linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, linklab.ResponseCancelReason.TTS_FAILED)
    )
    validator.accept(linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.SERVER_FAILED))


def test_wrong_scoped_target_is_rejected_without_mutation() -> None:
    validator = input_validator()
    error = linklab.ErrorEvent(
        linklab.ErrorScope.INPUT,
        linklab.ErrorCode.STT_FAILED,
        True,
        CONVERSATION_ID,
        linklab.InputId(2),
    )
    before = validator._data

    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(error)
    assert validator._data == before
