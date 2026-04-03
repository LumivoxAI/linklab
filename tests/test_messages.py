from typing import get_args
from dataclasses import FrozenInstanceError, fields

import pytest

import lumivox_linklab as linklab

CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(2)
RESPONSE_ID = linklab.ResponseId(3)
OUTPUT_ID = linklab.OutputId(4)
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")

CLIENT_SAMPLES: tuple[linklab.ClientMessage, ...] = (
    linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)),
    linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word", "hey lumi"),
    linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.ACTIVATION, 0),
    linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, b"\x00\x00"),
    linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.CAPTURE_FAILED),
    linklab.PlaybackFinishedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1),
    linklab.PlaybackInterruptedEvent(
        CONVERSATION_ID,
        RESPONSE_ID,
        OUTPUT_ID,
        0,
        linklab.PlaybackPosition.EXACT,
        linklab.PlaybackInterruptReason.BARGE_IN,
    ),
    linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER),
)

SERVER_SAMPLES: tuple[linklab.ServerMessage, ...] = (
    linklab.ServerHello(
        1,
        CAPABILITIES,
        PCM_16K,
        linklab.ConnectionLimits(max_output_audio_frames=1_600),
    ),
    linklab.StateEvent(CONVERSATION_ID, 1, linklab.CoarseState.LISTENING),
    linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 1, linklab.InputCloseReason.ENDPOINT),
    linklab.TranscriptUpdateEvent(CONVERSATION_ID, INPUT_ID, 1, "hello", "en"),
    linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "hello", "en"),
    linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, False),
    linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "hello"),
    linklab.ResponseTextFinalEvent(CONVERSATION_ID, RESPONSE_ID, "hello"),
    linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID),
    linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\x00\x00"),
    linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1),
    linklab.ResponseEndedEvent(CONVERSATION_ID, RESPONSE_ID),
    linklab.ResponseCancelledEvent(CONVERSATION_ID, RESPONSE_ID, linklab.ResponseCancelReason.BARGE_IN),
    linklab.ConversationEndedEvent(CONVERSATION_ID, linklab.ConversationEndReason.COMPLETED),
    linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, linklab.ErrorCode.PROTOCOL_STATE, True),
)

EXPECTED_FIELDS: dict[type[object], tuple[str, ...]] = {
    linklab.ClientHello: ("version", "capabilities", "input_format", "output_formats", "agent"),
    linklab.ServerHello: ("version", "capabilities", "output_format", "limits", "agent"),
    linklab.ConversationStartedEvent: ("conversation_id", "activation", "wake_word"),
    linklab.InputStartedEvent: ("conversation_id", "input_id", "reason", "generation", "interrupts_response_id"),
    linklab.InputAudioEvent: ("conversation_id", "input_id", "start_frame", "speech", "audio"),
    linklab.InputAbortedEvent: ("conversation_id", "input_id", "reason"),
    linklab.PlaybackFinishedEvent: ("conversation_id", "response_id", "output_id", "played_frames"),
    linklab.PlaybackInterruptedEvent: (
        "conversation_id",
        "response_id",
        "output_id",
        "played_frames",
        "position",
        "reason",
    ),
    linklab.ConversationCancelledEvent: ("conversation_id", "reason"),
    linklab.StateEvent: ("conversation_id", "revision", "state", "reason"),
    linklab.InputClosedEvent: ("conversation_id", "input_id", "accepted_end_frame", "reason"),
    linklab.TranscriptUpdateEvent: ("conversation_id", "input_id", "revision", "text", "language"),
    linklab.TranscriptFinalEvent: ("conversation_id", "input_id", "text", "language"),
    linklab.ResponseStartedEvent: ("conversation_id", "response_id", "input_id", "end_conversation"),
    linklab.ResponseTextDeltaEvent: ("conversation_id", "response_id", "sequence", "text"),
    linklab.ResponseTextFinalEvent: ("conversation_id", "response_id", "text"),
    linklab.OutputStartedEvent: ("conversation_id", "response_id", "output_id"),
    linklab.OutputAudioEvent: ("conversation_id", "response_id", "output_id", "start_frame", "audio"),
    linklab.OutputEndedEvent: ("conversation_id", "response_id", "output_id", "total_frames"),
    linklab.ResponseEndedEvent: ("conversation_id", "response_id"),
    linklab.ResponseCancelledEvent: ("conversation_id", "response_id", "reason"),
    linklab.ConversationEndedEvent: ("conversation_id", "reason"),
    linklab.ErrorEvent: ("scope", "code", "fatal", "conversation_id", "input_id", "response_id", "message"),
    linklab.ConnectionStateEvent: ("state", "reason"),
}


def test_every_message_class_is_exported_constructible_frozen_and_slotted() -> None:
    local_event = linklab.ConnectionStateEvent(linklab.ConnectionState.READY)
    samples: tuple[object, ...] = (*CLIENT_SAMPLES, *SERVER_SAMPLES, local_event)

    assert {type(sample) for sample in samples} == set(EXPECTED_FIELDS)
    for sample in samples:
        assert getattr(linklab, type(sample).__name__) is type(sample)
        assert tuple(field.name for field in fields(sample)) == EXPECTED_FIELDS[type(sample)]  # type: ignore[arg-type]
        assert not hasattr(sample, "__dict__")
        field_name = fields(sample)[0].name  # type: ignore[arg-type]
        with pytest.raises(FrozenInstanceError):
            setattr(sample, field_name, getattr(sample, field_name))


def test_message_type_discriminators_and_union_inventory() -> None:
    client_types = {type(message) for message in CLIENT_SAMPLES}
    server_types = {type(message) for message in SERVER_SAMPLES}

    assert set(get_args(linklab.ClientMessage.__value__)) == client_types
    assert set(get_args(linklab.ServerMessage.__value__)) == server_types
    assert get_args(linklab.Message.__value__) == (linklab.ClientMessage, linklab.ServerMessage)
    assert all(isinstance(message.type, str) for message in (*CLIENT_SAMPLES, *SERVER_SAMPLES))
    assert not hasattr(linklab.ConnectionStateEvent, "type")


def test_input_started_enforces_barge_in_reference_condition() -> None:
    event = linklab.InputStartedEvent(
        CONVERSATION_ID,
        INPUT_ID,
        linklab.InputStartReason.BARGE_IN,
        1,
        RESPONSE_ID,
    )
    assert event.interrupts_response_id == RESPONSE_ID

    with pytest.raises(ValueError):
        linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.BARGE_IN, 1)
    with pytest.raises(ValueError):
        linklab.InputStartedEvent(
            CONVERSATION_ID,
            INPUT_ID,
            linklab.InputStartReason.SPEECH,
            1,
            RESPONSE_ID,
        )


@pytest.mark.parametrize(
    ("scope", "code", "ids"),
    [
        (linklab.ErrorScope.CONNECTION, linklab.ErrorCode.PROTOCOL_STATE, {}),
        (
            linklab.ErrorScope.CONVERSATION,
            linklab.ErrorCode.CONVERSATION_FAILED,
            {"conversation_id": CONVERSATION_ID},
        ),
        (
            linklab.ErrorScope.INPUT,
            linklab.ErrorCode.STT_FAILED,
            {"conversation_id": CONVERSATION_ID, "input_id": INPUT_ID},
        ),
        (
            linklab.ErrorScope.RESPONSE,
            linklab.ErrorCode.TTS_FAILED,
            {"conversation_id": CONVERSATION_ID, "response_id": RESPONSE_ID},
        ),
    ],
)
def test_error_event_accepts_exact_scope_ids(
    scope: linklab.ErrorScope,
    code: linklab.ErrorCode,
    ids: dict[str, object],
) -> None:
    event = linklab.ErrorEvent(scope, code, True, message="safe", **ids)  # type: ignore[arg-type]
    assert event.scope is scope


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scope": linklab.ErrorScope.CONNECTION, "code": linklab.ErrorCode.PROTOCOL_STATE, "fatal": False},
        {
            "scope": linklab.ErrorScope.CONNECTION,
            "code": linklab.ErrorCode.PROTOCOL_STATE,
            "fatal": True,
            "conversation_id": CONVERSATION_ID,
        },
        {"scope": linklab.ErrorScope.INPUT, "code": linklab.ErrorCode.STT_FAILED, "fatal": True},
        {
            "scope": linklab.ErrorScope.RESPONSE,
            "code": linklab.ErrorCode.TTS_FAILED,
            "fatal": True,
            "conversation_id": CONVERSATION_ID,
            "input_id": INPUT_ID,
            "response_id": RESPONSE_ID,
        },
        {
            "scope": linklab.ErrorScope.INPUT,
            "code": linklab.ErrorCode.TTS_FAILED,
            "fatal": True,
            "conversation_id": CONVERSATION_ID,
            "input_id": INPUT_ID,
        },
    ],
)
def test_error_event_rejects_invalid_scope_combinations(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        linklab.ErrorEvent(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("event_type", [linklab.InputAudioEvent, linklab.OutputAudioEvent])
def test_pcm_is_copied_to_immutable_bytes(event_type: type[object]) -> None:
    source = bytearray(b"\x01\x02")
    event: linklab.InputAudioEvent | linklab.OutputAudioEvent
    if event_type is linklab.InputAudioEvent:
        event = linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, source)  # type: ignore[arg-type]
    else:
        event = linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, source)  # type: ignore[arg-type]
    source[:] = b"\x03\x04"

    assert event.audio == b"\x01\x02"
    assert type(event.audio) is bytes


def test_pcm_rejects_empty_misaligned_and_noncontiguous_buffers() -> None:
    with pytest.raises(ValueError):
        linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, b"")
    with pytest.raises(ValueError):
        linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, b"\x00")
    with pytest.raises(ValueError):
        linklab.InputAudioEvent(
            CONVERSATION_ID,
            INPUT_ID,
            0,
            True,
            memoryview(bytearray(4))[::2],  # type: ignore[arg-type]
        )


def test_pcm_enforces_global_direction_limits() -> None:
    linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, bytes(1_600 * 2))
    linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, bytes(4_800 * 2))
    with pytest.raises(ValueError):
        linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, bytes(1_601 * 2))
    with pytest.raises(ValueError):
        linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, bytes(4_801 * 2))


def test_hello_enforces_capabilities_formats_and_selected_limits() -> None:
    with pytest.raises(ValueError):
        linklab.ClientHello(1, tuple(reversed(CAPABILITIES)), PCM_16K, (PCM_16K,))
    with pytest.raises(ValueError):
        linklab.ClientHello(1, ("barge_in",), PCM_16K, (PCM_16K,))
    with pytest.raises(ValueError):
        linklab.ClientHello(
            1,
            CAPABILITIES,
            PCM_16K,
            (PCM_16K, linklab.AudioFormat("pcm_s16le", 24_000, 1)),
        )
    with pytest.raises(ValueError):
        linklab.ServerHello(1, CAPABILITIES, PCM_16K, linklab.ConnectionLimits())


def test_local_scalar_invariants_reject_invalid_values() -> None:
    with pytest.raises(TypeError):
        linklab.StateEvent(CONVERSATION_ID, True, linklab.CoarseState.WAITING)
    with pytest.raises(ValueError):
        linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "")
    with pytest.raises(ValueError):
        linklab.ConversationStartedEvent(CONVERSATION_ID, "manual")  # type: ignore[arg-type]


def test_empty_final_transcript_is_schema_valid_before_lifecycle_validation() -> None:
    event = linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, "")
    assert event.text == ""


def test_lifecycle_dependent_zero_output_total_is_schema_valid() -> None:
    event = linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0)
    assert event.total_frames == 0


@pytest.mark.parametrize("reason", [None, "", "x" * 1_000])
def test_local_connection_reason_has_no_wire_diagnostic_limit(reason: str | None) -> None:
    assert linklab.ConnectionStateEvent(linklab.ConnectionState.DISCONNECTED, reason).reason == reason
