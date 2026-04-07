from typing import cast

import pytest
import msgpack  # type: ignore[import-untyped]

import lumivox_linklab as linklab

CAPABILITIES = ["barge_in", "playback_accounting", "speech_spans"]
FORMAT_16K = {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1}
LIMITS = {
    "max_message_bytes": 262_144,
    "max_input_audio_frames": 1_600,
    "max_output_audio_frames": 1_600,
    "max_text_bytes": 16_384,
    "max_input_frames": 1_920_000,
    "idle_timeout_ms": 120_000,
}


def _pack(value: object) -> bytes:
    return cast(bytes, msgpack.packb(value, use_bin_type=True))


CLIENT_MESSAGES: tuple[tuple[dict[str, object], type[object]], ...] = (
    (
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "input_format": FORMAT_16K,
            "output_formats": [FORMAT_16K],
        },
        linklab.ClientHello,
    ),
    (
        {"type": "conversation.start", "conversation_id": 1, "activation": "wake_word"},
        linklab.ConversationStartedEvent,
    ),
    (
        {"type": "input.start", "conversation_id": 1, "input_id": 1, "reason": "activation", "generation": 0},
        linklab.InputStartedEvent,
    ),
    (
        {
            "type": "input.audio",
            "conversation_id": 1,
            "input_id": 1,
            "start_frame": 0,
            "speech": True,
            "audio": b"\0\0",
        },
        linklab.InputAudioEvent,
    ),
    (
        {"type": "input.abort", "conversation_id": 1, "input_id": 1, "reason": "capture_failed"},
        linklab.InputAbortedEvent,
    ),
    (
        {
            "type": "playback.finished",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "played_frames": 1,
        },
        linklab.PlaybackFinishedEvent,
    ),
    (
        {
            "type": "playback.interrupted",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "played_frames": 0,
            "position": "exact",
            "reason": "barge_in",
        },
        linklab.PlaybackInterruptedEvent,
    ),
    (
        {"type": "conversation.cancel", "conversation_id": 1, "reason": "user"},
        linklab.ConversationCancelledEvent,
    ),
)

SERVER_MESSAGES: tuple[tuple[dict[str, object], type[object]], ...] = (
    (
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "output_format": FORMAT_16K,
            "limits": LIMITS,
        },
        linklab.ServerHello,
    ),
    ({"type": "state", "conversation_id": 1, "revision": 1, "state": "listening"}, linklab.StateEvent),
    (
        {
            "type": "input.closed",
            "conversation_id": 1,
            "input_id": 1,
            "accepted_end_frame": 1,
            "reason": "endpoint",
        },
        linklab.InputClosedEvent,
    ),
    (
        {"type": "transcript.update", "conversation_id": 1, "input_id": 1, "revision": 1, "text": "hi"},
        linklab.TranscriptUpdateEvent,
    ),
    (
        {"type": "transcript.final", "conversation_id": 1, "input_id": 1, "text": "hi"},
        linklab.TranscriptFinalEvent,
    ),
    (
        {
            "type": "response.start",
            "conversation_id": 1,
            "response_id": 1,
            "input_id": 1,
            "end_conversation": False,
        },
        linklab.ResponseStartedEvent,
    ),
    (
        {"type": "response.text.delta", "conversation_id": 1, "response_id": 1, "sequence": 0, "text": "hi"},
        linklab.ResponseTextDeltaEvent,
    ),
    (
        {"type": "response.text.final", "conversation_id": 1, "response_id": 1, "text": "hi"},
        linklab.ResponseTextFinalEvent,
    ),
    (
        {"type": "output.start", "conversation_id": 1, "response_id": 1, "output_id": 1},
        linklab.OutputStartedEvent,
    ),
    (
        {
            "type": "output.audio",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "start_frame": 0,
            "audio": b"\0\0",
        },
        linklab.OutputAudioEvent,
    ),
    (
        {"type": "output.end", "conversation_id": 1, "response_id": 1, "output_id": 1, "total_frames": 1},
        linklab.OutputEndedEvent,
    ),
    ({"type": "response.end", "conversation_id": 1, "response_id": 1}, linklab.ResponseEndedEvent),
    (
        {"type": "response.cancelled", "conversation_id": 1, "response_id": 1, "reason": "barge_in"},
        linklab.ResponseCancelledEvent,
    ),
    (
        {"type": "conversation.end", "conversation_id": 1, "reason": "completed"},
        linklab.ConversationEndedEvent,
    ),
    (
        {"type": "error", "scope": "connection", "code": "protocol_state", "fatal": True},
        linklab.ErrorEvent,
    ),
)


@pytest.mark.parametrize(("wire", "expected_type"), CLIENT_MESSAGES)
def test_decodes_every_client_message(wire: dict[str, object], expected_type: type[object]) -> None:
    message = linklab.decode_message(
        _pack(wire), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
    )
    assert type(message) is expected_type


@pytest.mark.parametrize(("wire", "expected_type"), SERVER_MESSAGES)
def test_decodes_every_server_message(wire: dict[str, object], expected_type: type[object]) -> None:
    message = linklab.decode_message(
        _pack(wire), direction=linklab.MessageDirection.SERVER_TO_CLIENT, limits=linklab.ConnectionLimits()
    )
    assert type(message) is expected_type


@pytest.mark.parametrize(
    ("wire", "direction"),
    [
        *[(wire, linklab.MessageDirection.CLIENT_TO_SERVER) for wire, _ in CLIENT_MESSAGES],
        *[(wire, linklab.MessageDirection.SERVER_TO_CLIENT) for wire, _ in SERVER_MESSAGES],
    ],
)
def test_each_message_rejects_a_missing_required_field(
    wire: dict[str, object], direction: linklab.MessageDirection
) -> None:
    field = next(name for name in wire if name != "type")
    invalid = {name: value for name, value in wire.items() if name != field}
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(_pack(invalid), direction=direction, limits=linklab.ConnectionLimits())


def test_optional_and_conditional_fields_are_schema_checked() -> None:
    barge_in = {
        "type": "input.start",
        "conversation_id": 1,
        "input_id": 1,
        "reason": "barge_in",
        "generation": 0,
        "interrupts_response_id": 2,
        "ignored": "allowed",
    }
    result = linklab.decode_message(
        _pack(barge_in), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
    )
    assert isinstance(result, linklab.InputStartedEvent)
    assert result.interrupts_response_id == 2

    del barge_in["interrupts_response_id"]
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack(barge_in), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
        )


def test_rejects_wrong_direction_unknown_type_and_non_ascii_field_names() -> None:
    limits = linklab.ConnectionLimits()
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack(CLIENT_MESSAGES[1][0]), direction=linklab.MessageDirection.SERVER_TO_CLIENT, limits=limits
        )
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack({"type": "future"}), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=limits
        )
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack({**CLIENT_MESSAGES[1][0], "nested": {"naive": {"caf\N{LATIN SMALL LETTER E WITH ACUTE}": 1}}}),
            direction=linklab.MessageDirection.CLIENT_TO_SERVER,
            limits=limits,
        )


@pytest.mark.parametrize("value", [True, 0, 2**32])
def test_ids_reject_booleans_and_out_of_range_values(value: object) -> None:
    wire = {**CLIENT_MESSAGES[1][0], "conversation_id": value}
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack(wire), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
        )


def test_accepts_a_noncanonical_integer_width() -> None:
    packed = _pack(CLIENT_MESSAGES[1][0])
    packed = packed.replace(b"\xafconversation_id\x01", b"\xafconversation_id\xce\x00\x00\x00\x01")
    result = linklab.decode_message(
        packed, direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
    )
    assert isinstance(result, linklab.ConversationStartedEvent)
    assert result.conversation_id == 1


def test_enforces_negotiated_message_text_and_audio_limits() -> None:
    limits = linklab.ConnectionLimits(max_message_bytes=16_384, max_input_audio_frames=160, max_text_bytes=1_024)
    input_audio = {**CLIENT_MESSAGES[3][0], "audio": bytes(161 * 2)}
    text = {**SERVER_MESSAGES[3][0], "text": "x" * 1_025}

    with pytest.raises(linklab.CodecError):
        linklab.decode_message(_pack(input_audio), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=limits)
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(_pack(text), direction=linklab.MessageDirection.SERVER_TO_CLIENT, limits=limits)
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack({**CLIENT_MESSAGES[1][0], "padding": "x" * 16_384}),
            direction=linklab.MessageDirection.CLIENT_TO_SERVER,
            limits=limits,
        )


def test_server_hello_validates_embedded_limits_against_output_format() -> None:
    wire = SERVER_MESSAGES[0][0].copy()
    wire["limits"] = {**LIMITS, "max_output_audio_frames": 159}
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack(wire),
            direction=linklab.MessageDirection.SERVER_TO_CLIENT,
            limits=linklab.ConnectionLimits(max_output_audio_frames=160),
        )


def test_unknown_fields_are_ignored_only_after_structural_validation() -> None:
    wire = {**CLIENT_MESSAGES[1][0], "unknown": bytes(16_384)}
    result = linklab.decode_message(
        _pack(wire), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
    )
    assert isinstance(result, linklab.ConversationStartedEvent)

    wire["unknown"] = bytes(16_385)
    with pytest.raises(linklab.CodecError):
        linklab.decode_message(
            _pack(wire), direction=linklab.MessageDirection.CLIENT_TO_SERVER, limits=linklab.ConnectionLimits()
        )


def test_public_decoder_validates_its_control_arguments() -> None:
    packed = _pack(CLIENT_MESSAGES[1][0])
    with pytest.raises(TypeError):
        linklab.decode_message(packed, direction="client_to_server", limits=linklab.ConnectionLimits())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        linklab.decode_message(
            packed,
            direction=linklab.MessageDirection.CLIENT_TO_SERVER,
            limits=object(),  # type: ignore[arg-type]
        )
