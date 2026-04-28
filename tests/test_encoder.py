import json
from typing import cast
from pathlib import Path

import pytest
import msgpack  # type: ignore[import-untyped]

import lumivox_linklab as linklab
from lumivox_linklab._codec import _encode_message_with_limits
from lumivox_linklab._schema import _SCHEMAS

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "fixtures" / "protocol-v1.json"


def _from_json(value: object) -> object:
    if type(value) is dict:
        if set(value) == {"$binary_hex"}:
            encoded = value["$binary_hex"]
            assert type(encoded) is str
            return bytes.fromhex(encoded)
        return {key: _from_json(item) for key, item in value.items()}
    if type(value) is list:
        return [_from_json(item) for item in value]
    return value


def test_golden_manifest_covers_every_message_and_round_trips() -> None:
    manifest = json.loads(MANIFEST.read_text())
    fixtures = manifest["fixtures"]
    assert manifest["protocol"] == "lumivox.voice.v1"
    assert len(fixtures) == 24
    assert len({fixture["name"] for fixture in fixtures}) == len(fixtures)

    observed_types: set[type[object]] = set()
    for fixture in fixtures:
        direction = linklab.MessageDirection(fixture["direction"])
        expected = _from_json(fixture["message"])
        wire = bytes.fromhex(fixture["wire_hex"])
        assert msgpack.unpackb(wire, raw=False) == expected

        message = linklab.decode_message(wire, direction=direction, limits=linklab.ConnectionLimits())
        observed_types.add(type(message))
        assert linklab.encode_message(message) == wire

    assert observed_types == {
        linklab.ClientHello,
        linklab.ServerHello,
        linklab.ConversationStartedEvent,
        linklab.InputStartedEvent,
        linklab.InputAudioEvent,
        linklab.InputAbortedEvent,
        linklab.PlaybackFinishedEvent,
        linklab.PlaybackInterruptedEvent,
        linklab.ConversationCancelledEvent,
        linklab.StateEvent,
        linklab.InputClosedEvent,
        linklab.TranscriptUpdateEvent,
        linklab.TranscriptFinalEvent,
        linklab.ResponseStartedEvent,
        linklab.ResponseTextDeltaEvent,
        linklab.ResponseTextFinalEvent,
        linklab.OutputStartedEvent,
        linklab.OutputAudioEvent,
        linklab.OutputEndedEvent,
        linklab.ResponseEndedEvent,
        linklab.ResponseCancelledEvent,
        linklab.ConversationEndedEvent,
        linklab.ErrorEvent,
    }


def test_schema_registry_has_one_bidirectional_mapping_per_message_model() -> None:
    wire_keys = {(schema.direction, schema.wire_type) for schema in _SCHEMAS}
    model_types = {schema.model_type for schema in _SCHEMAS}

    assert len(_SCHEMAS) == 23
    assert len(wire_keys) == len(_SCHEMAS)
    assert len(model_types) == len(_SCHEMAS)


def test_encoder_sorts_every_map_by_utf8_key_bytes() -> None:
    hello = linklab.ServerHello(
        1,
        ("barge_in", "playback_accounting", "speech_spans"),
        linklab.AudioFormat("pcm_s16le", 16_000, 1),
        linklab.ConnectionLimits(max_output_audio_frames=1_600),
        "agent",
    )
    pairs = msgpack.unpackb(
        linklab.encode_message(hello),
        raw=False,
        object_pairs_hook=lambda value: value,
    )
    root_keys = [key for key, _ in pairs]
    nested = dict(pairs)
    output_keys = [key for key, _ in nested["output_format"]]
    limit_keys = [key for key, _ in nested["limits"]]

    assert root_keys == sorted(root_keys, key=str.encode)
    assert output_keys == sorted(output_keys, key=str.encode)
    assert limit_keys == sorted(limit_keys, key=str.encode)


def test_encoder_uses_shortest_standard_representations() -> None:
    event = linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word")
    encoded = linklab.encode_message(event)

    assert encoded.hex() == (
        "83aa61637469766174696f6ea977616b655f776f7264af636f6e766572736174696f6e5f696401"
        "a474797065b2636f6e766572736174696f6e2e7374617274"
    )


def test_semantically_equal_messages_have_identical_bytes() -> None:
    positional = linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.LISTENING, "wake")
    keyword = linklab.StateEvent(
        reason="wake",
        state=linklab.CoarseState.LISTENING,
        revision=1,
        conversation_id=linklab.ConversationId(1),
    )

    assert positional == keyword
    assert linklab.encode_message(positional) == linklab.encode_message(keyword)


def test_negotiated_encoder_accounts_for_complete_message_overhead() -> None:
    limits = linklab.ConnectionLimits(max_message_bytes=16_384, max_text_bytes=65_536)
    exact: linklab.ResponseTextFinalEvent | None = None
    for size in range(16_000, 16_385):
        candidate = linklab.ResponseTextFinalEvent(linklab.ConversationId(1), linklab.ResponseId(1), "x" * size)
        if len(linklab.encode_message(candidate)) == limits.max_message_bytes:
            exact = candidate
            break
    assert exact is not None
    assert len(_encode_message_with_limits(exact, limits)) == limits.max_message_bytes

    one_over = linklab.ResponseTextFinalEvent(exact.conversation_id, exact.response_id, exact.text + "x")
    with pytest.raises(linklab.CodecError, match="envelope"):
        _encode_message_with_limits(one_over, limits)


def test_negotiated_encoder_enforces_text_and_audio_limits() -> None:
    limits = linklab.ConnectionLimits(max_input_audio_frames=160, max_output_audio_frames=160, max_text_bytes=1_024)
    input_exact = linklab.InputAudioEvent(linklab.ConversationId(1), linklab.InputId(1), 0, True, bytes(160 * 2))
    output_exact = linklab.OutputAudioEvent(
        linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 0, bytes(160 * 2)
    )
    text_exact = linklab.ResponseTextFinalEvent(linklab.ConversationId(1), linklab.ResponseId(1), "x" * 1_024)

    assert _encode_message_with_limits(input_exact, limits)
    assert _encode_message_with_limits(output_exact, limits)
    assert _encode_message_with_limits(text_exact, limits)
    with pytest.raises(linklab.CodecError, match="input audio"):
        _encode_message_with_limits(
            linklab.InputAudioEvent(linklab.ConversationId(1), linklab.InputId(1), 0, True, bytes(161 * 2)),
            limits,
        )
    with pytest.raises(linklab.CodecError, match="output audio"):
        _encode_message_with_limits(
            linklab.OutputAudioEvent(
                linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 0, bytes(161 * 2)
            ),
            limits,
        )
    with pytest.raises(linklab.CodecError, match="text"):
        _encode_message_with_limits(
            linklab.ResponseTextFinalEvent(linklab.ConversationId(1), linklab.ResponseId(1), "x" * 1_025),
            limits,
        )


def test_encoder_rejects_non_message_and_invalid_limits() -> None:
    with pytest.raises(linklab.CodecError):
        linklab.encode_message(cast(linklab.Message, object()))
    event = linklab.ConversationEndedEvent(linklab.ConversationId(1), linklab.ConversationEndReason.COMPLETED)
    with pytest.raises(TypeError):
        _encode_message_with_limits(event, cast(linklab.ConnectionLimits, object()))
