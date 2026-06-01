from enum import StrEnum
from types import UnionType
from typing import Any, Literal, get_args, get_type_hints
from dataclasses import MISSING, fields

import pytest
import msgpack  # type: ignore[import-untyped]

import lumivox_linklab as linklab
from lumivox_linklab._codec import _decode_primitive_message
from lumivox_linklab._schema import _SCHEMAS
from lumivox_linklab._server import _copy_output_pcm
from lumivox_linklab._client_ingress import _ClientAudioIngress

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
C2S = linklab.MessageDirection.CLIENT_TO_SERVER
S2C = linklab.MessageDirection.SERVER_TO_CLIENT

# Every optional field is present here so exact-type checks cover it too.
SCHEMA_CASES: tuple[tuple[linklab.MessageDirection, type[object], dict[str, object], frozenset[str]], ...] = (
    (
        C2S,
        linklab.ClientHello,
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "input_format": FORMAT_16K,
            "output_formats": [FORMAT_16K],
            "agent": "agent",
        },
        frozenset({"agent"}),
    ),
    (
        C2S,
        linklab.ConversationStartedEvent,
        {"type": "conversation.start", "conversation_id": 1, "activation": "wake_word", "wake_word": "lumi"},
        frozenset({"wake_word"}),
    ),
    (
        C2S,
        linklab.InputStartedEvent,
        {
            "type": "input.start",
            "conversation_id": 1,
            "input_id": 1,
            "reason": "barge_in",
            "generation": 0,
            "interrupts_response_id": 1,
        },
        frozenset({"interrupts_response_id"}),
    ),
    (
        C2S,
        linklab.InputAudioEvent,
        {
            "type": "input.audio",
            "conversation_id": 1,
            "input_id": 1,
            "start_frame": 0,
            "speech": True,
            "audio": b"\0\0",
        },
        frozenset(),
    ),
    (
        C2S,
        linklab.InputAbortedEvent,
        {"type": "input.abort", "conversation_id": 1, "input_id": 1, "reason": "capture_failed"},
        frozenset(),
    ),
    (
        C2S,
        linklab.PlaybackFinishedEvent,
        {"type": "playback.finished", "conversation_id": 1, "response_id": 1, "output_id": 1, "played_frames": 0},
        frozenset(),
    ),
    (
        C2S,
        linklab.PlaybackInterruptedEvent,
        {
            "type": "playback.interrupted",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "played_frames": 0,
            "position": "exact",
            "reason": "barge_in",
        },
        frozenset(),
    ),
    (
        C2S,
        linklab.ConversationCancelledEvent,
        {"type": "conversation.cancel", "conversation_id": 1, "reason": "user"},
        frozenset(),
    ),
    (
        S2C,
        linklab.ServerHello,
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "output_format": FORMAT_16K,
            "limits": LIMITS,
            "agent": "agent",
        },
        frozenset({"agent"}),
    ),
    (
        S2C,
        linklab.StateEvent,
        {"type": "state", "conversation_id": 1, "revision": 1, "state": "listening", "reason": "speech"},
        frozenset({"reason"}),
    ),
    (
        S2C,
        linklab.InputClosedEvent,
        {"type": "input.closed", "conversation_id": 1, "input_id": 1, "accepted_end_frame": 0, "reason": "endpoint"},
        frozenset(),
    ),
    (
        S2C,
        linklab.TranscriptUpdateEvent,
        {
            "type": "transcript.update",
            "conversation_id": 1,
            "input_id": 1,
            "revision": 1,
            "text": "x",
            "language": "en",
        },
        frozenset({"language"}),
    ),
    (
        S2C,
        linklab.TranscriptFinalEvent,
        {"type": "transcript.final", "conversation_id": 1, "input_id": 1, "text": "", "language": "en"},
        frozenset({"language"}),
    ),
    (
        S2C,
        linklab.ResponseStartedEvent,
        {"type": "response.start", "conversation_id": 1, "response_id": 1, "input_id": 1, "end_conversation": False},
        frozenset(),
    ),
    (
        S2C,
        linklab.ResponseTextDeltaEvent,
        {"type": "response.text.delta", "conversation_id": 1, "response_id": 1, "sequence": 0, "text": "x"},
        frozenset(),
    ),
    (
        S2C,
        linklab.ResponseTextFinalEvent,
        {"type": "response.text.final", "conversation_id": 1, "response_id": 1, "text": "x"},
        frozenset(),
    ),
    (
        S2C,
        linklab.OutputStartedEvent,
        {"type": "output.start", "conversation_id": 1, "response_id": 1, "output_id": 1},
        frozenset(),
    ),
    (
        S2C,
        linklab.OutputAudioEvent,
        {
            "type": "output.audio",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "start_frame": 0,
            "audio": b"\0\0",
        },
        frozenset(),
    ),
    (
        S2C,
        linklab.OutputEndedEvent,
        {"type": "output.end", "conversation_id": 1, "response_id": 1, "output_id": 1, "total_frames": 0},
        frozenset(),
    ),
    (S2C, linklab.ResponseEndedEvent, {"type": "response.end", "conversation_id": 1, "response_id": 1}, frozenset()),
    (
        S2C,
        linklab.ResponseCancelledEvent,
        {"type": "response.cancelled", "conversation_id": 1, "response_id": 1, "reason": "barge_in"},
        frozenset(),
    ),
    (
        S2C,
        linklab.ConversationEndedEvent,
        {"type": "conversation.end", "conversation_id": 1, "reason": "completed"},
        frozenset(),
    ),
    (
        S2C,
        linklab.ErrorEvent,
        {
            "type": "error",
            "scope": "response",
            "code": "tts_failed",
            "fatal": True,
            "conversation_id": 1,
            "response_id": 1,
            "message": "safe",
        },
        frozenset({"conversation_id", "input_id", "response_id", "message"}),
    ),
)


def _pack(value: object) -> bytes:
    return msgpack.packb(value, use_bin_type=True)  # type: ignore[no-any-return]


def _decode(
    wire: dict[str, object],
    direction: linklab.MessageDirection,
    limits: linklab.ConnectionLimits | None = None,
) -> linklab.Message:
    return linklab.decode_message(_pack(wire), direction=direction, limits=limits or linklab.ConnectionLimits())


def _wrong_type(value: object) -> object:
    if type(value) is bool:
        return 0
    if type(value) is int:
        return True
    if type(value) is str:
        return b"x"
    if type(value) is bytes:
        return "x"
    if type(value) is list:
        return {"not": "an array"}
    if type(value) is dict:
        return list(value.items())
    raise AssertionError(f"unhandled schema value {value!r}")


def test_every_schema_field_exact_type_range_presence_and_condition() -> None:
    expected_inventory = {(direction, wire["type"], model) for direction, model, wire, _ in SCHEMA_CASES}
    assert {(schema.direction, schema.wire_type, schema.model_type) for schema in _SCHEMAS} == expected_inventory

    for direction, model, wire, optional in SCHEMA_CASES:
        assert type(_decode(wire, direction)) is model
        schema = next(item for item in _SCHEMAS if item.direction is direction and item.model_type is model)
        assert (
            tuple(field.name for field in schema.fields) == tuple(name for name in wire if name != "type")
            or model is linklab.ErrorEvent
        )
        assert {field.name for field in schema.fields if field.optional} == optional
        for name, value in wire.items():
            if name == "type":
                continue
            invalid = {**wire, name: _wrong_type(value)}
            with pytest.raises(linklab.CodecError, match=name):
                _decode(invalid, direction)
            if name not in optional:
                missing = wire.copy()
                del missing[name]
                with pytest.raises(linklab.CodecError, match=name):
                    _decode(missing, direction)
        for name in optional & wire.keys():
            omitted = wire.copy()
            del omitted[name]
            if model is linklab.InputStartedEvent:
                omitted["reason"] = "speech"
            elif model is linklab.ErrorEvent:
                continue
            assert type(_decode(omitted, direction)) is model

    input_start = SCHEMA_CASES[2][2]
    with pytest.raises(linklab.CodecError):
        _decode({name: value for name, value in input_start.items() if name != "interrupts_response_id"}, C2S)
    with pytest.raises(linklab.CodecError):
        _decode({**input_start, "reason": "speech"}, C2S)


@pytest.mark.parametrize("wire_type", ["input.audio", "output.audio"])
@pytest.mark.parametrize("value", [[], "\0\0", None])
def test_pcm_wire_fields_accept_only_binary(wire_type: str, value: object) -> None:
    direction, _, wire, _ = next(case for case in SCHEMA_CASES if case[2]["type"] == wire_type)
    with pytest.raises(linklab.CodecError, match="audio"):
        _decode({**wire, "audio": value}, direction)


def test_every_integer_wire_field_rejects_bool() -> None:
    checked: set[tuple[str, str]] = set()
    for direction, _, wire, _ in SCHEMA_CASES:
        for name, value in wire.items():
            if type(value) is int:
                checked.add((str(wire["type"]), name))
                with pytest.raises(linklab.CodecError, match=name):
                    _decode({**wire, name: True}, direction)
        for parent_name in ("input_format", "output_format", "limits"):
            nested = wire.get(parent_name)
            if type(nested) is dict:
                for name, value in nested.items():
                    if type(value) is int:
                        changed = nested.copy()
                        changed[name] = True
                        with pytest.raises(linklab.CodecError, match=name):
                            _decode({**wire, parent_name: changed}, direction)
    assert checked == {
        (str(wire["type"]), name)
        for _, _, wire, _ in SCHEMA_CASES
        for name, value in wire.items()
        if type(value) is int
    }


def test_every_primitive_alias_accepts_bounds_and_rejects_adjacent_bool_and_empty_cases() -> None:
    rejected: object
    conversation = SCHEMA_CASES[1][2]
    for accepted in (1, 4_294_967_295):
        assert _decode({**conversation, "conversation_id": accepted}, C2S).conversation_id == accepted  # type: ignore[union-attr]
    for rejected in (True, 0, 4_294_967_296):
        with pytest.raises(linklab.CodecError):
            _decode({**conversation, "conversation_id": rejected}, C2S)

    for case_index, field_name, minimum in ((9, "revision", 1), (14, "sequence", 0), (5, "played_frames", 0)):
        direction, _, wire, _ = SCHEMA_CASES[case_index]
        for accepted in (minimum, 4_294_967_295 if field_name != "played_frames" else 9_223_372_036_854_775_807):
            assert _decode({**wire, field_name: accepted}, direction)
        for rejected in (
            True,
            minimum - 1,
            4_294_967_296 if field_name != "played_frames" else 9_223_372_036_854_775_808,
        ):
            with pytest.raises(linklab.CodecError):
                _decode({**wire, field_name: rejected}, direction)

    update = SCHEMA_CASES[11][2]
    final = SCHEMA_CASES[12][2]
    text_limits = linklab.ConnectionLimits(max_text_bytes=65_536)
    assert _decode({**update, "text": "x" * 65_536}, S2C, text_limits)
    assert _decode({**final, "text": ""}, S2C)
    for wire in (update, SCHEMA_CASES[14][2], SCHEMA_CASES[15][2]):
        with pytest.raises(linklab.CodecError):
            _decode({**wire, "text": ""}, S2C)
    with pytest.raises(linklab.CodecError):
        _decode({**final, "text": "x" * 65_537}, S2C, text_limits)

    audio = SCHEMA_CASES[3][2]
    assert _decode({**audio, "audio": bytes(3_200)}, C2S)
    for rejected in (b"", bytes(3_202)):
        with pytest.raises(linklab.CodecError):
            _decode({**audio, "audio": rejected}, C2S)


def test_short_identifier_reason_and_type_utf8_byte_limit() -> None:
    hello = SCHEMA_CASES[0][2]
    state = SCHEMA_CASES[9][2]
    assert _decode({**hello, "agent": "é" * 32}, C2S)
    assert _decode({**state, "reason": "é" * 32}, S2C)
    for wire, direction, field in ((hello, C2S, "agent"), (state, S2C, "reason")):
        with pytest.raises(linklab.CodecError):
            _decode({**wire, field: "é" * 32 + "x"}, direction)

    with pytest.raises(linklab.CodecError, match="unknown"):
        _decode({"type": "x" * 64}, C2S)
    with pytest.raises(linklab.CodecError, match="64-byte"):
        _decode({"type": "x" * 65}, C2S)

    for case_index, field in (
        (0, "agent"),
        (1, "wake_word"),
        (8, "agent"),
        (9, "reason"),
        (11, "language"),
        (12, "language"),
    ):
        direction, _, wire, _ = SCHEMA_CASES[case_index]
        assert _decode({**wire, field: "é" * 32}, direction)
        with pytest.raises(linklab.CodecError):
            _decode({**wire, field: "é" * 32 + "x"}, direction)
    assert _decode({**hello, "capabilities": [*CAPABILITIES, "z" * 64]}, C2S)
    with pytest.raises(linklab.CodecError):
        _decode({**hello, "capabilities": [*CAPABILITIES, "z" * 65]}, C2S)


def test_hello_agent_format_counts_and_complete_server_limits() -> None:
    client = SCHEMA_CASES[0][2]
    server = SCHEMA_CASES[8][2]
    for size in (0, 4):
        with pytest.raises(linklab.CodecError):
            _decode({**client, "output_formats": [FORMAT_16K] * size}, C2S)
    for wire, direction in ((client, C2S), (server, S2C)):
        assert _decode({**wire, "agent": "é" * 32}, direction)
        with pytest.raises(linklab.CodecError):
            _decode({**wire, "agent": "é" * 32 + "x"}, direction)
    for name in LIMITS:
        incomplete = LIMITS.copy()
        del incomplete[name]
        with pytest.raises(linklab.CodecError, match=name):
            _decode({**server, "limits": incomplete}, S2C)


def test_all_wire_enum_fields_reject_unknown_values() -> None:
    checked: set[type[StrEnum]] = set()
    for direction, model, wire, _ in SCHEMA_CASES:
        hints = get_type_hints(model)
        for name, annotation in hints.items():
            if isinstance(annotation, type) and issubclass(annotation, StrEnum):
                checked.add(annotation)
                with pytest.raises(linklab.CodecError, match=name):
                    _decode({**wire, name: "future_value"}, direction)
    assert checked == {
        linklab.InputStartReason,
        linklab.InputAbortReason,
        linklab.PlaybackPosition,
        linklab.PlaybackInterruptReason,
        linklab.ConversationCancelReason,
        linklab.CoarseState,
        linklab.InputCloseReason,
        linklab.ResponseCancelReason,
        linklab.ConversationEndReason,
        linklab.ErrorScope,
        linklab.ErrorCode,
    }


ERROR_CODES = {
    linklab.ErrorScope.CONNECTION: tuple(linklab.ErrorCode)[:10],
    linklab.ErrorScope.CONVERSATION: tuple(linklab.ErrorCode)[10:12],
    linklab.ErrorScope.INPUT: tuple(linklab.ErrorCode)[12:18],
    linklab.ErrorScope.RESPONSE: tuple(linklab.ErrorCode)[18:],
}


def test_error_diagnostic_boundaries_and_every_code_scope_compatibility() -> None:
    ids: dict[linklab.ErrorScope, dict[str, int]] = {
        linklab.ErrorScope.CONNECTION: {},
        linklab.ErrorScope.CONVERSATION: {"conversation_id": 1},
        linklab.ErrorScope.INPUT: {"conversation_id": 1, "input_id": 1},
        linklab.ErrorScope.RESPONSE: {"conversation_id": 1, "response_id": 1},
    }
    for scope, accepted_codes in ERROR_CODES.items():
        for code in linklab.ErrorCode:
            kwargs: dict[str, Any] = {"scope": scope, "code": code, "fatal": True, **ids[scope]}
            if code in accepted_codes:
                assert linklab.ErrorEvent(**kwargs).code is code
            else:
                with pytest.raises(ValueError, match="code"):
                    linklab.ErrorEvent(**kwargs)

    for message in ("x", "é" * 256):
        assert (
            linklab.ErrorEvent(
                linklab.ErrorScope.CONNECTION, linklab.ErrorCode.PROTOCOL_STATE, True, message=message
            ).message
            == message
        )
    for message in ("", "é" * 256 + "x"):
        with pytest.raises(ValueError, match="message"):
            linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, linklab.ErrorCode.PROTOCOL_STATE, True, message=message)


def test_decoder_configuration_rejects_extensions_duplicates_and_limits_before_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_unpackb = msgpack.unpackb
    observed: dict[str, object] = {}

    def inspect_unpackb(data: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return real_unpackb(data, **kwargs)

    monkeypatch.setattr(msgpack, "unpackb", inspect_unpackb)
    assert _decode_primitive_message(_pack({"type": "future"}))
    assert observed == {
        "raw": False,
        "use_list": True,
        "strict_map_key": True,
        "object_pairs_hook": observed["object_pairs_hook"],
        "ext_hook": observed["ext_hook"],
        "max_str_len": 65_536,
        "max_bin_len": 16_384,
        "max_array_len": 32,
        "max_map_len": 64,
        "max_ext_len": 0,
    }
    with pytest.raises(linklab.CodecError):
        _decode_primitive_message(b"\x82\xa1a\x01\xa1a\x02")
    with pytest.raises(linklab.CodecError):
        _decode_primitive_message(_pack({"x": msgpack.ExtType(1, b"x")}))
    nested: object = 0
    for _ in range(8):
        nested = [nested]
    with pytest.raises(linklab.CodecError):
        _decode_primitive_message(_pack({"x": nested}))


def test_peer_controlled_limits_cannot_raise_local_or_absolute_limits() -> None:
    server = SCHEMA_CASES[8][2]
    invalid_values = {
        "max_message_bytes": 262_145,
        "max_input_audio_frames": 1_601,
        "max_output_audio_frames": 1_601,
        "max_text_bytes": 65_537,
        "max_input_frames": 1_920_001,
        "idle_timeout_ms": 600_001,
    }
    for name, value in invalid_values.items():
        with pytest.raises(linklab.CodecError):
            _decode({**server, "limits": {**LIMITS, name: value}}, S2C)


EXPECTED_MODEL_HINTS: dict[type[object], dict[str, object]] = {
    linklab.AudioFormat: {"encoding": str, "sample_rate_hz": int, "channels": int},
    linklab.ConnectionLimits: {
        "max_message_bytes": int,
        "max_input_audio_frames": int,
        "max_output_audio_frames": int,
        "max_text_bytes": int,
        "max_input_frames": int,
        "idle_timeout_ms": int,
    },
    linklab.AnnotatedAudio: {
        "audio": linklab.ReadableBuffer,
        "generation": int,
        "discontinuity": bool,
        "speech": bool,
        "activated": bool,
        "wake_word": str | None,
    },
    linklab.ConnectionStateEvent: {"state": linklab.ConnectionState, "reason": str | None},
}

EXPECTED_MESSAGE_HINTS: dict[type[object], dict[str, object]] = {
    linklab.ClientHello: {
        "version": int,
        "capabilities": tuple[str, ...],
        "input_format": linklab.AudioFormat,
        "output_formats": tuple[linklab.AudioFormat, ...],
        "agent": str | None,
    },
    linklab.ServerHello: {
        "version": int,
        "capabilities": tuple[str, ...],
        "output_format": linklab.AudioFormat,
        "limits": linklab.ConnectionLimits,
        "agent": str | None,
    },
    linklab.ConversationStartedEvent: {
        "conversation_id": linklab.ConversationId,
        "activation": Literal["wake_word"],
        "wake_word": str | None,
    },
    linklab.InputStartedEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "reason": linklab.InputStartReason,
        "generation": int,
        "interrupts_response_id": linklab.ResponseId | None,
    },
    linklab.InputAudioEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "start_frame": int,
        "speech": bool,
        "audio": bytes,
    },
    linklab.InputAbortedEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "reason": linklab.InputAbortReason,
    },
    linklab.PlaybackFinishedEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
        "played_frames": int,
    },
    linklab.PlaybackInterruptedEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
        "played_frames": int,
        "position": linklab.PlaybackPosition,
        "reason": linklab.PlaybackInterruptReason,
    },
    linklab.ConversationCancelledEvent: {
        "conversation_id": linklab.ConversationId,
        "reason": linklab.ConversationCancelReason,
    },
    linklab.StateEvent: {
        "conversation_id": linklab.ConversationId,
        "revision": int,
        "state": linklab.CoarseState,
        "reason": str | None,
    },
    linklab.InputClosedEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "accepted_end_frame": int,
        "reason": linklab.InputCloseReason,
    },
    linklab.TranscriptUpdateEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "revision": int,
        "text": str,
        "language": str | None,
    },
    linklab.TranscriptFinalEvent: {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "text": str,
        "language": str | None,
    },
    linklab.ResponseStartedEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "input_id": linklab.InputId,
        "end_conversation": bool,
    },
    linklab.ResponseTextDeltaEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "sequence": int,
        "text": str,
    },
    linklab.ResponseTextFinalEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "text": str,
    },
    linklab.OutputStartedEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
    },
    linklab.OutputAudioEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
        "start_frame": int,
        "audio": bytes,
    },
    linklab.OutputEndedEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
        "total_frames": int,
    },
    linklab.ResponseEndedEvent: {"conversation_id": linklab.ConversationId, "response_id": linklab.ResponseId},
    linklab.ResponseCancelledEvent: {
        "conversation_id": linklab.ConversationId,
        "response_id": linklab.ResponseId,
        "reason": linklab.ResponseCancelReason,
    },
    linklab.ConversationEndedEvent: {
        "conversation_id": linklab.ConversationId,
        "reason": linklab.ConversationEndReason,
    },
    linklab.ErrorEvent: {
        "scope": linklab.ErrorScope,
        "code": linklab.ErrorCode,
        "fatal": bool,
        "conversation_id": linklab.ConversationId | None,
        "input_id": linklab.InputId | None,
        "response_id": linklab.ResponseId | None,
        "message": str | None,
    },
}


def test_public_model_exports_and_exact_field_types() -> None:
    for _, model, _, _ in SCHEMA_CASES:
        assert getattr(linklab, model.__name__) is model
        hints = get_type_hints(model)
        hints.pop("type", None)
        assert hints == EXPECTED_MESSAGE_HINTS[model]
        assert tuple(hints) == tuple(field.name for field in fields(model))
    for model, expected in EXPECTED_MODEL_HINTS.items():
        assert getattr(linklab, model.__name__) is model
        assert get_type_hints(model) == expected
        assert tuple(field.name for field in fields(model)) == tuple(expected)
    annotated_fields = fields(linklab.AnnotatedAudio)
    assert annotated_fields[-1].default is None
    assert all(field.default is MISSING for field in annotated_fields[:-1])
    assert tuple(item.value for item in linklab.AudioSubmitResult) == (
        "accepted",
        "ignored_inactive",
        "ignored_waiting_silence",
        "closed_input",
        "overflow",
    )
    assert len({id(linklab.ConversationId), id(linklab.InputId), id(linklab.ResponseId), id(linklab.OutputId)}) == 4
    for name in (
        "ConversationId",
        "InputId",
        "ResponseId",
        "OutputId",
        "ReadableBuffer",
        "AudioFormat",
        "ConnectionLimits",
        "ClientConfig",
        "ServerConfig",
        "AnnotatedAudio",
        "AudioSubmitResult",
    ):
        assert name in linklab.__all__


def test_public_semantic_values_contain_no_msgpack_objects() -> None:
    samples: tuple[object, ...] = tuple(_decode(wire, direction) for direction, _, wire, _ in SCHEMA_CASES) + (
        linklab.AnnotatedAudio(b"\0\0", 0, False, False, False),
        linklab.ConnectionStateEvent(linklab.ConnectionState.READY),
        linklab.ClientConfig("ws://localhost", (linklab.AudioFormat("pcm_s16le", 16_000, 1),)),
        linklab.ServerConfig(9000, (linklab.AudioFormat("pcm_s16le", 16_000, 1),)),
    )

    def assert_transport_neutral(value: object) -> None:
        assert not type(value).__module__.startswith("msgpack")
        if hasattr(value, "__dataclass_fields__"):
            for field in fields(value):  # type: ignore[arg-type]
                assert_transport_neutral(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                assert_transport_neutral(item)

    for sample in samples:
        assert_transport_neutral(sample)
        for annotation in get_type_hints(type(sample)).values():
            assert not getattr(annotation, "__module__", "").startswith("msgpack")
            if isinstance(annotation, UnionType):
                assert all(not getattr(arg, "__module__", "").startswith("msgpack") for arg in get_args(annotation))


@pytest.mark.parametrize("provider", ["bytes", "bytearray", "memoryview"])
def test_readable_buffer_providers_across_codec_constructors_ingress_and_output(provider: str) -> None:
    pcm_source = b"\x01\x02"
    wire_source = _pack(SCHEMA_CASES[1][2])

    def provide(source: bytes) -> Any:
        if provider == "bytes":
            return bytes(source)
        if provider == "bytearray":
            return bytearray(source)
        return memoryview(source)

    assert isinstance(_decode_primitive_message(provide(wire_source)), dict)
    assert (
        linklab.InputAudioEvent(linklab.ConversationId(1), linklab.InputId(1), 0, True, provide(pcm_source)).audio
        == pcm_source
    )
    assert (
        linklab.OutputAudioEvent(
            linklab.ConversationId(1), linklab.ResponseId(1), linklab.OutputId(1), 0, provide(pcm_source)
        ).audio
        == pcm_source
    )
    annotated = linklab.AnnotatedAudio(provide(pcm_source), 0, False, False, False)
    assert _ClientAudioIngress._copy_and_validate(annotated) == pcm_source
    assert _copy_output_pcm(provide(pcm_source)) == pcm_source


def test_unreadable_buffer_exporter_is_rejected_at_every_buffer_boundary() -> None:
    released = memoryview(b"\x01\x02")
    released.release()
    with pytest.raises(linklab.CodecError, match="readable buffer"):
        _decode_primitive_message(released)
    with pytest.raises(TypeError, match="buffer protocol"):
        linklab.InputAudioEvent(
            linklab.ConversationId(1),
            linklab.InputId(1),
            0,
            True,
            released,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="buffer protocol"):
        linklab.OutputAudioEvent(
            linklab.ConversationId(1),
            linklab.ResponseId(1),
            linklab.OutputId(1),
            0,
            released,  # type: ignore[arg-type]
        )
    annotated = linklab.AnnotatedAudio(released, 0, False, False, False)
    with pytest.raises(ValueError, match="buffer protocol"):
        _ClientAudioIngress._copy_and_validate(annotated)
    with pytest.raises(TypeError, match="buffer protocol"):
        _copy_output_pcm(released)
