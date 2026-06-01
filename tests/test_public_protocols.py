import typing
import inspect
from types import SimpleNamespace
from dataclasses import MISSING, FrozenInstanceError, fields
from collections.abc import Mapping

import pytest

import lumivox_linklab as linklab


def _assert_async_signatures(protocol: type[object], expected: Mapping[str, tuple[tuple[object, ...], object]]) -> None:
    declared = {
        name
        for name, value in protocol.__dict__.items()
        if inspect.isfunction(value) and (not name.startswith("_") or name in {"__aenter__", "__aexit__"})
    }
    assert declared == set(expected)
    for name, (parameter_types, return_type) in expected.items():
        method = getattr(protocol, name)
        assert inspect.iscoroutinefunction(method)
        signature = inspect.signature(method)
        assert tuple(signature.parameters)[0] == "self"
        hints = typing.get_type_hints(method)
        actual_parameter_types = tuple(hints[name] for name in tuple(signature.parameters)[1:])
        assert actual_parameter_types == parameter_types
        assert hints["return"] is return_type


def test_client_callbacks_runtime_protocol_and_exact_async_signatures() -> None:
    expected = {
        "on_connection_state": ((linklab.ConnectionStateEvent,), type(None)),
        "on_conversation_state": ((linklab.StateEvent,), type(None)),
        "on_transcript_update": ((linklab.TranscriptUpdateEvent,), type(None)),
        "on_transcript_final": ((linklab.TranscriptFinalEvent,), type(None)),
        "on_response_started": ((linklab.ResponseStartedEvent,), type(None)),
        "on_response_text_delta": ((linklab.ResponseTextDeltaEvent,), type(None)),
        "on_response_text_final": ((linklab.ResponseTextFinalEvent,), type(None)),
        "on_response_ended": ((linklab.ResponseEndedEvent,), type(None)),
        "on_response_cancelled": ((linklab.ResponseCancelledEvent,), type(None)),
        "on_output_started": ((linklab.OutputStartedEvent,), type(None)),
        "on_output_audio": ((linklab.OutputAudioEvent,), type(None)),
        "on_output_ended": ((linklab.OutputEndedEvent,), type(None)),
        "on_conversation_ended": ((linklab.ConversationEndedEvent,), type(None)),
        "on_error": ((linklab.ErrorEvent,), type(None)),
    }
    _assert_async_signatures(linklab.ClientCallbacks, expected)

    async def callback(_event: object) -> None:
        return None

    implementation = SimpleNamespace(**dict.fromkeys(expected, callback))
    assert isinstance(implementation, linklab.ClientCallbacks)
    assert not isinstance(object(), linklab.ClientCallbacks)


def test_server_handler_runtime_protocol_and_exact_async_signatures() -> None:
    expected = {
        "on_conversation_started": ((linklab.ServerSession, linklab.ConversationStartedEvent), type(None)),
        "on_input_started": ((linklab.ServerSession, linklab.InputStartedEvent), type(None)),
        "on_input_audio": ((linklab.ServerSession, linklab.InputAudioEvent), type(None)),
        "on_input_aborted": ((linklab.ServerSession, linklab.InputAbortedEvent), type(None)),
        "on_playback_outcome": (
            (linklab.ServerSession, linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent),
            type(None),
        ),
        "on_conversation_cancelled": ((linklab.ServerSession, linklab.ConversationCancelledEvent), type(None)),
    }
    _assert_async_signatures(linklab.ServerHandler, expected)

    async def handler(_session: object, _event: object) -> None:
        return None

    implementation = SimpleNamespace(**dict.fromkeys(expected, handler))
    assert isinstance(implementation, linklab.ServerHandler)
    assert not isinstance(object(), linklab.ServerHandler)


def test_server_session_exact_async_signatures_and_readonly_surface() -> None:
    expected = {
        "close_input": ((linklab.InputId, linklab.InputCloseReason), type(None)),
        "update_transcript": ((linklab.InputId, int, str, str | None), type(None)),
        "finalize_transcript": ((linklab.InputId, str, str | None), type(None)),
        "start_response": ((linklab.InputId, bool), linklab.ResponseWriter),
        "end_conversation": ((linklab.ConversationEndReason,), type(None)),
        "fail": ((linklab.ErrorScope, linklab.ErrorCode, str | None), type(None)),
    }
    _assert_async_signatures(linklab.ServerSession, expected)
    assert {name for name in dir(linklab.ServerSession) if not name.startswith("_")} == set(expected)

    assert (
        inspect.signature(linklab.ServerSession.start_response).parameters["end_conversation"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert inspect.signature(linklab.ServerSession.start_response).parameters["end_conversation"].default is False
    for method, parameter in (
        ("update_transcript", "language"),
        ("finalize_transcript", "language"),
        ("fail", "message"),
    ):
        assert inspect.signature(getattr(linklab.ServerSession, method)).parameters[parameter].default is None


def test_writer_exact_async_signatures_readonly_ids_and_context_annotations() -> None:
    response_expected = {
        "send_text_delta": ((str,), type(None)),
        "finalize_text": ((str,), type(None)),
        "start_output": ((), linklab.OutputWriter),
        "finish": ((), type(None)),
        "cancel": ((linklab.ResponseCancelReason,), type(None)),
        "__aenter__": ((), typing.Self),
        "__aexit__": ((object, object, object), typing.Literal[False]),
    }
    output_expected = {
        "send_audio": ((linklab.ReadableBuffer,), type(None)),
        "finish": ((), type(None)),
        "__aenter__": ((), typing.Self),
        "__aexit__": ((object, object, object), typing.Literal[False]),
    }
    _assert_async_signatures(linklab.ResponseWriter, response_expected)
    _assert_async_signatures(linklab.OutputWriter, output_expected)

    response_id = inspect.getattr_static(linklab.ResponseWriter, "response_id")
    output_id = inspect.getattr_static(linklab.OutputWriter, "output_id")
    assert isinstance(response_id, property) and response_id.fset is None
    assert isinstance(output_id, property) and output_id.fset is None
    assert typing.get_type_hints(response_id.fget)["return"] is linklab.ResponseId
    assert typing.get_type_hints(output_id.fget)["return"] is linklab.OutputId


def test_codec_and_validator_exact_public_signatures_and_readonly_properties() -> None:
    encode_signature = inspect.signature(linklab.encode_message)
    decode_signature = inspect.signature(linklab.decode_message)
    assert tuple(encode_signature.parameters) == ("message",)
    assert typing.get_type_hints(linklab.encode_message) == {"message": linklab.Message, "return": bytes}
    assert tuple(decode_signature.parameters) == ("data", "direction", "limits")
    assert decode_signature.parameters["direction"].kind is inspect.Parameter.KEYWORD_ONLY
    assert decode_signature.parameters["limits"].kind is inspect.Parameter.KEYWORD_ONLY
    assert typing.get_type_hints(linklab.decode_message) == {
        "data": linklab.ReadableBuffer,
        "direction": linklab.MessageDirection,
        "limits": linklab.ConnectionLimits,
        "return": linklab.Message,
    }
    assert str(inspect.signature(linklab.ProtocolValidator)) == "(direction: 'EndpointRole') -> 'None'"
    for name in ("transport_connected", "begin_close", "transport_disconnected"):
        assert str(inspect.signature(getattr(linklab.ProtocolValidator, name))) == "(self) -> 'None'"
    assert str(inspect.signature(linklab.ProtocolValidator.accept)) == "(self, message: 'Message') -> 'None'"

    state = inspect.getattr_static(linklab.ProtocolValidator, "state")
    tombstones = inspect.getattr_static(linklab.ProtocolValidator, "tombstones")
    assert isinstance(state, property) and state.fset is None
    assert isinstance(tombstones, property) and tombstones.fset is None
    assert typing.get_type_hints(state.fget)["return"] is linklab.ProtocolStateSnapshot
    assert typing.get_type_hints(tombstones.fget)["return"] == tuple[linklab.ProtocolTombstone, ...]


def test_protocol_inspection_values_have_exact_frozen_slotted_shapes() -> None:
    snapshot_fields = fields(linklab.ProtocolStateSnapshot)
    tombstone_fields = fields(linklab.ProtocolTombstone)
    assert tuple(field.name for field in snapshot_fields) == (
        "connection_state",
        "conversation_id",
        "input_id",
        "response_id",
        "output_id",
        "coarse_state",
        "state_revision",
    )
    assert tuple(field.name for field in tombstone_fields) == (
        "kind",
        "conversation_id",
        "input_id",
        "response_id",
        "output_id",
        "terminal_state",
    )
    assert all(
        field.default is MISSING and field.default_factory is MISSING for field in (*snapshot_fields, *tombstone_fields)
    )
    assert typing.get_type_hints(linklab.ProtocolStateSnapshot) == {
        "connection_state": linklab.ConnectionState,
        "conversation_id": linklab.ConversationId | None,
        "input_id": linklab.InputId | None,
        "response_id": linklab.ResponseId | None,
        "output_id": linklab.OutputId | None,
        "coarse_state": linklab.CoarseState | None,
        "state_revision": int | None,
    }
    assert typing.get_type_hints(linklab.ProtocolTombstone) == {
        "kind": linklab.ProtocolObjectKind,
        "conversation_id": linklab.ConversationId | None,
        "input_id": linklab.InputId | None,
        "response_id": linklab.ResponseId | None,
        "output_id": linklab.OutputId | None,
        "terminal_state": str,
    }
    assert [kind.value for kind in linklab.ProtocolObjectKind] == ["conversation", "input", "response", "output"]

    value = linklab.ProtocolTombstone(
        linklab.ProtocolObjectKind.CONVERSATION,
        linklab.ConversationId(1),
        None,
        None,
        None,
        "completed",
    )
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.terminal_state = "changed"  # type: ignore[misc]
