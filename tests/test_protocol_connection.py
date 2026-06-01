import typing
from typing import Any, Callable
from dataclasses import FrozenInstanceError, fields

import pytest

import lumivox_linklab as linklab
from lumivox_linklab._protocol import _IdAllocator

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
LIMITS_16K = linklab.ConnectionLimits(max_output_audio_frames=1_600)


def client_hello() -> linklab.ClientHello:
    return linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,))


def server_hello() -> linklab.ServerHello:
    return linklab.ServerHello(1, CAPABILITIES, PCM_16K, LIMITS_16K)


def ready_validator(role: linklab.EndpointRole) -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(role)
    validator.transport_connected()
    validator.accept(client_hello())
    validator.accept(server_hello())
    return validator


@pytest.mark.parametrize("role", list(linklab.EndpointRole))
def test_handshake_and_transport_lifecycle_for_both_roles(role: linklab.EndpointRole) -> None:
    validator = linklab.ProtocolValidator(role)
    assert validator.state.connection_state is linklab.ConnectionState.DISCONNECTED

    validator.transport_connected()
    handshaking = validator.state
    assert handshaking.connection_state is linklab.ConnectionState.HANDSHAKING
    validator.accept(client_hello())
    validator.accept(server_hello())
    ready = validator.state
    assert ready.connection_state is linklab.ConnectionState.READY
    validator.begin_close()
    closing = validator.state
    assert closing.connection_state is linklab.ConnectionState.CLOSING
    validator.transport_disconnected()
    assert validator.state == linklab.ProtocolStateSnapshot(
        linklab.ConnectionState.DISCONNECTED, None, None, None, None, None, None
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda validator: validator.begin_close(),
        lambda validator: validator.transport_disconnected(),
    ],
)
def test_invalid_disconnected_lifecycle_call_is_atomic(operation: Callable[[linklab.ProtocolValidator], None]) -> None:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
    before = validator.state
    with pytest.raises(linklab.ProtocolViolation):
        operation(validator)
    assert validator.state == before


def test_connect_rejects_duplicate_and_disconnects_from_every_active_state() -> None:
    for state in (linklab.ConnectionState.HANDSHAKING, linklab.ConnectionState.READY, linklab.ConnectionState.CLOSING):
        validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
        validator.transport_connected()
        with pytest.raises(linklab.ProtocolViolation):
            validator.transport_connected()
        if state is not linklab.ConnectionState.HANDSHAKING:
            validator.accept(client_hello())
            validator.accept(server_hello())
        if state is linklab.ConnectionState.CLOSING:
            validator.begin_close()
        validator.transport_disconnected()
        assert validator.state.connection_state is linklab.ConnectionState.DISCONNECTED


def test_handshake_order_format_and_error_in_place_of_server_hello() -> None:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
    validator.transport_connected()
    before = validator.state
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(server_hello())
    assert validator.state == before

    validator.accept(client_hello())
    error = linklab.ErrorEvent(linklab.ErrorScope.CONNECTION, linklab.ErrorCode.FORMAT_MISMATCH, True)
    validator.accept(error)
    assert validator.state.connection_state is linklab.ConnectionState.CLOSING


def test_server_hello_must_select_advertised_format_atomically() -> None:
    pcm_24k = linklab.AudioFormat("pcm_s16le", 24_000, 1)
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
    validator.transport_connected()
    validator.accept(client_hello())
    before = validator.state
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(
            linklab.ServerHello(
                1,
                CAPABILITIES,
                pcm_24k,
                linklab.ConnectionLimits(max_output_audio_frames=2_400),
            )
        )
    assert validator.state == before


@pytest.mark.parametrize("role", list(linklab.EndpointRole))
def test_conversation_start_state_cancel_and_end(role: linklab.EndpointRole) -> None:
    validator = ready_validator(role)
    conversation_id = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    validator.accept(linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING))
    listening = validator.state
    assert listening.coarse_state is linklab.CoarseState.LISTENING
    assert listening.state_revision == 1

    validator.accept(linklab.StateEvent(conversation_id, 2, linklab.CoarseState.PROCESSING))
    cancel = linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.USER)
    validator.accept(cancel)
    validator.accept(cancel)
    end = linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.CANCELLED)
    validator.accept(end)
    validator.accept(end)

    ended = validator.state
    assert ended.conversation_id is None
    assert ended.coarse_state is None
    assert validator.tombstones == (
        linklab.ProtocolTombstone(
            linklab.ProtocolObjectKind.CONVERSATION,
            conversation_id,
            None,
            None,
            None,
            "cancelled",
        ),
    )


def test_conversation_ids_are_monotonic_and_tombstoned_ids_are_not_reused() -> None:
    validator = ready_validator(linklab.EndpointRole.SERVER)
    first = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(first, "wake_word"))
    validator.accept(linklab.ConversationEndedEvent(first, linklab.ConversationEndReason.COMPLETED))

    before = validator.state
    tombstones = validator.tombstones
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(linklab.ConversationStartedEvent(first, "wake_word"))
    assert validator.state == before
    assert validator.tombstones == tombstones

    second = linklab.ConversationId(2)
    validator.accept(linklab.ConversationStartedEvent(second, "wake_word"))
    assert validator.state.conversation_id == second


def test_late_identical_terminals_and_newer_state_do_not_target_new_conversation() -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    first = linklab.ConversationId(1)
    cancel = linklab.ConversationCancelledEvent(first, linklab.ConversationCancelReason.USER)
    end = linklab.ConversationEndedEvent(first, linklab.ConversationEndReason.CANCELLED)
    validator.accept(linklab.ConversationStartedEvent(first, "wake_word"))
    validator.accept(linklab.StateEvent(first, 1, linklab.CoarseState.LISTENING))
    validator.accept(cancel)
    validator.accept(end)

    second = linklab.ConversationId(2)
    validator.accept(linklab.ConversationStartedEvent(second, "wake_word"))
    validator.accept(cancel)
    validator.accept(end)
    validator.accept(linklab.StateEvent(first, 2, linklab.CoarseState.PROCESSING))
    assert validator.state.conversation_id == second


def test_conflicting_duplicates_and_state_revision_fail_atomically() -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    conversation_id = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    first = linklab.StateEvent(conversation_id, 1, linklab.CoarseState.LISTENING)
    validator.accept(first)
    validator.accept(first)

    for invalid in (
        linklab.StateEvent(conversation_id, 2, linklab.CoarseState.LISTENING),
        linklab.StateEvent(conversation_id, 3, linklab.CoarseState.PROCESSING),
    ):
        before = validator.state
        with pytest.raises(linklab.ProtocolViolation):
            validator.accept(invalid)
        assert validator.state == before

    cancel = linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.USER)
    validator.accept(cancel)
    before = validator.state
    with pytest.raises(linklab.ProtocolViolation):
        validator.accept(
            linklab.ConversationCancelledEvent(conversation_id, linklab.ConversationCancelReason.CLIENT_FAILED)
        )
    assert validator.state == before


@pytest.mark.parametrize(
    ("code", "end_reason"),
    [
        (linklab.ErrorCode.CONVERSATION_FAILED, linklab.ConversationEndReason.SERVER_FAILED),
        (linklab.ErrorCode.IDLE_TIMEOUT, linklab.ConversationEndReason.IDLE_TIMEOUT),
    ],
)
def test_conversation_error_requires_matching_terminal_end(
    code: linklab.ErrorCode, end_reason: linklab.ConversationEndReason
) -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    conversation_id = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    validator.accept(linklab.ErrorEvent(linklab.ErrorScope.CONVERSATION, code, True, conversation_id))
    validator.accept(linklab.ConversationEndedEvent(conversation_id, end_reason))
    assert validator.tombstones[0].terminal_state == end_reason.value


def test_disconnect_invalidates_ids_tombstones_and_handshake() -> None:
    validator = ready_validator(linklab.EndpointRole.SERVER)
    conversation_id = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    validator.accept(linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.COMPLETED))
    assert validator.tombstones

    validator.transport_disconnected()
    assert validator.tombstones == ()
    validator.transport_connected()
    validator.accept(client_hello())
    validator.accept(server_hello())
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    assert validator.state.conversation_id == conversation_id


def test_public_snapshots_and_tombstones_are_immutable_copies() -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    conversation_id = linklab.ConversationId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    snapshot = validator.state
    with pytest.raises(FrozenInstanceError):
        snapshot.conversation_id = None  # type: ignore[misc]

    validator.accept(linklab.ConversationEndedEvent(conversation_id, linklab.ConversationEndReason.COMPLETED))
    tombstone = validator.tombstones[0]
    with pytest.raises(FrozenInstanceError):
        tombstone.terminal_state = "changed"  # type: ignore[misc]
    assert validator.tombstones[0].terminal_state == "completed"


def test_allocator_emits_uint32_max_then_fails_without_mutation_or_wrap() -> None:
    allocator = _IdAllocator(4_294_967_295)
    value, exhausted = allocator.allocate()
    assert value == 4_294_967_295
    assert exhausted.next_value == 4_294_967_296

    with pytest.raises(linklab.ProtocolViolation, match="id_exhausted"):
        exhausted.allocate()
    assert exhausted.next_value == 4_294_967_296


def test_all_four_allocators_are_independent_sequential_and_exhaustion_is_local_fatal() -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    conversation_id = linklab.ConversationId(1)
    input_id = linklab.InputId(1)
    response_id = linklab.ResponseId(1)
    output_id = linklab.OutputId(1)
    validator.accept(linklab.ConversationStartedEvent(conversation_id, "wake_word"))
    validator.accept(linklab.InputStartedEvent(conversation_id, input_id, linklab.InputStartReason.ACTIVATION, 0))
    validator.accept(linklab.InputClosedEvent(conversation_id, input_id, 0, linklab.InputCloseReason.NO_SPEECH))
    validator.accept(linklab.TranscriptFinalEvent(conversation_id, input_id, ""))
    validator.accept(linklab.ResponseStartedEvent(conversation_id, response_id, input_id, False))
    validator.accept(linklab.OutputStartedEvent(conversation_id, response_id, output_id))

    allocators = (
        validator._data.conversation_ids,
        validator._data.input_ids,
        validator._data.response_ids,
        validator._data.output_ids,
    )
    assert tuple(allocator.next_value for allocator in allocators) == (2, 2, 2, 2)
    assert tuple(allocator.allocate()[0] for allocator in allocators) == (2, 2, 2, 2)
    assert tuple(allocator.next_value for allocator in allocators) == (2, 2, 2, 2)

    exhausted_allocators = tuple(_IdAllocator(4_294_967_296) for _ in allocators)
    for exhausted in exhausted_allocators:
        with pytest.raises(linklab.ProtocolViolation, match="id_exhausted"):
            exhausted.allocate()
    assert tuple(allocator.next_value for allocator in exhausted_allocators) == (4_294_967_296,) * 4


def test_every_post_handshake_message_schema_carries_explicit_scope_ids() -> None:
    expected_ids: dict[type[Any], tuple[str, ...]] = {
        linklab.ConversationStartedEvent: ("conversation_id",),
        linklab.InputStartedEvent: ("conversation_id", "input_id"),
        linklab.InputAudioEvent: ("conversation_id", "input_id"),
        linklab.InputAbortedEvent: ("conversation_id", "input_id"),
        linklab.PlaybackFinishedEvent: ("conversation_id", "response_id", "output_id"),
        linklab.PlaybackInterruptedEvent: ("conversation_id", "response_id", "output_id"),
        linklab.ConversationCancelledEvent: ("conversation_id",),
        linklab.StateEvent: ("conversation_id",),
        linklab.InputClosedEvent: ("conversation_id", "input_id"),
        linklab.TranscriptUpdateEvent: ("conversation_id", "input_id"),
        linklab.TranscriptFinalEvent: ("conversation_id", "input_id"),
        linklab.ResponseStartedEvent: ("conversation_id", "response_id", "input_id"),
        linklab.ResponseTextDeltaEvent: ("conversation_id", "response_id"),
        linklab.ResponseTextFinalEvent: ("conversation_id", "response_id"),
        linklab.OutputStartedEvent: ("conversation_id", "response_id", "output_id"),
        linklab.OutputAudioEvent: ("conversation_id", "response_id", "output_id"),
        linklab.OutputEndedEvent: ("conversation_id", "response_id", "output_id"),
        linklab.ResponseEndedEvent: ("conversation_id", "response_id"),
        linklab.ResponseCancelledEvent: ("conversation_id", "response_id"),
        linklab.ConversationEndedEvent: ("conversation_id",),
    }
    id_types = {
        "conversation_id": linklab.ConversationId,
        "input_id": linklab.InputId,
        "response_id": linklab.ResponseId,
        "output_id": linklab.OutputId,
    }
    for message_type, names in expected_ids.items():
        assert names == tuple(field.name for field in fields(message_type) if field.name in id_types)
        hints = typing.get_type_hints(message_type)
        assert all(hints[name] is id_types[name] for name in names)

    error_hints = typing.get_type_hints(linklab.ErrorEvent)
    error_id_types = {name: id_type for name, id_type in id_types.items() if name != "output_id"}
    assert all(error_hints[name] == id_type | None for name, id_type in error_id_types.items())


@pytest.mark.parametrize(
    "message",
    [
        linklab.ConversationStartedEvent(linklab.ConversationId(2), "wake_word"),
        linklab.StateEvent(linklab.ConversationId(1), 1, linklab.CoarseState.WAITING),
    ],
)
def test_conversation_messages_after_begin_close_are_rejected_atomically(message: linklab.Message) -> None:
    validator = ready_validator(linklab.EndpointRole.CLIENT)
    if not isinstance(message, linklab.ConversationStartedEvent):
        validator.accept(linklab.ConversationStartedEvent(linklab.ConversationId(1), "wake_word"))
    validator.begin_close()
    before = validator._data
    with pytest.raises(linklab.ProtocolViolation, match="active connection"):
        validator.accept(message)
    assert validator._data == before


def test_constructor_requires_endpoint_role() -> None:
    with pytest.raises(TypeError):
        linklab.ProtocolValidator("client")  # type: ignore[arg-type]
