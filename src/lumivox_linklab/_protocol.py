from __future__ import annotations

from dataclasses import replace, dataclass

from ._enums import (
    ErrorCode,
    ErrorScope,
    EndpointRole,
    ConnectionState,
    ProtocolObjectKind,
)
from ._errors import ProtocolViolation
from ._values import ConversationId, ProtocolTombstone, ProtocolStateSnapshot
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
    ClientHello,
    ServerHello,
    ConversationEndedEvent,
    ConversationStartedEvent,
    ConversationCancelledEvent,
)

_MAX_ID = 4_294_967_295


@dataclass(frozen=True, slots=True)
class _TransitionResult:
    dispatch: bool
    outbound: tuple[Message, ...] = ()
    cancellation_targets: tuple[ProtocolObjectKind, ...] = ()
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class _IdAllocator:
    next_value: int = 1

    def __post_init__(self) -> None:
        if type(self.next_value) is not int or not 1 <= self.next_value <= _MAX_ID + 1:
            raise ValueError("next_value must be in 1..4294967296")

    def allocate(self) -> tuple[int, _IdAllocator]:
        if self.next_value > _MAX_ID:
            raise ProtocolViolation("id_exhausted")
        return self.next_value, replace(self, next_value=self.next_value + 1)

    def observe(self, value: int) -> _IdAllocator:
        if value != self.next_value:
            raise ProtocolViolation(f"expected ID {self.next_value}, got {value}")
        _, allocator = self.allocate()
        return allocator


@dataclass(frozen=True, slots=True)
class _Conversation:
    conversation_id: ConversationId
    cancel: ConversationCancelledEvent | None = None
    failure: ErrorEvent | None = None
    state: StateEvent | None = None


@dataclass(frozen=True, slots=True)
class _ValidatorData:
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    client_hello: ClientHello | None = None
    server_hello: ServerHello | None = None
    conversation: _Conversation | None = None
    conversation_ids: _IdAllocator = _IdAllocator()
    tombstones: tuple[ProtocolTombstone, ...] = ()
    ended_conversations: tuple[ConversationEndedEvent, ...] = ()
    cancelled_conversations: tuple[ConversationCancelledEvent, ...] = ()
    terminal_states: tuple[StateEvent, ...] = ()


class ProtocolValidator:
    def __init__(self, direction: EndpointRole) -> None:
        if not isinstance(direction, EndpointRole):
            raise TypeError("direction must be EndpointRole")
        self._role = direction
        self._data = _ValidatorData()

    def transport_connected(self) -> None:
        if self._data.connection_state is not ConnectionState.DISCONNECTED:
            raise ProtocolViolation("transport is already connected")
        self._data = _ValidatorData(connection_state=ConnectionState.HANDSHAKING)

    def begin_close(self) -> None:
        if self._data.connection_state not in (ConnectionState.HANDSHAKING, ConnectionState.READY):
            raise ProtocolViolation("connection cannot begin closing from its current state")
        self._data = replace(self._data, connection_state=ConnectionState.CLOSING)

    def transport_disconnected(self) -> None:
        if self._data.connection_state is ConnectionState.DISCONNECTED:
            raise ProtocolViolation("transport is already disconnected")
        self._data = _ValidatorData()

    def accept(self, message: Message) -> None:
        data, _ = self._transition(self._data, message)
        self._data = data

    @property
    def state(self) -> ProtocolStateSnapshot:
        conversation = self._data.conversation
        state = conversation.state if conversation is not None else None
        return ProtocolStateSnapshot(
            connection_state=self._data.connection_state,
            conversation_id=conversation.conversation_id if conversation is not None else None,
            input_id=None,
            response_id=None,
            output_id=None,
            coarse_state=state.state if state is not None else None,
            state_revision=state.revision if state is not None else None,
        )

    @property
    def tombstones(self) -> tuple[ProtocolTombstone, ...]:
        return self._data.tombstones

    def _transition(self, data: _ValidatorData, message: Message) -> tuple[_ValidatorData, _TransitionResult]:
        if data.connection_state is ConnectionState.HANDSHAKING:
            return self._accept_handshake(data, message)
        if data.connection_state is not ConnectionState.READY:
            raise ProtocolViolation("application messages require an active connection")

        if isinstance(message, ConversationStartedEvent):
            return self._start_conversation(data, message)
        if isinstance(message, ConversationCancelledEvent):
            return self._cancel_conversation(data, message)
        if isinstance(message, ConversationEndedEvent):
            return self._end_conversation(data, message)
        if isinstance(message, StateEvent):
            return self._accept_state(data, message)
        if isinstance(message, ErrorEvent):
            return self._accept_error(data, message)
        if isinstance(message, (ClientHello, ServerHello)):
            raise ProtocolViolation("hello is only legal during handshake")
        raise ProtocolViolation(f"{message.type} is not implemented by this validator stage")

    def _accept_handshake(self, data: _ValidatorData, message: Message) -> tuple[_ValidatorData, _TransitionResult]:
        if isinstance(message, ClientHello):
            if data.client_hello is not None or data.server_hello is not None:
                raise ProtocolViolation("client hello must be the first application message")
            return replace(data, client_hello=message), self._result(message, dispatch=False)

        if isinstance(message, ServerHello):
            if data.client_hello is None or data.server_hello is not None:
                raise ProtocolViolation("server hello must follow exactly one client hello")
            if message.output_format not in data.client_hello.output_formats:
                raise ProtocolViolation("server selected an output format the client did not advertise")
            ready = replace(data, server_hello=message, connection_state=ConnectionState.READY)
            return ready, self._result(message, dispatch=False)

        if isinstance(message, ErrorEvent) and message.scope is ErrorScope.CONNECTION:
            if data.client_hello is None or data.server_hello is not None:
                raise ProtocolViolation("handshake error must replace server hello")
            closing = replace(data, connection_state=ConnectionState.CLOSING)
            return closing, self._result(message, terminal=True)

        raise ProtocolViolation("only hello or a server connection error is legal during handshake")

    def _start_conversation(
        self, data: _ValidatorData, message: ConversationStartedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        if data.conversation is not None:
            raise ProtocolViolation("only one conversation may be open")
        allocator = data.conversation_ids.observe(message.conversation_id)
        conversation = _Conversation(message.conversation_id)
        return replace(data, conversation=conversation, conversation_ids=allocator), self._result(message)

    def _cancel_conversation(
        self, data: _ValidatorData, message: ConversationCancelledEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        previous = self._find_by_id(data.cancelled_conversations, message.conversation_id)
        if previous is not None:
            if previous == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting conversation cancellation")
        if self._find_by_id(data.ended_conversations, message.conversation_id) is not None:
            raise ProtocolViolation("ended conversation cannot be cancelled")

        conversation = data.conversation
        if conversation is None:
            raise ProtocolViolation("conversation cancellation targets no open conversation")
        if message.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("conversation cancellation ID does not match the open conversation")
        if conversation.failure is not None:
            raise ProtocolViolation("failed conversation cannot be cancelled")
        updated = replace(conversation, cancel=message)
        cancelled = (*data.cancelled_conversations, message)
        return replace(data, conversation=updated, cancelled_conversations=cancelled), self._result(
            message,
            cancellation_targets=(ProtocolObjectKind.CONVERSATION,),
            terminal=True,
        )

    def _end_conversation(
        self, data: _ValidatorData, message: ConversationEndedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        previous = self._find_by_id(data.ended_conversations, message.conversation_id)
        if previous is not None:
            if previous == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting conversation end")

        conversation = data.conversation
        if conversation is None:
            raise ProtocolViolation("conversation end targets no open conversation")
        if message.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("conversation end ID does not match the open conversation")
        if conversation.cancel is not None and message.reason.value != "cancelled":
            raise ProtocolViolation("cancelled conversation must end as cancelled")
        if conversation.failure is not None:
            expected = "idle_timeout" if conversation.failure.code is ErrorCode.IDLE_TIMEOUT else "server_failed"
            if message.reason.value != expected:
                raise ProtocolViolation(f"failed conversation must end as {expected}")

        tombstone = ProtocolTombstone(
            kind=ProtocolObjectKind.CONVERSATION,
            conversation_id=message.conversation_id,
            input_id=None,
            response_id=None,
            output_id=None,
            terminal_state=message.reason.value,
        )
        terminal_states = data.terminal_states
        if conversation.state is not None:
            terminal_states = (*terminal_states, conversation.state)
        updated = replace(
            data,
            conversation=None,
            tombstones=(*data.tombstones, tombstone),
            ended_conversations=(*data.ended_conversations, message),
            terminal_states=terminal_states,
        )
        return updated, self._result(message, terminal=True)

    def _accept_state(self, data: _ValidatorData, message: StateEvent) -> tuple[_ValidatorData, _TransitionResult]:
        ended = self._find_by_id(data.ended_conversations, message.conversation_id)
        if ended is not None:
            previous = self._find_by_id(data.terminal_states, message.conversation_id)
            if previous is None or message.revision > previous.revision or message == previous:
                return data, self._result(message, dispatch=False)
            raise ProtocolViolation("late state conflicts with terminal conversation state")

        conversation = data.conversation
        if conversation is None:
            raise ProtocolViolation("state targets no known conversation")
        if message.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("state conversation ID does not match the open conversation")
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("terminal conversation cannot change coarse state")

        previous = conversation.state
        if previous is None:
            if message.revision != 1:
                raise ProtocolViolation("first state revision must be 1")
        else:
            if message == previous:
                return data, self._result(message, dispatch=False)
            if message.revision != previous.revision + 1:
                raise ProtocolViolation("state revision must increment exactly by one")
            if message.state is previous.state:
                raise ProtocolViolation("state revision requires a coarse-state value change")
        updated = replace(data, conversation=replace(conversation, state=message))
        return updated, self._result(message)

    def _accept_error(self, data: _ValidatorData, message: ErrorEvent) -> tuple[_ValidatorData, _TransitionResult]:
        if message.scope is ErrorScope.CONNECTION:
            closing = replace(data, connection_state=ConnectionState.CLOSING)
            return closing, self._result(
                message,
                cancellation_targets=(ProtocolObjectKind.CONVERSATION,),
                terminal=True,
            )
        if message.scope is not ErrorScope.CONVERSATION:
            raise ProtocolViolation("input and response errors are not implemented by this validator stage")
        conversation = data.conversation
        if conversation is None or message.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("conversation error targets no open conversation")
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("conversation is already terminal")
        updated = replace(data, conversation=replace(conversation, failure=message))
        return updated, self._result(
            message,
            cancellation_targets=(ProtocolObjectKind.CONVERSATION,),
            terminal=True,
        )

    def _result(
        self,
        message: Message,
        *,
        dispatch: bool | None = None,
        cancellation_targets: tuple[ProtocolObjectKind, ...] = (),
        terminal: bool = False,
    ) -> _TransitionResult:
        if dispatch is None:
            from_client = isinstance(message, (ClientHello, ConversationStartedEvent, ConversationCancelledEvent))
            dispatch = from_client is (self._role is EndpointRole.SERVER)
        return _TransitionResult(
            dispatch=dispatch,
            cancellation_targets=cancellation_targets,
            terminal=terminal,
        )

    @staticmethod
    def _find_by_id[T: ConversationCancelledEvent | ConversationEndedEvent | StateEvent](
        events: tuple[T, ...], conversation_id: ConversationId
    ) -> T | None:
        for event in reversed(events):
            if event.conversation_id == conversation_id:
                return event
        return None
