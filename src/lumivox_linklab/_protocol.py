from __future__ import annotations

from dataclasses import replace, dataclass

from ._enums import (
    ErrorCode,
    ErrorScope,
    CoarseState,
    EndpointRole,
    ConnectionState,
    InputCloseReason,
    InputStartReason,
    ProtocolObjectKind,
    ResponseCancelReason,
    ConversationEndReason,
)
from ._errors import ProtocolViolation
from ._values import InputId, OutputId, ResponseId, ConversationId, ProtocolTombstone, ProtocolStateSnapshot
from ._messages import (
    Message,
    ErrorEvent,
    StateEvent,
    ClientHello,
    ServerHello,
    InputAudioEvent,
    InputClosedEvent,
    OutputAudioEvent,
    OutputEndedEvent,
    InputAbortedEvent,
    InputStartedEvent,
    OutputStartedEvent,
    ResponseEndedEvent,
    ResponseStartedEvent,
    TranscriptFinalEvent,
    TranscriptUpdateEvent,
    ConversationEndedEvent,
    ResponseCancelledEvent,
    ResponseTextDeltaEvent,
    ResponseTextFinalEvent,
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
    input_id: InputId | None = None
    response_id: ResponseId | None = None
    output_id: OutputId | None = None
    expected_end_reason: ConversationEndReason | None = None


@dataclass(frozen=True, slots=True)
class _CancelledInput:
    pass


@dataclass(frozen=True, slots=True)
class _Input:
    start: InputStartedEvent
    received_end_frame: int = 0
    received_boundaries: tuple[int, ...] = ()
    committed_end_frame: int = 0
    terminal: InputClosedEvent | InputAbortedEvent | _CancelledInput | None = None
    transcript_update: TranscriptUpdateEvent | None = None
    transcript_final: TranscriptFinalEvent | None = None
    failure: ErrorEvent | None = None
    response_id: ResponseId | None = None


@dataclass(frozen=True, slots=True)
class _CancelledOutput:
    pass


@dataclass(frozen=True, slots=True)
class _Output:
    start: OutputStartedEvent
    received_end_frame: int = 0
    terminal: OutputEndedEvent | _CancelledOutput | None = None


@dataclass(frozen=True, slots=True)
class _Response:
    start: ResponseStartedEvent
    next_text_sequence: int = 0
    text: str = ""
    text_final: ResponseTextFinalEvent | None = None
    output: _Output | None = None
    end: ResponseEndedEvent | None = None
    terminal: ResponseEndedEvent | ResponseCancelledEvent | None = None


@dataclass(frozen=True, slots=True)
class _ValidatorData:
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    client_hello: ClientHello | None = None
    server_hello: ServerHello | None = None
    conversation: _Conversation | None = None
    conversation_ids: _IdAllocator = _IdAllocator()
    input_ids: _IdAllocator = _IdAllocator()
    response_ids: _IdAllocator = _IdAllocator()
    output_ids: _IdAllocator = _IdAllocator()
    inputs: tuple[_Input, ...] = ()
    responses: tuple[_Response, ...] = ()
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

    def _commit_input_audio(self, input_id: InputId, end_frame: int) -> tuple[Message, ...]:
        """Record PCM delivery to the server application callback."""
        if type(end_frame) is not int or end_frame < 0:
            raise ProtocolViolation("committed input boundary must be a nonnegative integer")
        input_ = self._find_input(self._data, input_id)
        if input_ is None:
            raise ProtocolViolation("committed input boundary targets no known input")
        if input_.terminal is not None:
            raise ProtocolViolation("terminal input boundary is immutable")
        if not input_.committed_end_frame <= end_frame <= input_.received_end_frame:
            raise ProtocolViolation("committed input boundary must be monotonic and already received")
        if end_frame != 0 and end_frame not in input_.received_boundaries:
            raise ProtocolViolation("committed input boundary must end a received audio chunk")
        input_ = replace(input_, committed_end_frame=end_frame)
        self._data = self._replace_input(self._data, input_)
        limits = self._data.server_hello.limits if self._data.server_hello is not None else None
        if limits is None or end_frame != limits.max_input_frames:
            return ()
        close = InputClosedEvent(
            input_.start.conversation_id,
            input_.start.input_id,
            end_frame,
            InputCloseReason.MAX_DURATION,
        )
        self._data = self._terminalize_input(self._data, input_, close)
        return (close,)

    def _response_input_ready(self, input_id: InputId) -> bool:
        input_ = self._find_input(self._data, input_id)
        conversation = self._data.conversation
        return (
            input_ is not None
            and conversation is not None
            and conversation.conversation_id == input_.start.conversation_id
            and conversation.input_id == input_id
            and conversation.cancel is None
            and conversation.failure is None
            and input_.terminal is not None
            and input_.transcript_final is not None
            and input_.response_id is None
        )

    @property
    def state(self) -> ProtocolStateSnapshot:
        conversation = self._data.conversation
        state = conversation.state if conversation is not None else None
        return ProtocolStateSnapshot(
            connection_state=self._data.connection_state,
            conversation_id=conversation.conversation_id if conversation is not None else None,
            input_id=conversation.input_id if conversation is not None else None,
            response_id=conversation.response_id if conversation is not None else None,
            output_id=conversation.output_id if conversation is not None else None,
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
        if isinstance(message, InputStartedEvent):
            return self._start_input(data, message)
        if isinstance(message, InputAudioEvent):
            return self._accept_input_audio(data, message)
        if isinstance(message, InputAbortedEvent):
            return self._abort_input(data, message)
        if isinstance(message, InputClosedEvent):
            return self._close_input(data, message)
        if isinstance(message, TranscriptUpdateEvent):
            return self._accept_transcript_update(data, message)
        if isinstance(message, TranscriptFinalEvent):
            return self._accept_transcript_final(data, message)
        if isinstance(message, ResponseStartedEvent):
            return self._start_response(data, message)
        if isinstance(message, ResponseTextDeltaEvent):
            return self._accept_response_text_delta(data, message)
        if isinstance(message, ResponseTextFinalEvent):
            return self._accept_response_text_final(data, message)
        if isinstance(message, OutputStartedEvent):
            return self._start_output(data, message)
        if isinstance(message, OutputAudioEvent):
            return self._accept_output_audio(data, message)
        if isinstance(message, OutputEndedEvent):
            return self._end_output(data, message)
        if isinstance(message, ResponseEndedEvent):
            return self._end_response(data, message)
        if isinstance(message, ResponseCancelledEvent):
            return self._cancel_response(data, message)
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

    def _start_input(
        self, data: _ValidatorData, message: InputStartedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        conversation = self._require_conversation(data, message.conversation_id)
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("terminal conversation cannot start an input")
        if conversation.expected_end_reason is not None:
            raise ProtocolViolation("conversation is awaiting its terminal end")
        if conversation.response_id is not None:
            raise ProtocolViolation("conversation already has a current response")
        if conversation.input_id is not None:
            raise ProtocolViolation("conversation already has a current input")
        has_conversation_input = any(item.start.conversation_id == message.conversation_id for item in data.inputs)
        if not has_conversation_input and message.reason is not InputStartReason.ACTIVATION:
            raise ProtocolViolation("the first input must start from activation")
        if has_conversation_input and message.reason is InputStartReason.ACTIVATION:
            raise ProtocolViolation("activation is only valid for the first input")
        if message.reason is InputStartReason.BARGE_IN:
            raise ProtocolViolation("barge-in requires a current response")
        allocator = data.input_ids.observe(message.input_id)
        input_ = _Input(message)
        updated = replace(
            data,
            input_ids=allocator,
            inputs=(*data.inputs, input_),
            conversation=replace(conversation, input_id=message.input_id),
        )
        return updated, self._result(message)

    def _accept_input_audio(
        self, data: _ValidatorData, message: InputAudioEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if self._role is EndpointRole.CLIENT and input_.failure is not None:
            raise ProtocolViolation("failed input cannot send more audio")
        frame_count = len(message.audio) // 2
        if message.start_frame != input_.received_end_frame:
            raise ProtocolViolation("input audio must be exactly contiguous")
        end_frame = message.start_frame + frame_count

        limits = data.server_hello.limits if data.server_hello is not None else None
        if limits is None:
            raise ProtocolViolation("input audio requires negotiated limits")
        if frame_count > limits.max_input_audio_frames:
            raise ProtocolViolation("input audio exceeds the negotiated message limit")
        if input_.terminal is not None and end_frame > limits.max_input_frames:
            raise ProtocolViolation("late input audio exceeds the negotiated aggregate limit")

        if input_.terminal is not None:
            if not isinstance(input_.terminal, InputClosedEvent):
                raise ProtocolViolation("audio is not legal after input abort")
            updated_input = replace(
                input_,
                received_end_frame=end_frame,
                received_boundaries=(*input_.received_boundaries, end_frame),
            )
            return self._replace_input(data, updated_input), self._result(message, dispatch=False)

        if end_frame > limits.max_input_frames:
            if self._role is EndpointRole.CLIENT:
                raise ProtocolViolation("input audio exceeds the negotiated aggregate limit")
            error = ErrorEvent(
                ErrorScope.INPUT,
                ErrorCode.INPUT_TOO_LONG,
                True,
                message.conversation_id,
                message.input_id,
            )
            return self._fail_input(data, input_, error, source=message)

        updated_input = replace(
            input_,
            received_end_frame=end_frame,
            received_boundaries=(*input_.received_boundaries, end_frame),
        )
        return self._replace_input(data, updated_input), self._result(message)

    def _abort_input(
        self, data: _ValidatorData, message: InputAbortedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if self._role is EndpointRole.CLIENT and input_.failure is not None:
            raise ProtocolViolation("failed input cannot be aborted")
        if input_.terminal is not None:
            if input_.terminal == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting input terminal event")
        self._require_conversation(data, message.conversation_id)
        return self._terminalize_input(data, input_, message), self._result(message, terminal=True)

    def _close_input(self, data: _ValidatorData, message: InputClosedEvent) -> tuple[_ValidatorData, _TransitionResult]:
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if input_.terminal is not None:
            if input_.terminal == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting input terminal event")
        self._require_conversation(data, message.conversation_id)
        if message.accepted_end_frame > input_.received_end_frame:
            raise ProtocolViolation("input close cannot accept frames that were not received")
        if message.accepted_end_frame != 0 and message.accepted_end_frame not in input_.received_boundaries:
            raise ProtocolViolation("input close boundary must end a received audio chunk")
        if self._role is EndpointRole.SERVER and message.accepted_end_frame != input_.committed_end_frame:
            raise ProtocolViolation("input close must use the callback-committed boundary")
        if message.reason is not InputCloseReason.FAILED and input_.failure is not None:
            raise ProtocolViolation("failed input must close as failed")
        input_ = replace(input_, committed_end_frame=message.accepted_end_frame)
        return self._terminalize_input(data, input_, message), self._result(message, terminal=True)

    def _accept_transcript_update(
        self, data: _ValidatorData, message: TranscriptUpdateEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        self._require_live_conversation(data, message.conversation_id)
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if input_.terminal is not None or input_.failure is not None:
            raise ProtocolViolation("transcript updates are legal only while input is open")
        previous = input_.transcript_update
        if previous is None:
            if message.revision != 1:
                raise ProtocolViolation("first transcript revision must be 1")
        else:
            if message == previous:
                return data, self._result(message, dispatch=False)
            if message.revision != previous.revision + 1:
                raise ProtocolViolation("transcript revision must increment exactly by one")
        updated = self._replace_input(data, replace(input_, transcript_update=message))
        return updated, self._result(message)

    def _accept_transcript_final(
        self, data: _ValidatorData, message: TranscriptFinalEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        self._require_live_conversation(data, message.conversation_id)
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if input_.transcript_final is not None:
            if input_.transcript_final == message:
                return data, self._result(message, dispatch=False)
            raise ProtocolViolation("conflicting final transcript")
        terminal = input_.terminal
        if terminal is None:
            raise ProtocolViolation("final transcript requires a terminal input")
        empty_allowed = (
            input_.failure is not None
            or isinstance(terminal, InputAbortedEvent)
            or (
                isinstance(terminal, InputClosedEvent)
                and terminal.reason in (InputCloseReason.NO_SPEECH, InputCloseReason.FAILED)
            )
        )
        if not message.text and not empty_allowed:
            raise ProtocolViolation("empty final transcript is not valid for this input terminal")
        updated = self._replace_input(data, replace(input_, transcript_final=message))
        return updated, self._result(message)

    def _start_response(
        self, data: _ValidatorData, message: ResponseStartedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        conversation = self._require_live_conversation(data, message.conversation_id)
        if conversation.response_id is not None:
            raise ProtocolViolation("conversation already has a current response")
        if conversation.expected_end_reason is not None:
            raise ProtocolViolation("conversation is awaiting its terminal end")
        input_ = self._require_input(data, message.input_id, message.conversation_id)
        if conversation.input_id != message.input_id:
            raise ProtocolViolation("response input is not the current conversation input")
        if input_.terminal is None or input_.transcript_final is None:
            raise ProtocolViolation("response requires a terminal input and final transcript")
        if input_.response_id is not None:
            raise ProtocolViolation("input already has a response")

        allocator = data.response_ids.observe(message.response_id)
        response = _Response(message)
        updated_input = replace(input_, response_id=message.response_id)
        updated_conversation = replace(conversation, response_id=message.response_id)
        updated = replace(
            self._replace_input(data, updated_input),
            response_ids=allocator,
            responses=(*data.responses, response),
            conversation=updated_conversation,
        )
        return updated, self._result(message)

    def _accept_response_text_delta(
        self, data: _ValidatorData, message: ResponseTextDeltaEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response = self._require_response(data, message.response_id, message.conversation_id)
        stale = isinstance(response.terminal, ResponseCancelledEvent)
        if response.end is not None:
            raise ProtocolViolation("response text is not legal after response end")
        if response.terminal is not None and not stale:
            raise ProtocolViolation("response text is not legal after response termination")
        if response.text_final is not None:
            raise ProtocolViolation("response text delta is not legal after text final")
        if message.sequence != response.next_text_sequence:
            raise ProtocolViolation("response text sequence must increment exactly from zero")
        updated = replace(
            response,
            next_text_sequence=response.next_text_sequence + 1,
            text=response.text + message.text,
        )
        return self._replace_response(data, updated), self._result(message, dispatch=not stale)

    def _accept_response_text_final(
        self, data: _ValidatorData, message: ResponseTextFinalEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response = self._require_response(data, message.response_id, message.conversation_id)
        stale = isinstance(response.terminal, ResponseCancelledEvent)
        if response.end is not None:
            raise ProtocolViolation("response text final is not legal after response end")
        if response.terminal is not None and not stale:
            raise ProtocolViolation("response text final is not legal after response termination")
        if response.text_final is not None:
            raise ProtocolViolation("response text final may appear at most once")
        if response.next_text_sequence and message.text != response.text:
            raise ProtocolViolation("response text final must equal concatenated deltas")
        updated = replace(response, text_final=message)
        return self._replace_response(data, updated), self._result(message, dispatch=not stale)

    def _start_output(
        self, data: _ValidatorData, message: OutputStartedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response = self._require_response(data, message.response_id, message.conversation_id)
        if response.terminal is not None:
            raise ProtocolViolation("output cannot start after response termination")
        if response.output is not None:
            raise ProtocolViolation("response may have at most one output")
        allocator = data.output_ids.observe(message.output_id)
        output = _Output(message)
        conversation = self._require_conversation(data, message.conversation_id)
        updated_response = replace(response, output=output)
        updated = replace(
            self._replace_response(data, updated_response),
            output_ids=allocator,
            conversation=replace(conversation, output_id=message.output_id),
        )
        return updated, self._result(message)

    def _accept_output_audio(
        self, data: _ValidatorData, message: OutputAudioEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response, output = self._require_output(
            data,
            message.response_id,
            message.output_id,
            message.conversation_id,
        )
        stale = isinstance(response.terminal, ResponseCancelledEvent)
        if response.terminal is not None and not stale:
            raise ProtocolViolation("output audio is not legal after response end")
        if isinstance(output.terminal, OutputEndedEvent):
            raise ProtocolViolation("output audio is not legal after output end")
        if message.start_frame != output.received_end_frame:
            raise ProtocolViolation("output audio must be exactly contiguous")
        limits = data.server_hello.limits if data.server_hello is not None else None
        if limits is None:
            raise ProtocolViolation("output audio requires negotiated limits")
        frame_count = len(message.audio) // 2
        if frame_count > limits.max_output_audio_frames:
            raise ProtocolViolation("output audio exceeds the negotiated message limit")
        updated_output = replace(output, received_end_frame=message.start_frame + frame_count)
        updated_response = replace(response, output=updated_output)
        return self._replace_response(data, updated_response), self._result(message, dispatch=not stale)

    def _end_output(self, data: _ValidatorData, message: OutputEndedEvent) -> tuple[_ValidatorData, _TransitionResult]:
        response, output = self._require_output(
            data,
            message.response_id,
            message.output_id,
            message.conversation_id,
        )
        stale = isinstance(response.terminal, ResponseCancelledEvent)
        if response.terminal is not None and not stale:
            raise ProtocolViolation("output cannot end after response end")
        if isinstance(output.terminal, OutputEndedEvent):
            if output.terminal == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting output end")
        if message.total_frames == 0 or output.received_end_frame == 0:
            raise ProtocolViolation("output must contain at least one audio frame")
        if message.total_frames != output.received_end_frame:
            raise ProtocolViolation("output total must equal received audio frames")
        if stale:
            return data, self._result(message, dispatch=False, terminal=True)

        updated_response = replace(response, output=replace(output, terminal=message))
        tombstone = ProtocolTombstone(
            ProtocolObjectKind.OUTPUT,
            message.conversation_id,
            None,
            message.response_id,
            message.output_id,
            "ended",
        )
        updated = replace(
            self._replace_response(data, updated_response),
            tombstones=(*data.tombstones, tombstone),
        )
        return updated, self._result(message, terminal=True)

    def _end_response(
        self, data: _ValidatorData, message: ResponseEndedEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response = self._require_response(data, message.response_id, message.conversation_id)
        if response.end is not None:
            if response.end == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting response terminal event")
        if response.terminal is not None:
            raise ProtocolViolation("cancelled response cannot end")
        output = response.output
        if output is not None and not isinstance(output.terminal, OutputEndedEvent):
            raise ProtocolViolation("response cannot end before its output")

        updated_response = replace(response, end=message, terminal=message)
        data = self._replace_response(data, updated_response)
        if output is not None:
            return data, self._result(message, terminal=True)

        conversation = self._require_conversation(data, message.conversation_id)
        expected_end = ConversationEndReason.COMPLETED if response.start.end_conversation else None
        tombstone = ProtocolTombstone(
            ProtocolObjectKind.RESPONSE,
            message.conversation_id,
            None,
            message.response_id,
            None,
            "completed",
        )
        updated = replace(
            data,
            conversation=replace(
                conversation,
                input_id=None,
                response_id=None,
                expected_end_reason=expected_end,
            ),
            tombstones=(*data.tombstones, tombstone),
        )
        return updated, self._result(message, terminal=True)

    def _cancel_response(
        self, data: _ValidatorData, message: ResponseCancelledEvent
    ) -> tuple[_ValidatorData, _TransitionResult]:
        response = self._require_response(data, message.response_id, message.conversation_id)
        conversation = self._require_conversation(data, message.conversation_id)
        if isinstance(response.terminal, ResponseCancelledEvent):
            if response.terminal == message:
                return data, self._result(message, dispatch=False, terminal=True)
            raise ProtocolViolation("conflicting response cancellation")
        if response.end is not None and response.output is None:
            raise ProtocolViolation("completed response cannot be cancelled")
        if conversation.cancel is not None and message.reason is not ResponseCancelReason.CONVERSATION_CANCELLED:
            raise ProtocolViolation("conversation cancellation requires matching response cancellation")

        tombstones = data.tombstones
        output = response.output
        if output is not None and output.terminal is None:
            output = replace(output, terminal=_CancelledOutput())
            tombstones = (
                *tombstones,
                ProtocolTombstone(
                    ProtocolObjectKind.OUTPUT,
                    message.conversation_id,
                    None,
                    message.response_id,
                    output.start.output_id,
                    "cancelled",
                ),
            )
        updated_response = replace(response, output=output, terminal=message)
        tombstones = (
            *tombstones,
            ProtocolTombstone(
                ProtocolObjectKind.RESPONSE,
                message.conversation_id,
                None,
                message.response_id,
                None,
                "cancelled",
            ),
        )
        expected_end = self._response_cancellation_end_reason(response, message.reason)
        updated = replace(
            self._replace_response(data, updated_response),
            conversation=replace(
                conversation,
                input_id=None,
                response_id=None,
                output_id=None,
                expected_end_reason=expected_end,
            ),
            tombstones=tombstones,
        )
        return updated, self._result(message, terminal=True)

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
        data = self._terminate_current_input(data, conversation)
        conversation = data.conversation
        assert conversation is not None
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
        current_input = self._current_input(data, conversation)
        if current_input is not None and current_input.terminal is None:
            raise ProtocolViolation("conversation cannot end with an open input")
        if conversation.response_id is not None:
            raise ProtocolViolation("conversation cannot end with a current response")
        if (
            current_input is not None
            and conversation.cancel is None
            and conversation.failure is None
            and current_input.transcript_final is None
        ):
            raise ProtocolViolation("conversation cannot end before the final transcript")
        if conversation.cancel is not None and message.reason.value != "cancelled":
            raise ProtocolViolation("cancelled conversation must end as cancelled")
        if conversation.failure is not None:
            expected = "idle_timeout" if conversation.failure.code is ErrorCode.IDLE_TIMEOUT else "server_failed"
            if message.reason.value != expected:
                raise ProtocolViolation(f"failed conversation must end as {expected}")
        if conversation.expected_end_reason is not None and message.reason is not conversation.expected_end_reason:
            raise ProtocolViolation(f"conversation must end as {conversation.expected_end_reason.value}")

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
        if conversation.expected_end_reason is not None:
            raise ProtocolViolation("conversation awaiting terminal end cannot change coarse state")

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
        response = self._current_response(data, conversation)
        input_ = self._current_input(data, conversation)
        if response is not None:
            if message.state is not CoarseState.RESPONDING:
                raise ProtocolViolation("current response requires responding state")
        elif input_ is not None:
            if input_.failure is not None:
                if input_.terminal is None or input_.transcript_final is None:
                    raise ProtocolViolation("failed input terminal sequence must complete before state changes")
                if message.state not in (CoarseState.PROCESSING, CoarseState.WAITING):
                    raise ProtocolViolation("failed input requires processing or waiting state")
            else:
                expected = CoarseState.LISTENING if input_.terminal is None else CoarseState.PROCESSING
                if message.state is not expected:
                    raise ProtocolViolation(f"current input requires {expected.value} state")
        input_id = None if input_ is not None and message.state is CoarseState.WAITING else conversation.input_id
        updated = replace(data, conversation=replace(conversation, state=message, input_id=input_id))
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
            if message.scope is ErrorScope.INPUT:
                conversation_id = message.conversation_id
                input_id = message.input_id
                assert conversation_id is not None and input_id is not None
                self._require_live_conversation(data, conversation_id)
                input_ = self._require_input(data, input_id, conversation_id)
                if self._role is EndpointRole.CLIENT:
                    if input_.failure is not None:
                        raise ProtocolViolation("input already has a fatal error")
                    updated = self._replace_input(data, replace(input_, failure=message))
                    return updated, self._result(
                        message,
                        cancellation_targets=(ProtocolObjectKind.INPUT,),
                        terminal=True,
                    )
                return self._fail_input(data, input_, message)
            raise ProtocolViolation("response errors are not implemented by this validator stage")
        conversation = data.conversation
        if conversation is None or message.conversation_id != conversation.conversation_id:
            raise ProtocolViolation("conversation error targets no open conversation")
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("conversation is already terminal")
        data = self._terminate_current_input(data, conversation)
        conversation = data.conversation
        assert conversation is not None
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
            from_client = isinstance(
                message,
                (
                    ClientHello,
                    ConversationStartedEvent,
                    InputStartedEvent,
                    InputAudioEvent,
                    InputAbortedEvent,
                    ConversationCancelledEvent,
                ),
            )
            dispatch = from_client is (self._role is EndpointRole.SERVER)
        return _TransitionResult(
            dispatch=dispatch,
            cancellation_targets=cancellation_targets,
            terminal=terminal,
        )

    def _fail_input(
        self,
        data: _ValidatorData,
        input_: _Input,
        error: ErrorEvent,
        *,
        source: Message | None = None,
    ) -> tuple[_ValidatorData, _TransitionResult]:
        if input_.failure is not None:
            raise ProtocolViolation("input already has a fatal error")
        terminal = input_.terminal
        outbound: list[Message] = []
        if terminal is None:
            terminal = InputClosedEvent(
                input_.start.conversation_id,
                input_.start.input_id,
                input_.committed_end_frame,
                InputCloseReason.FAILED,
            )
            outbound.append(terminal)
            data = self._terminalize_input(data, input_, terminal)
            input_ = self._require_input(data, input_.start.input_id, input_.start.conversation_id)
        final = input_.transcript_final
        if final is None:
            final = TranscriptFinalEvent(input_.start.conversation_id, input_.start.input_id, "")
            outbound.append(final)
        updated_input = replace(input_, failure=error, transcript_final=final)
        updated = self._replace_input(data, updated_input)
        return updated, _TransitionResult(
            dispatch=False if source is not None else self._result(error).dispatch,
            outbound=tuple((error, *outbound)) if source is not None else tuple(outbound),
            cancellation_targets=(ProtocolObjectKind.INPUT,),
            terminal=True,
        )

    def _terminalize_input(
        self, data: _ValidatorData, input_: _Input, terminal: InputClosedEvent | InputAbortedEvent
    ) -> _ValidatorData:
        terminal_state = "closed" if isinstance(terminal, InputClosedEvent) else "aborted"
        tombstone = ProtocolTombstone(
            ProtocolObjectKind.INPUT,
            input_.start.conversation_id,
            input_.start.input_id,
            None,
            None,
            terminal_state,
        )
        return self._replace_input(
            replace(data, tombstones=(*data.tombstones, tombstone)),
            replace(input_, terminal=terminal),
        )

    def _terminate_current_input(self, data: _ValidatorData, conversation: _Conversation) -> _ValidatorData:
        input_ = self._current_input(data, conversation)
        if input_ is None or input_.terminal is not None:
            return data
        tombstone = ProtocolTombstone(
            ProtocolObjectKind.INPUT,
            conversation.conversation_id,
            input_.start.input_id,
            None,
            None,
            "aborted",
        )
        return self._replace_input(
            replace(data, tombstones=(*data.tombstones, tombstone)),
            replace(input_, terminal=_CancelledInput()),
        )

    @staticmethod
    def _response_cancellation_end_reason(
        response: _Response, reason: ResponseCancelReason
    ) -> ConversationEndReason | None:
        if reason is ResponseCancelReason.CONVERSATION_CANCELLED:
            return ConversationEndReason.CANCELLED
        if reason is ResponseCancelReason.SHUTDOWN:
            return ConversationEndReason.CANCELLED
        if reason is ResponseCancelReason.PLAYBACK_FAILED:
            return ConversationEndReason.PLAYBACK_FAILED
        if not response.start.end_conversation:
            return None
        if reason is ResponseCancelReason.LOCAL_CANCEL:
            return ConversationEndReason.CANCELLED
        if reason in (
            ResponseCancelReason.GENERATION_FAILED,
            ResponseCancelReason.TTS_FAILED,
            ResponseCancelReason.OVERFLOW,
        ):
            return ConversationEndReason.SERVER_FAILED
        return None

    @staticmethod
    def _require_conversation(data: _ValidatorData, conversation_id: ConversationId) -> _Conversation:
        conversation = data.conversation
        if conversation is None or conversation.conversation_id != conversation_id:
            raise ProtocolViolation("message targets no open conversation")
        return conversation

    @classmethod
    def _require_live_conversation(cls, data: _ValidatorData, conversation_id: ConversationId) -> _Conversation:
        conversation = cls._require_conversation(data, conversation_id)
        if conversation.cancel is not None or conversation.failure is not None:
            raise ProtocolViolation("terminal conversation cannot accept child events")
        return conversation

    @staticmethod
    def _find_input(data: _ValidatorData, input_id: InputId) -> _Input | None:
        for input_ in reversed(data.inputs):
            if input_.start.input_id == input_id:
                return input_
        return None

    def _require_input(self, data: _ValidatorData, input_id: InputId, conversation_id: ConversationId) -> _Input:
        input_ = self._find_input(data, input_id)
        if input_ is None or input_.start.conversation_id != conversation_id:
            raise ProtocolViolation("message targets no known input in this conversation")
        return input_

    def _current_input(self, data: _ValidatorData, conversation: _Conversation) -> _Input | None:
        if conversation.input_id is None:
            return None
        return self._find_input(data, conversation.input_id)

    @staticmethod
    def _find_response(data: _ValidatorData, response_id: ResponseId) -> _Response | None:
        for response in reversed(data.responses):
            if response.start.response_id == response_id:
                return response
        return None

    def _require_response(
        self,
        data: _ValidatorData,
        response_id: ResponseId,
        conversation_id: ConversationId,
    ) -> _Response:
        response = self._find_response(data, response_id)
        if response is None or response.start.conversation_id != conversation_id:
            raise ProtocolViolation("message targets no known response in this conversation")
        return response

    def _current_response(self, data: _ValidatorData, conversation: _Conversation) -> _Response | None:
        if conversation.response_id is None:
            return None
        return self._find_response(data, conversation.response_id)

    def _require_output(
        self,
        data: _ValidatorData,
        response_id: ResponseId,
        output_id: OutputId,
        conversation_id: ConversationId,
    ) -> tuple[_Response, _Output]:
        response = self._require_response(data, response_id, conversation_id)
        output = response.output
        if output is None or output.start.output_id != output_id:
            raise ProtocolViolation("message targets no known output of this response")
        return response, output

    @staticmethod
    def _replace_input(data: _ValidatorData, updated: _Input) -> _ValidatorData:
        inputs = tuple(updated if item.start.input_id == updated.start.input_id else item for item in data.inputs)
        return replace(data, inputs=inputs)

    @staticmethod
    def _replace_response(data: _ValidatorData, updated: _Response) -> _ValidatorData:
        responses = tuple(
            updated if item.start.response_id == updated.start.response_id else item for item in data.responses
        )
        return replace(data, responses=responses)

    @staticmethod
    def _find_by_id[T: ConversationCancelledEvent | ConversationEndedEvent | StateEvent](
        events: tuple[T, ...], conversation_id: ConversationId
    ) -> T | None:
        for event in reversed(events):
            if event.conversation_id == conversation_id:
                return event
        return None
