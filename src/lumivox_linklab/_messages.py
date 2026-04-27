from typing import Literal, ClassVar
from dataclasses import dataclass

from ._enums import (
    ErrorCode,
    ErrorScope,
    CoarseState,
    ConnectionState,
    InputAbortReason,
    InputCloseReason,
    InputStartReason,
    PlaybackPosition,
    ResponseCancelReason,
    ConversationEndReason,
    PlaybackInterruptReason,
    ConversationCancelReason,
)
from ._config import (
    _REQUIRED_CAPABILITIES,
    AudioFormat,
    ConnectionLimits,
    _validate_output_formats,
)
from ._values import InputId, OutputId, ResponseId, ConversationId, ReadableBuffer

_MAX_ID = 4_294_967_295
_MAX_FRAME = 9_223_372_036_854_775_807
_MAX_TEXT_BYTES = 65_536

_CONNECTION_ERROR_CODES = frozenset(
    {
        ErrorCode.MALFORMED_MESSAGE,
        ErrorCode.MESSAGE_TOO_LARGE,
        ErrorCode.UNKNOWN_MESSAGE,
        ErrorCode.UNSUPPORTED_VERSION,
        ErrorCode.CAPABILITY_MISMATCH,
        ErrorCode.FORMAT_MISMATCH,
        ErrorCode.HANDSHAKE_TIMEOUT,
        ErrorCode.PROTOCOL_STATE,
        ErrorCode.ID_EXHAUSTED,
        ErrorCode.PEER_UNRESPONSIVE,
    }
)
_CONVERSATION_ERROR_CODES = frozenset({ErrorCode.CONVERSATION_FAILED, ErrorCode.IDLE_TIMEOUT})
_INPUT_ERROR_CODES = frozenset(
    {
        ErrorCode.INPUT_DISCONTINUITY,
        ErrorCode.INPUT_OVERFLOW,
        ErrorCode.INPUT_TOO_LONG,
        ErrorCode.CAPTURE_FAILED,
        ErrorCode.STT_FAILED,
        ErrorCode.PROCESSING_TIMEOUT,
    }
)
_RESPONSE_ERROR_CODES = frozenset(
    {
        ErrorCode.GENERATION_FAILED,
        ErrorCode.TTS_FAILED,
        ErrorCode.PLAYBACK_FAILED,
        ErrorCode.OUTPUT_OVERFLOW,
        ErrorCode.RESPONSE_CANCELLED,
    }
)
_ERROR_CODES_BY_SCOPE = {
    ErrorScope.CONNECTION: _CONNECTION_ERROR_CODES,
    ErrorScope.CONVERSATION: _CONVERSATION_ERROR_CODES,
    ErrorScope.INPUT: _INPUT_ERROR_CODES,
    ErrorScope.RESPONSE: _RESPONSE_ERROR_CODES,
}


def _require_exact_type(name: str, value: object, expected: type[object]) -> None:
    if type(value) is not expected:
        raise TypeError(f"{name} must be {expected.__name__}")


def _require_enum(name: str, value: object, expected: type[object]) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be {expected.__name__}")


def _require_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")


def _require_id(name: str, value: object) -> None:
    _require_int(name, value, 1, _MAX_ID)


def _require_frame(name: str, value: object) -> None:
    _require_int(name, value, 0, _MAX_FRAME)


def _require_string(name: str, value: object, minimum: int, maximum: int) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a str")
    size = len(value.encode("utf-8"))
    if not minimum <= size <= maximum:
        raise ValueError(f"{name} must contain {minimum}..{maximum} UTF-8 bytes")


def _require_optional_short_string(name: str, value: object) -> None:
    if value is not None:
        _require_string(name, value, 1, 64)


def _copy_pcm(name: str, value: ReadableBuffer, max_frames: int) -> bytes:
    try:
        view = memoryview(value)
    except TypeError as error:
        raise TypeError(f"{name} must support the buffer protocol") from error
    if not view.contiguous:
        raise ValueError(f"{name} must be contiguous")
    try:
        result = view.cast("B").tobytes()
    except TypeError as error:
        raise ValueError(f"{name} must be a contiguous byte-addressable buffer") from error
    if not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) % 2:
        raise ValueError(f"{name} must contain frame-aligned PCM S16LE")
    if len(result) // 2 > max_frames:
        raise ValueError(f"{name} must not exceed {max_frames} frames")
    return result


def _validate_audio_range(start_frame: int, audio: bytes) -> None:
    _require_frame("start_frame", start_frame)
    if start_frame + len(audio) // 2 > _MAX_FRAME:
        raise ValueError("audio range end exceeds the maximum frame")


def _validate_capabilities(capabilities: object) -> None:
    if type(capabilities) is not tuple:
        raise TypeError("capabilities must be a tuple")
    if not 1 <= len(capabilities) <= 32:
        raise ValueError("capabilities must contain 1..32 entries")
    for capability in capabilities:
        _require_string("capability", capability, 1, 64)
    if len(set(capabilities)) != len(capabilities):
        raise ValueError("capabilities must be unique")
    if capabilities != tuple(sorted(capabilities, key=str.encode)):
        raise ValueError("capabilities must be sorted by UTF-8 bytes")
    if not set(_REQUIRED_CAPABILITIES).issubset(capabilities):
        raise ValueError("capabilities must include all required capabilities")


def _validate_client_output_formats(output_formats: object) -> None:
    _validate_output_formats(output_formats, canonical_order=True)


@dataclass(frozen=True, slots=True)
class ClientHello:
    type: ClassVar[Literal["hello"]] = "hello"

    version: int
    capabilities: tuple[str, ...]
    input_format: AudioFormat
    output_formats: tuple[AudioFormat, ...]
    agent: str | None = None

    def __post_init__(self) -> None:
        _require_int("version", self.version, 1, 1)
        _validate_capabilities(self.capabilities)
        if not isinstance(self.input_format, AudioFormat):
            raise TypeError("input_format must be AudioFormat")
        if self.input_format != AudioFormat("pcm_s16le", 16_000, 1):
            raise ValueError("input_format must be PCM S16LE, 16000 Hz, mono")
        _validate_client_output_formats(self.output_formats)
        _require_optional_short_string("agent", self.agent)


@dataclass(frozen=True, slots=True)
class ServerHello:
    type: ClassVar[Literal["hello"]] = "hello"

    version: int
    capabilities: tuple[str, ...]
    output_format: AudioFormat
    limits: ConnectionLimits
    agent: str | None = None

    def __post_init__(self) -> None:
        _require_int("version", self.version, 1, 1)
        _validate_capabilities(self.capabilities)
        if not isinstance(self.output_format, AudioFormat):
            raise TypeError("output_format must be AudioFormat")
        if not isinstance(self.limits, ConnectionLimits):
            raise TypeError("limits must be ConnectionLimits")
        output_frames = self.limits.max_output_audio_frames
        rate = self.output_format.sample_rate_hz
        if not rate // 100 <= output_frames <= rate // 10:
            raise ValueError("max_output_audio_frames is invalid for output_format")
        _require_optional_short_string("agent", self.agent)


@dataclass(frozen=True, slots=True)
class ConversationStartedEvent:
    type: ClassVar[Literal["conversation.start"]] = "conversation.start"

    conversation_id: ConversationId
    activation: Literal["wake_word"]
    wake_word: str | None = None

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_string("activation", self.activation, 1, 64)
        if self.activation != "wake_word":
            raise ValueError("activation must be wake_word")
        _require_optional_short_string("wake_word", self.wake_word)


@dataclass(frozen=True, slots=True)
class InputStartedEvent:
    type: ClassVar[Literal["input.start"]] = "input.start"

    conversation_id: ConversationId
    input_id: InputId
    reason: InputStartReason
    generation: int
    interrupts_response_id: ResponseId | None = None

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_enum("reason", self.reason, InputStartReason)
        _require_int("generation", self.generation, 0, _MAX_ID)
        if self.reason is InputStartReason.BARGE_IN:
            if self.interrupts_response_id is None:
                raise ValueError("interrupts_response_id is required for barge_in")
            _require_id("interrupts_response_id", self.interrupts_response_id)
        elif self.interrupts_response_id is not None:
            raise ValueError("interrupts_response_id is only valid for barge_in")


@dataclass(frozen=True, slots=True)
class InputAudioEvent:
    type: ClassVar[Literal["input.audio"]] = "input.audio"

    conversation_id: ConversationId
    input_id: InputId
    start_frame: int
    speech: bool
    audio: bytes

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_exact_type("speech", self.speech, bool)
        audio = _copy_pcm("audio", self.audio, 1_600)
        _validate_audio_range(self.start_frame, audio)
        object.__setattr__(self, "audio", audio)


@dataclass(frozen=True, slots=True)
class InputAbortedEvent:
    type: ClassVar[Literal["input.abort"]] = "input.abort"

    conversation_id: ConversationId
    input_id: InputId
    reason: InputAbortReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_enum("reason", self.reason, InputAbortReason)


@dataclass(frozen=True, slots=True)
class PlaybackFinishedEvent:
    type: ClassVar[Literal["playback.finished"]] = "playback.finished"

    conversation_id: ConversationId
    response_id: ResponseId
    output_id: OutputId
    played_frames: int

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("output_id", self.output_id)
        _require_frame("played_frames", self.played_frames)


@dataclass(frozen=True, slots=True)
class PlaybackInterruptedEvent:
    type: ClassVar[Literal["playback.interrupted"]] = "playback.interrupted"

    conversation_id: ConversationId
    response_id: ResponseId
    output_id: OutputId
    played_frames: int
    position: PlaybackPosition
    reason: PlaybackInterruptReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("output_id", self.output_id)
        _require_frame("played_frames", self.played_frames)
        _require_enum("position", self.position, PlaybackPosition)
        _require_enum("reason", self.reason, PlaybackInterruptReason)


@dataclass(frozen=True, slots=True)
class ConversationCancelledEvent:
    type: ClassVar[Literal["conversation.cancel"]] = "conversation.cancel"

    conversation_id: ConversationId
    reason: ConversationCancelReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_enum("reason", self.reason, ConversationCancelReason)


@dataclass(frozen=True, slots=True)
class StateEvent:
    type: ClassVar[Literal["state"]] = "state"

    conversation_id: ConversationId
    revision: int
    state: CoarseState
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_int("revision", self.revision, 1, _MAX_ID)
        _require_enum("state", self.state, CoarseState)
        _require_optional_short_string("reason", self.reason)


@dataclass(frozen=True, slots=True)
class InputClosedEvent:
    type: ClassVar[Literal["input.closed"]] = "input.closed"

    conversation_id: ConversationId
    input_id: InputId
    accepted_end_frame: int
    reason: InputCloseReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_frame("accepted_end_frame", self.accepted_end_frame)
        _require_enum("reason", self.reason, InputCloseReason)


@dataclass(frozen=True, slots=True)
class TranscriptUpdateEvent:
    type: ClassVar[Literal["transcript.update"]] = "transcript.update"

    conversation_id: ConversationId
    input_id: InputId
    revision: int
    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_int("revision", self.revision, 1, _MAX_ID)
        _require_string("text", self.text, 1, _MAX_TEXT_BYTES)
        _require_optional_short_string("language", self.language)


@dataclass(frozen=True, slots=True)
class TranscriptFinalEvent:
    type: ClassVar[Literal["transcript.final"]] = "transcript.final"

    conversation_id: ConversationId
    input_id: InputId
    text: str
    language: str | None = None

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("input_id", self.input_id)
        _require_string("text", self.text, 0, _MAX_TEXT_BYTES)
        _require_optional_short_string("language", self.language)


@dataclass(frozen=True, slots=True)
class ResponseStartedEvent:
    type: ClassVar[Literal["response.start"]] = "response.start"

    conversation_id: ConversationId
    response_id: ResponseId
    input_id: InputId
    end_conversation: bool

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("input_id", self.input_id)
        _require_exact_type("end_conversation", self.end_conversation, bool)


@dataclass(frozen=True, slots=True)
class ResponseTextDeltaEvent:
    type: ClassVar[Literal["response.text.delta"]] = "response.text.delta"

    conversation_id: ConversationId
    response_id: ResponseId
    sequence: int
    text: str

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_int("sequence", self.sequence, 0, _MAX_ID)
        _require_string("text", self.text, 1, _MAX_TEXT_BYTES)


@dataclass(frozen=True, slots=True)
class ResponseTextFinalEvent:
    type: ClassVar[Literal["response.text.final"]] = "response.text.final"

    conversation_id: ConversationId
    response_id: ResponseId
    text: str

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_string("text", self.text, 1, _MAX_TEXT_BYTES)


@dataclass(frozen=True, slots=True)
class OutputStartedEvent:
    type: ClassVar[Literal["output.start"]] = "output.start"

    conversation_id: ConversationId
    response_id: ResponseId
    output_id: OutputId

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("output_id", self.output_id)


@dataclass(frozen=True, slots=True)
class OutputAudioEvent:
    type: ClassVar[Literal["output.audio"]] = "output.audio"

    conversation_id: ConversationId
    response_id: ResponseId
    output_id: OutputId
    start_frame: int
    audio: bytes

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("output_id", self.output_id)
        audio = _copy_pcm("audio", self.audio, 4_800)
        _validate_audio_range(self.start_frame, audio)
        object.__setattr__(self, "audio", audio)


@dataclass(frozen=True, slots=True)
class OutputEndedEvent:
    type: ClassVar[Literal["output.end"]] = "output.end"

    conversation_id: ConversationId
    response_id: ResponseId
    output_id: OutputId
    total_frames: int

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_id("output_id", self.output_id)
        _require_frame("total_frames", self.total_frames)


@dataclass(frozen=True, slots=True)
class ResponseEndedEvent:
    type: ClassVar[Literal["response.end"]] = "response.end"

    conversation_id: ConversationId
    response_id: ResponseId

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)


@dataclass(frozen=True, slots=True)
class ResponseCancelledEvent:
    type: ClassVar[Literal["response.cancelled"]] = "response.cancelled"

    conversation_id: ConversationId
    response_id: ResponseId
    reason: ResponseCancelReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_id("response_id", self.response_id)
        _require_enum("reason", self.reason, ResponseCancelReason)


@dataclass(frozen=True, slots=True)
class ConversationEndedEvent:
    type: ClassVar[Literal["conversation.end"]] = "conversation.end"

    conversation_id: ConversationId
    reason: ConversationEndReason

    def __post_init__(self) -> None:
        _require_id("conversation_id", self.conversation_id)
        _require_enum("reason", self.reason, ConversationEndReason)


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    type: ClassVar[Literal["error"]] = "error"

    scope: ErrorScope
    code: ErrorCode
    fatal: bool
    conversation_id: ConversationId | None = None
    input_id: InputId | None = None
    response_id: ResponseId | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        _require_enum("scope", self.scope, ErrorScope)
        _require_enum("code", self.code, ErrorCode)
        _require_exact_type("fatal", self.fatal, bool)
        if not self.fatal:
            raise ValueError("fatal must be true")
        if self.code not in _ERROR_CODES_BY_SCOPE[self.scope]:
            raise ValueError("code is not valid for scope")

        required = {
            ErrorScope.CONNECTION: (False, False, False),
            ErrorScope.CONVERSATION: (True, False, False),
            ErrorScope.INPUT: (True, True, False),
            ErrorScope.RESPONSE: (True, False, True),
        }[self.scope]
        values = (self.conversation_id, self.input_id, self.response_id)
        names = ("conversation_id", "input_id", "response_id")
        for name, value, is_required in zip(names, values, required, strict=True):
            if is_required:
                if value is None:
                    raise ValueError(f"{name} is required for {self.scope.value} scope")
                _require_id(name, value)
            elif value is not None:
                raise ValueError(f"{name} is forbidden for {self.scope.value} scope")
        if self.message is not None:
            _require_string("message", self.message, 1, 512)


@dataclass(frozen=True, slots=True)
class ConnectionStateEvent:
    state: ConnectionState
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_enum("state", self.state, ConnectionState)
        if self.reason is not None:
            _require_exact_type("reason", self.reason, str)


type ClientMessage = (
    ClientHello
    | ConversationStartedEvent
    | InputStartedEvent
    | InputAudioEvent
    | InputAbortedEvent
    | PlaybackFinishedEvent
    | PlaybackInterruptedEvent
    | ConversationCancelledEvent
)

type ServerMessage = (
    ServerHello
    | StateEvent
    | InputClosedEvent
    | TranscriptUpdateEvent
    | TranscriptFinalEvent
    | ResponseStartedEvent
    | ResponseTextDeltaEvent
    | ResponseTextFinalEvent
    | OutputStartedEvent
    | OutputAudioEvent
    | OutputEndedEvent
    | ResponseEndedEvent
    | ResponseCancelledEvent
    | ConversationEndedEvent
    | ErrorEvent
)

type Message = ClientMessage | ServerMessage
