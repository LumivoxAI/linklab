from enum import StrEnum


class InputStartReason(StrEnum):
    ACTIVATION = "activation"
    SPEECH = "speech"
    BARGE_IN = "barge_in"


class InputAbortReason(StrEnum):
    DISCONTINUITY = "discontinuity"
    OVERFLOW = "overflow"
    CAPTURE_FAILED = "capture_failed"
    SHUTDOWN = "shutdown"


class PlaybackPosition(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"


class PlaybackInterruptReason(StrEnum):
    BARGE_IN = "barge_in"
    LOCAL_CANCEL = "local_cancel"
    PLAYBACK_FAILED = "playback_failed"
    OVERFLOW = "overflow"
    SHUTDOWN = "shutdown"


class ConversationCancelReason(StrEnum):
    USER = "user"
    SHUTDOWN = "shutdown"
    CLIENT_FAILED = "client_failed"


class CoarseState(StrEnum):
    WAITING = "waiting"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"


class InputCloseReason(StrEnum):
    ENDPOINT = "endpoint"
    MAX_DURATION = "max_duration"
    NO_SPEECH = "no_speech"
    FAILED = "failed"


class ResponseCancelReason(StrEnum):
    BARGE_IN = "barge_in"
    LOCAL_CANCEL = "local_cancel"
    CONVERSATION_CANCELLED = "conversation_cancelled"
    GENERATION_FAILED = "generation_failed"
    TTS_FAILED = "tts_failed"
    PLAYBACK_FAILED = "playback_failed"
    OVERFLOW = "overflow"
    SHUTDOWN = "shutdown"


class ConversationEndReason(StrEnum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    IDLE_TIMEOUT = "idle_timeout"
    CLIENT_FAILED = "client_failed"
    SERVER_FAILED = "server_failed"
    PLAYBACK_FAILED = "playback_failed"


class ErrorScope(StrEnum):
    CONNECTION = "connection"
    CONVERSATION = "conversation"
    INPUT = "input"
    RESPONSE = "response"


class ErrorCode(StrEnum):
    MALFORMED_MESSAGE = "malformed_message"
    MESSAGE_TOO_LARGE = "message_too_large"
    UNKNOWN_MESSAGE = "unknown_message"
    UNSUPPORTED_VERSION = "unsupported_version"
    CAPABILITY_MISMATCH = "capability_mismatch"
    FORMAT_MISMATCH = "format_mismatch"
    HANDSHAKE_TIMEOUT = "handshake_timeout"
    PROTOCOL_STATE = "protocol_state"
    ID_EXHAUSTED = "id_exhausted"
    PEER_UNRESPONSIVE = "peer_unresponsive"
    CONVERSATION_FAILED = "conversation_failed"
    IDLE_TIMEOUT = "idle_timeout"
    INPUT_DISCONTINUITY = "input_discontinuity"
    INPUT_OVERFLOW = "input_overflow"
    INPUT_TOO_LONG = "input_too_long"
    CAPTURE_FAILED = "capture_failed"
    STT_FAILED = "stt_failed"
    PROCESSING_TIMEOUT = "processing_timeout"
    GENERATION_FAILED = "generation_failed"
    TTS_FAILED = "tts_failed"
    PLAYBACK_FAILED = "playback_failed"
    OUTPUT_OVERFLOW = "output_overflow"
    RESPONSE_CANCELLED = "response_cancelled"


class ConnectionState(StrEnum):
    DISCONNECTED = "disconnected"
    HANDSHAKING = "handshaking"
    READY = "ready"
    CLOSING = "closing"


class MessageDirection(StrEnum):
    CLIENT_TO_SERVER = "client_to_server"
    SERVER_TO_CLIENT = "server_to_client"


class EndpointRole(StrEnum):
    CLIENT = "client"
    SERVER = "server"


class ProtocolObjectKind(StrEnum):
    CONVERSATION = "conversation"
    INPUT = "input"
    RESPONSE = "response"
    OUTPUT = "output"


class AudioSubmitResult(StrEnum):
    ACCEPTED = "accepted"
    IGNORED_INACTIVE = "ignored_inactive"
    IGNORED_WAITING_SILENCE = "ignored_waiting_silence"
    CLOSED_INPUT = "closed_input"
    OVERFLOW = "overflow"
