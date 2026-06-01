from collections.abc import Callable

import pytest

import lumivox_linklab as linklab

CAPABILITIES = ("barge_in", "playback_accounting", "speech_spans")
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
LIMITS = linklab.ConnectionLimits(max_output_audio_frames=1_600)
CONVERSATION_ID = linklab.ConversationId(1)
INPUT_ID = linklab.InputId(1)
RESPONSE_ID = linklab.ResponseId(1)
OUTPUT_ID = linklab.OutputId(1)

TRACE_INVENTORY = (
    "connection/legal",
    "connection/illegal",
    "conversation/legal",
    "conversation/illegal",
    "input/legal",
    "input/illegal",
    "response/legal",
    "response/illegal",
    "output/legal",
    "output/illegal",
    "playback/legal",
    "playback/illegal",
    "cancellation/legal",
    "cancellation/illegal",
    "error/legal",
    "error/illegal",
    "tombstone/legal",
    "tombstone/illegal",
    "disconnect/legal",
    "disconnect/illegal",
)


def connected() -> linklab.ProtocolValidator:
    validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
    validator.transport_connected()
    return validator


def ready() -> linklab.ProtocolValidator:
    validator = connected()
    validator.accept(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
    validator.accept(linklab.ServerHello(1, CAPABILITIES, PCM_16K, LIMITS))
    return validator


def conversation() -> linklab.ProtocolValidator:
    validator = ready()
    validator.accept(linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    return validator


def open_input() -> linklab.ProtocolValidator:
    validator = conversation()
    validator.accept(linklab.InputStartedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputStartReason.ACTIVATION, 0))
    return validator


def response() -> linklab.ProtocolValidator:
    validator = open_input()
    validator.accept(linklab.InputClosedEvent(CONVERSATION_ID, INPUT_ID, 0, linklab.InputCloseReason.NO_SPEECH))
    validator.accept(linklab.TranscriptFinalEvent(CONVERSATION_ID, INPUT_ID, ""))
    validator.accept(linklab.ResponseStartedEvent(CONVERSATION_ID, RESPONSE_ID, INPUT_ID, False))
    return validator


def output(*, ended: bool = False) -> linklab.ProtocolValidator:
    validator = response()
    validator.accept(linklab.OutputStartedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID))
    validator.accept(linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\0\0"))
    if ended:
        validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1))
    return validator


def trace_case(name: str) -> tuple[linklab.ProtocolValidator, Callable[[], None]]:
    if name == "connection/legal":
        validator = connected()
        return validator, lambda: validator.accept(linklab.ClientHello(1, CAPABILITIES, PCM_16K, (PCM_16K,)))
    if name == "connection/illegal":
        validator = connected()
        return validator, lambda: validator.accept(linklab.ServerHello(1, CAPABILITIES, PCM_16K, LIMITS))
    if name == "conversation/legal":
        validator = ready()
        return validator, lambda: validator.accept(linklab.ConversationStartedEvent(CONVERSATION_ID, "wake_word"))
    if name == "conversation/illegal":
        validator = conversation()
        return validator, lambda: validator.accept(
            linklab.ConversationStartedEvent(linklab.ConversationId(2), "wake_word")
        )
    if name == "input/legal":
        validator = open_input()
        return validator, lambda: validator.accept(linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 0, True, b"\0\0"))
    if name == "input/illegal":
        validator = open_input()
        return validator, lambda: validator.accept(linklab.InputAudioEvent(CONVERSATION_ID, INPUT_ID, 1, True, b"\0\0"))
    if name == "response/legal":
        validator = response()
        return validator, lambda: validator.accept(
            linklab.ResponseTextDeltaEvent(CONVERSATION_ID, RESPONSE_ID, 0, "text")
        )
    if name == "response/illegal":
        validator = response()
        return validator, lambda: validator.accept(linklab.ResponseEndedEvent(CONVERSATION_ID, linklab.ResponseId(2)))
    if name == "output/legal":
        validator = output()
        return validator, lambda: validator.accept(linklab.OutputEndedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1))
    if name == "output/illegal":
        validator = output()
        return validator, lambda: validator.accept(
            linklab.OutputAudioEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 0, b"\0\0")
        )
    if name == "playback/legal":
        validator = output(ended=True)
        return validator, lambda: validator.accept(
            linklab.PlaybackFinishedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1)
        )
    if name == "playback/illegal":
        validator = output()
        return validator, lambda: validator.accept(
            linklab.PlaybackFinishedEvent(CONVERSATION_ID, RESPONSE_ID, OUTPUT_ID, 1)
        )
    if name == "cancellation/legal":
        validator = conversation()
        return validator, lambda: validator.accept(
            linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER)
        )
    if name == "cancellation/illegal":
        validator = conversation()
        validator.accept(linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.USER))
        return validator, lambda: validator.accept(
            linklab.ConversationCancelledEvent(CONVERSATION_ID, linklab.ConversationCancelReason.CLIENT_FAILED)
        )
    if name == "error/legal":
        validator = conversation()
        return validator, lambda: validator.accept(
            linklab.ErrorEvent(
                linklab.ErrorScope.CONVERSATION,
                linklab.ErrorCode.CONVERSATION_FAILED,
                True,
                CONVERSATION_ID,
            )
        )
    if name == "error/illegal":
        validator = conversation()
        return validator, lambda: validator.accept(
            linklab.ErrorEvent(
                linklab.ErrorScope.INPUT,
                linklab.ErrorCode.STT_FAILED,
                True,
                CONVERSATION_ID,
                linklab.InputId(2),
            )
        )
    if name == "tombstone/legal":
        validator = open_input()
        abort = linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.CAPTURE_FAILED)
        validator.accept(abort)
        return validator, lambda: validator.accept(abort)
    if name == "tombstone/illegal":
        validator = open_input()
        validator.accept(
            linklab.InputAbortedEvent(
                CONVERSATION_ID,
                INPUT_ID,
                linklab.InputAbortReason.CAPTURE_FAILED,
            )
        )
        return validator, lambda: validator.accept(
            linklab.InputAbortedEvent(CONVERSATION_ID, INPUT_ID, linklab.InputAbortReason.DISCONTINUITY)
        )
    if name == "disconnect/legal":
        validator = conversation()
        return validator, validator.transport_disconnected
    if name == "disconnect/illegal":
        validator = linklab.ProtocolValidator(linklab.EndpointRole.CLIENT)
        return validator, validator.transport_disconnected
    raise AssertionError(f"unimplemented trace case: {name}")


@pytest.mark.parametrize("name", TRACE_INVENTORY)
def test_every_declared_legal_and_illegal_trace_is_atomic(name: str) -> None:
    categories = {item.split("/", 1)[0] for item in TRACE_INVENTORY}
    assert categories == {
        "connection",
        "conversation",
        "input",
        "response",
        "output",
        "playback",
        "cancellation",
        "error",
        "tombstone",
        "disconnect",
    }
    assert len(TRACE_INVENTORY) == len(set(TRACE_INVENTORY)) == len(categories) * 2

    validator, operation = trace_case(name)
    before = validator._data
    if name.endswith("/legal"):
        operation()
    else:
        with pytest.raises(linklab.ProtocolViolation):
            operation()
        assert validator._data == before
