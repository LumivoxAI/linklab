import typing
import inspect
from pathlib import Path

import lumivox_linklab as linklab

ROOT = Path(__file__).parents[1]
EXPECTED_EXPORTS = {
    "AnnotatedAudio",
    "AudioFormat",
    "AudioSubmitResult",
    "ClientCallbacks",
    "ClientConfig",
    "ClientHello",
    "ClientMessage",
    "CoarseState",
    "CodecError",
    "ConnectionClosed",
    "ConnectionLimits",
    "ConnectionState",
    "ConnectionStateEvent",
    "ConversationCancelReason",
    "ConversationCancelledEvent",
    "ConversationEndReason",
    "ConversationEndedEvent",
    "ConversationId",
    "ConversationStartedEvent",
    "EndpointRole",
    "ErrorCode",
    "ErrorEvent",
    "ErrorScope",
    "InputAbortReason",
    "InputAbortedEvent",
    "InputAudioEvent",
    "InputCloseReason",
    "InputClosedEvent",
    "InputId",
    "InputStartReason",
    "InputStartedEvent",
    "LinklabError",
    "Message",
    "MessageDirection",
    "OutputAudioEvent",
    "OutputEndedEvent",
    "OutputId",
    "OutputStartedEvent",
    "OutputWriter",
    "PlaybackFinishedEvent",
    "PlaybackInterruptReason",
    "PlaybackInterruptedEvent",
    "PlaybackPosition",
    "ProtocolObjectKind",
    "ProtocolStateSnapshot",
    "ProtocolTombstone",
    "ProtocolValidator",
    "ProtocolViolation",
    "QueueOverflow",
    "ReadableBuffer",
    "ResponseCancelReason",
    "ResponseCancelledEvent",
    "ResponseEndedEvent",
    "ResponseId",
    "ResponseStartedEvent",
    "ResponseTextDeltaEvent",
    "ResponseTextFinalEvent",
    "ResponseWriter",
    "ServerConfig",
    "ServerHandler",
    "ServerHello",
    "ServerMessage",
    "ServerSession",
    "StateEvent",
    "TranscriptFinalEvent",
    "TranscriptUpdateEvent",
    "VoiceClient",
    "VoiceServer",
    "WriterClosed",
    "decode_message",
    "encode_message",
}


def test_exact_root_export_inventory_hides_transport_and_mutable_protocol_state() -> None:
    assert set(linklab.__all__) == EXPECTED_EXPORTS
    assert len(linklab.__all__) == len(EXPECTED_EXPORTS)
    assert all(not name.startswith("_") for name in linklab.__all__)
    assert not ({"WebSocket", "ClientConnection", "ServerConnection", "_ProtocolState"} & EXPECTED_EXPORTS)

    for name in linklab.__all__:
        value = getattr(linklab, name)
        assert value.__module__.startswith("lumivox_linklab") or name in {
            "ConversationId",
            "InputId",
            "OutputId",
            "ReadableBuffer",
            "ResponseId",
        }


def test_voice_client_exact_surface_and_facade_context_annotations() -> None:
    expected_methods = {
        "connect",
        "close",
        "wait_closed",
        "submit_annotated_audio",
        "abort_input",
        "cancel_conversation",
        "playback_finished",
        "playback_interrupted",
        "connection_state",
        "output_format",
    }
    assert {name for name in dir(linklab.VoiceClient) if not name.startswith("_")} == expected_methods
    for name in ("connection_state", "output_format"):
        value = inspect.getattr_static(linklab.VoiceClient, name)
        assert isinstance(value, property) and value.fset is None

    for facade in (linklab.VoiceClient, linklab.VoiceServer):
        assert typing.get_type_hints(facade.__aenter__)["return"] is typing.Self
        assert typing.get_type_hints(facade.__aexit__)["return"] == typing.Literal[False]


def test_public_security_documentation_contains_required_deployment_controls() -> None:
    english = (ROOT / "doc/en/operations.md").read_text()
    russian = (ROOT / "doc/ru/operations.md").read_text()

    for text in (english, russian):
        assert "127.0.0.1" in text
        assert "max_connections" in text
        assert "ws://" in text
        assert "wss://" in text
        assert "ssl.SSLContext" in text
        assert "firewall" in text
        assert "1008" in text
        assert "262,144" in text or "262 144" in text


def test_conformance_matrix_has_no_remaining_gap_or_planned_rows() -> None:
    matrix = (ROOT / ".agent/conformance-matrix.md").read_text()
    table_rows = [line for line in matrix.splitlines() if line.startswith("| ")]
    requirement_rows = [
        line
        for line in table_rows
        if line.split("|")[1].strip().partition("-")[0] in {"PRJ", "ARC", "WIRE", "API", "VER"}
        and "-" in line.split("|")[1]
    ]
    assert requirement_rows
    assert all("| covered |" in row for row in requirement_rows)
