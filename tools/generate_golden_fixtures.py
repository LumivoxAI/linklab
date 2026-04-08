import json
import argparse
from typing import cast
from pathlib import Path

import msgpack  # type: ignore[import-untyped]

import lumivox_linklab as linklab

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "fixtures" / "protocol-v1.json"

CAPABILITIES = ["barge_in", "playback_accounting", "speech_spans"]
FORMAT_16K = {"encoding": "pcm_s16le", "sample_rate_hz": 16_000, "channels": 1}
FORMAT_24K = {"encoding": "pcm_s16le", "sample_rate_hz": 24_000, "channels": 1}
LIMITS = {
    "max_message_bytes": 262_144,
    "max_input_audio_frames": 1_600,
    "max_output_audio_frames": 2_400,
    "max_text_bytes": 16_384,
    "max_input_frames": 1_920_000,
    "idle_timeout_ms": 120_000,
}

type Fixture = tuple[str, linklab.MessageDirection, dict[str, object]]

FIXTURES: tuple[Fixture, ...] = (
    (
        "client-hello",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "input_format": FORMAT_16K,
            "output_formats": [FORMAT_24K, FORMAT_16K],
            "agent": "golden-client",
        },
    ),
    (
        "conversation-start",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {"type": "conversation.start", "conversation_id": 1, "activation": "wake_word", "wake_word": "lumi"},
    ),
    (
        "input-start",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {"type": "input.start", "conversation_id": 1, "input_id": 1, "reason": "activation", "generation": 0},
    ),
    (
        "input-start-barge-in",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {
            "type": "input.start",
            "conversation_id": 1,
            "input_id": 2,
            "reason": "barge_in",
            "generation": 1,
            "interrupts_response_id": 1,
        },
    ),
    (
        "input-audio",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {
            "type": "input.audio",
            "conversation_id": 1,
            "input_id": 1,
            "start_frame": 0,
            "speech": True,
            "audio": b"\x01\x02\x03\x04",
        },
    ),
    (
        "input-abort",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {"type": "input.abort", "conversation_id": 1, "input_id": 1, "reason": "capture_failed"},
    ),
    (
        "playback-finished",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {
            "type": "playback.finished",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "played_frames": 2,
        },
    ),
    (
        "playback-interrupted",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {
            "type": "playback.interrupted",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "played_frames": 1,
            "position": "estimated",
            "reason": "barge_in",
        },
    ),
    (
        "conversation-cancel",
        linklab.MessageDirection.CLIENT_TO_SERVER,
        {"type": "conversation.cancel", "conversation_id": 1, "reason": "user"},
    ),
    (
        "server-hello",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "hello",
            "version": 1,
            "capabilities": CAPABILITIES,
            "output_format": FORMAT_24K,
            "limits": LIMITS,
            "agent": "golden-server",
        },
    ),
    (
        "state",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "state", "conversation_id": 1, "revision": 1, "state": "listening", "reason": "activation"},
    ),
    (
        "input-closed",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "input.closed",
            "conversation_id": 1,
            "input_id": 1,
            "accepted_end_frame": 2,
            "reason": "endpoint",
        },
    ),
    (
        "transcript-update",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "transcript.update",
            "conversation_id": 1,
            "input_id": 1,
            "revision": 1,
            "text": "hello",
            "language": "en",
        },
    ),
    (
        "transcript-final",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "transcript.final", "conversation_id": 1, "input_id": 1, "text": "hello", "language": "en"},
    ),
    (
        "response-start",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "response.start",
            "conversation_id": 1,
            "response_id": 1,
            "input_id": 1,
            "end_conversation": False,
        },
    ),
    (
        "response-text-delta",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "response.text.delta", "conversation_id": 1, "response_id": 1, "sequence": 0, "text": "hi"},
    ),
    (
        "response-text-final",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "response.text.final", "conversation_id": 1, "response_id": 1, "text": "hi"},
    ),
    (
        "output-start",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "output.start", "conversation_id": 1, "response_id": 1, "output_id": 1},
    ),
    (
        "output-audio",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "output.audio",
            "conversation_id": 1,
            "response_id": 1,
            "output_id": 1,
            "start_frame": 0,
            "audio": b"\x01\x02\x03\x04",
        },
    ),
    (
        "output-end",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "output.end", "conversation_id": 1, "response_id": 1, "output_id": 1, "total_frames": 2},
    ),
    (
        "response-end",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "response.end", "conversation_id": 1, "response_id": 1},
    ),
    (
        "response-cancelled",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "response.cancelled", "conversation_id": 1, "response_id": 1, "reason": "barge_in"},
    ),
    (
        "conversation-end",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {"type": "conversation.end", "conversation_id": 1, "reason": "completed"},
    ),
    (
        "error",
        linklab.MessageDirection.SERVER_TO_CLIENT,
        {
            "type": "error",
            "scope": "response",
            "code": "tts_failed",
            "fatal": True,
            "conversation_id": 1,
            "response_id": 1,
            "message": "safe diagnostic",
        },
    ),
)


def _json_value(value: object) -> object:
    if type(value) is bytes:
        return {"$binary_hex": value.hex()}
    if type(value) is list:
        return [_json_value(item) for item in value]
    if type(value) is dict:
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _render() -> str:
    fixtures: list[dict[str, object]] = []
    for name, direction, primitive in FIXTURES:
        source = cast(bytes, msgpack.packb(primitive, use_bin_type=True))
        message = linklab.decode_message(source, direction=direction, limits=linklab.ConnectionLimits())
        fixtures.append(
            {
                "name": name,
                "direction": direction.value,
                "message": _json_value(primitive),
                "wire_hex": linklab.encode_message(message).hex(),
            }
        )
    return json.dumps({"protocol": "lumivox.voice.v1", "fixtures": fixtures}, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic v1 MessagePack golden fixtures.")
    parser.add_argument("--check", action="store_true", help="fail instead of writing when the manifest differs")
    args = parser.parse_args()
    rendered = _render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text() != rendered:
            parser.error(f"{OUTPUT.relative_to(ROOT)} is not up to date")
        return 0
    OUTPUT.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
