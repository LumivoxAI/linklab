# Using Linklab

## Common Audio Formats

```python
import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
PCM_24K = linklab.AudioFormat("pcm_s16le", 24_000, 1)
PCM_48K = linklab.AudioFormat("pcm_s16le", 48_000, 1)
```

Input is always mono PCM S16LE at 16 kHz. Client output preferences must be a
unique subset in the order 24 kHz, 48 kHz, 16 kHz and must include 16 kHz.
Server formats are a unique supported set and must include 16 kHz.

## Client Flow

Implement every async method in `ClientCallbacks`, then create `VoiceClient`
inside the running event loop. Supply a `lumivox_core.logger.Logger` compatible
structured logger.

```python
config = linklab.ClientConfig(
    uri="ws://127.0.0.1:8765",
    output_formats=(PCM_24K, PCM_16K),
    reconnect=True,
)
client = linklab.VoiceClient(config, callbacks, logger)

async with client:
    await application_stopped.wait()
```

`connect()` returns after the protocol handshake. Read `client.output_format`
after that point to configure playback. `close()` is idempotent and
`wait_closed()` waits for complete termination.

Capture or Wakelab code submits uniform annotations synchronously:

```python
result = client.submit_annotated_audio(
    linklab.AnnotatedAudio(
        audio=pcm_s16le,
        generation=generation,
        discontinuity=False,
        speech=is_speech,
        activated=is_activated,
        wake_word="lumivox" if is_activated else None,
    )
)
```

The call is thread-safe, bounded, non-blocking, and copies the buffer before it
returns. Handle all `AudioSubmitResult` values. `OVERFLOW` aborts an already-open
input or rejects an atomic activation. Never retry or replay rejected PCM.

Wakelab integration must preserve these rules:

- call Wakelab synchronously and serially outside Linklab;
- submit each output exactly once and in order;
- each chunk must have uniform speech/activation metadata and contiguous 16 kHz
  mono int16 PCM;
- activation must include retained wake-word audio and pre-roll, not start at
  the detection point;
- pass generation changes and discontinuities exactly as reported;
- keep activation latched until re-arm;
- re-arm after `ConversationEndedEvent` or disconnected
  `ConnectionStateEvent`.

During `responding`, an accepted confirmed-speech chunk is a barge-in signal.
Stop/fade and flush physical playback immediately; do not wait for a later
callback. Linklab prevents stale queued output from reaching callbacks.

## Playback Accounting

`on_output_audio` receives immutable PCM and absolute frame offsets. Submit it to
the player without blocking the callback indefinitely. After one output:

```python
client.playback_finished(output_id, played_frames)
```

or:

```python
client.playback_interrupted(
    output_id,
    played_frames,
    linklab.PlaybackPosition.EXACT,
    linklab.PlaybackInterruptReason.LOCAL_CANCEL,
)
```

Use `ESTIMATED` only when the device cannot report an exact played frame count.
Report exactly one outcome unless disconnected. A callback failure or playback
queue overflow maps to playback failure and can end the conversation.

## Server Flow

Implement every async method in `ServerHandler`. A synchronous, non-blocking
factory creates a separate handler for each accepted connection:

```python
config = linklab.ServerConfig(
    port=8765,
    output_formats=(PCM_24K, PCM_16K),
)
server = linklab.VoiceServer(config, lambda session: Handler(), logger)

async with server:
    await application_stopped.wait()
```

`on_input_audio` receives immutable input PCM in validated contiguous ranges.
The application controls endpointing and the STT/LLM/TTS pipeline:

```python
await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
await session.update_transcript(event.input_id, 1, "partial", "en")
await session.finalize_transcript(event.input_id, "final", "en")

async with await session.start_response(event.input_id) as response:
    await response.send_text_delta("Hello")
    await response.finalize_text("Hello")
    async with await response.start_output() as output:
        await output.send_audio(pcm_s16le)
```

Transcript revisions start at 1 and increase exactly by one. Finalize the
transcript after input termination and before starting a response. A response
may contain text, output, both, or neither. A started output must contain at
least one frame. `end_conversation=True` on `start_response` applies terminal
intent after successful completion/playback.

External model tasks remain application-owned. When a handler is cancelled or a
writer raises `WriterClosed`, cancel the associated STT/LLM/TTS work. Stale
writer calls cannot publish data.

## Discovery

Set `discovery_service_id` to a lowercase DNS label such as `production`.
Servers advertise `_lumivox-voice._tcp.local.` only after listening; clients with
`uri=None` browse for the exact instance. An explicit client `uri` always wins
and disables discovery. Discovery is optional, unauthenticated, limited to the
local multicast domain, and does not replace TLS identity checks.

## Runnable Fake Pipeline

See [`examples/README.md`](../../examples/README.md) for separate client and
server programs using only public facades. The deterministic loopback path
demonstrates activation, fake STT/LLM/TTS, streamed PCM, playback accounting,
barge-in cancellation, re-arm, and clean shutdown without devices or services.

## Low-Level Codec And Validator

Most applications should use the facades. Tools and non-WebSocket integrations
may use `encode_message`, `decode_message`, and `ProtocolValidator`. Decode with
the correct `MessageDirection` and negotiated `ConnectionLimits`, and feed both
sent and received semantic messages to one endpoint-role validator in exact
connection order.
