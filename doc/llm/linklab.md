# Linklab Library Integration Instructions

Use this document to integrate `lumivox-linklab` without reading its source.
The package import name is `lumivox_linklab`.

## Purpose And Boundaries

Linklab transports typed voice-agent events between one local client and server.
Use `VoiceClient` and `VoiceServer`; do not construct wire dictionaries, allocate
IDs, access WebSockets, or implement protocol state machines.

Linklab does not implement audio device I/O, Wakelab, wake-word detection, VAD,
AEC, STT, LLM, TTS, resampling, authentication, or conversation policy. Keep
those in the integrating application.

Runtime: Linux, CPython `>=3.13,<3.15`, asyncio. NumPy is not required.

## Installation

The current development source is branch `master`:

```bash
pip install "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

For reproducible deployments, replace `master` with a tested commit SHA or rely
on a committed lockfile.

```python
import lumivox_linklab as linklab
```

Runnable bounded client/server integrations are in `examples/`. Start
`fake_pipeline_server.py` and then `fake_pipeline_client.py` for a deterministic
loopback demonstration of activation, fake STT/LLM/TTS, playback accounting,
barge-in cancellation, re-arm, and shutdown. Their matching `--service-id`
options demonstrate opt-in LAN discovery; the default smoke path uses no mDNS or
external resources.

The constructors for `VoiceClient` and `VoiceServer` require an object satisfying
`lumivox_core.logger.Logger`. Linklab directly calls `bind`, `debug`, `info`,
`warning`, `error`, and `exception`. Do not pass the stdlib logger directly
unless an adapter implements that contract.

## Audio Rules

Define formats explicitly:

```python
PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
PCM_24K = linklab.AudioFormat("pcm_s16le", 24_000, 1)
PCM_48K = linklab.AudioFormat("pcm_s16le", 48_000, 1)
```

Input is always mono PCM S16LE at 16 kHz. Output is mono PCM S16LE at 24, 48,
or 16 kHz. A frame is one int16 sample because channels are fixed to one;
`bytes_per_frame == 2`. Frame offsets are not byte offsets or timestamps.

Every PCM buffer must be nonempty, contiguous, readable through the Python
buffer protocol, and frame-aligned. Linklab copies accepted input/output buffers
before the public call returns. Decoded PCM is immutable `bytes`.

Client `output_formats` is a unique 1..3-item subset in canonical preference
order `(PCM_24K, PCM_48K, PCM_16K)` with omitted entries removed, and must
include `PCM_16K`. Server `output_formats` is a unique 1..3-item supported set
and must include `PCM_16K`; its order is not a client preference.

## Concurrency Guarantees

- Construct `VoiceClient`, `VoiceServer`, `ServerSession`, and writers in a
  running event loop.
- Async objects bind to their creation loop. Invoke all async methods and read
  loop-owned workflows on that loop. Linklab never creates or owns an event loop.
- These `VoiceClient` methods are explicitly thread-safe, bounded, synchronous,
  and non-blocking: `submit_annotated_audio`, `abort_input`,
  `cancel_conversation`, `playback_finished`, `playback_interrupted`.
- Do not assume any other public method or object is thread-safe.
- Thread-safe calls use finite handoff queues. They report rejection instead of
  blocking for network pressure.
- Linklab copies readable buffers before a submitting/sending method returns.
- Client callbacks and server handlers are async, serialized in validated event
  order, and dispatched through bounded queues.
- The WebSocket reader does not wait for physical playback. Do not block a
  callback while waiting for speaker completion.
- One `VoiceServer` connection owns one isolated `ServerSession` and one handler
  instance returned by the handler factory.
- External STT/LLM/TTS tasks are application-owned. Cancel them when the handler
  is cancelled or a writer becomes stale/closed.

## Client Configuration

Create `ClientConfig` with these fields:

```text
uri: str | None                              required
output_formats: tuple[AudioFormat, ...]      required
discovery_service_id: str | None             None
discovery_timeout_s: positive float          10.0
ssl_context: ssl.SSLContext | None           None
input_queue_frames: positive int             16000
playback_queue_ms: positive int              2000
waiting_pre_roll_frames: nonnegative int     8000
connect_timeout_s: float in (0, 10]          10.0
handshake_timeout_s: float in (0, 5]         5.0
close_timeout_s: float in (0, 10]            10.0
ping_interval_s: float in (0, 20]            20.0
ping_timeout_s: float in (0, 20]             20.0
websocket_max_queue: int in 1..16            16
websocket_write_limit: int in 1..65536       65536
reconnect: bool                              False
reconnect_initial_s: positive float          0.5
reconnect_max_s: positive float              30.0
agent: str containing 1..64 UTF-8 bytes      "lumivox-linklab"
```

At least one of `uri` or `discovery_service_id` is required. A non-`None` `uri`
takes precedence and prevents discovery from starting. A service ID is a
lowercase ASCII DNS label matching `[a-z0-9][a-z0-9-]{0,62}`.

## Client Callbacks

Implement every method of the runtime-checkable `ClientCallbacks` protocol. All
methods are async and return `None`:

```python
async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None: ...
async def on_conversation_state(self, event: linklab.StateEvent) -> None: ...
async def on_transcript_update(self, event: linklab.TranscriptUpdateEvent) -> None: ...
async def on_transcript_final(self, event: linklab.TranscriptFinalEvent) -> None: ...
async def on_response_started(self, event: linklab.ResponseStartedEvent) -> None: ...
async def on_response_text_delta(self, event: linklab.ResponseTextDeltaEvent) -> None: ...
async def on_response_text_final(self, event: linklab.ResponseTextFinalEvent) -> None: ...
async def on_response_ended(self, event: linklab.ResponseEndedEvent) -> None: ...
async def on_response_cancelled(self, event: linklab.ResponseCancelledEvent) -> None: ...
async def on_output_started(self, event: linklab.OutputStartedEvent) -> None: ...
async def on_output_audio(self, event: linklab.OutputAudioEvent) -> None: ...
async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None: ...
async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None: ...
async def on_error(self, event: linklab.ErrorEvent) -> None: ...
```

Store current `output_id` from output events if the playback layer needs it for
accounting. `OutputAudioEvent.audio` is immutable PCM; `start_frame` is absolute.

## Client Lifecycle

Construct and run the client as follows:

```python
config = linklab.ClientConfig(
    uri="ws://127.0.0.1:8765",
    output_formats=(PCM_24K, PCM_16K),
    reconnect=True,
)
client = linklab.VoiceClient(config, callbacks, logger)

async with client:
    # connect() has completed the server hello here.
    negotiated_format = client.output_format
    await stop_event.wait()
```

`connect()` may be called once and returns after handshake. Concurrent/repeated
calls raise `RuntimeError`. `close()` is bounded and idempotent.
`wait_closed()` waits for full shutdown. Read-only properties are
`connection_state: ConnectionState` and `output_format: AudioFormat | None`.

Optional reconnect uses full-jitter exponential backoff from 0.5 to 30 seconds.
It is eligible after close 1001 or abnormal transport loss, not after protocol
1002 or policy 1008. It never resumes IDs/conversations or replays PCM.

## Wakelab Or Capture Integration

For every Wakelab-like output, call:

```python
result = client.submit_annotated_audio(
    linklab.AnnotatedAudio(
        audio=pcm,
        generation=generation,
        discontinuity=discontinuity,
        speech=speech,
        activated=activated,
        wake_word=wake_word,
    )
)
```

Required adapter behavior:

- Call Wakelab synchronously and serially outside Linklab.
- Submit outputs exactly once and in order.
- Preserve generation, discontinuity, speech, activation, and wake-word values.
- Each chunk must have uniform speech/activation metadata.
- Activation audio must begin at the retained activated-region boundary and
  include the wake word and measured pre-roll. Do not begin at detection time.
- Activation remains latched until application re-arm.
- Re-arm on `ConversationEndedEvent`; re-arm on disconnected
  `ConnectionStateEvent` after connection loss.
- Do not submit idle background audio. Linklab internally retains bounded
  waiting pre-roll when applicable.

Handle `AudioSubmitResult`:

```text
ACCEPTED                 PCM committed for activation/open input/speech
IGNORED_INACTIVE         disconnected/unavailable or no active activation
IGNORED_WAITING_SILENCE  retained or ignored as bounded waiting pre-roll
CLOSED_INPUT             server already closed current input
OVERFLOW                 no atomic capacity; do not retry/replay the chunk
```

Invalid empty, odd-byte, noncontiguous, or otherwise malformed audio raises
`ValueError` and consumes no state/ID.

Use `abort_input(InputAbortReason.CAPTURE_FAILED)` for a capture-side failure.
Use `cancel_conversation(ConversationCancelReason.USER)` for local user cancel.
The methods return true only when accepted into reserved capacity.

If confirmed speech returns `ACCEPTED` while the last observed state is
`RESPONDING`, stop/fade and flush physical playback immediately. This is
barge-in. Do not wait for the listening or cancellation callback.

## Playback Accounting

For every started output, report exactly one outcome unless transport disconnects:

```python
accepted = client.playback_finished(output_id, played_frames)
```

Finished `played_frames` must equal output total frames.

```python
accepted = client.playback_interrupted(
    output_id,
    played_frames,
    linklab.PlaybackPosition.EXACT,  # or ESTIMATED
    linklab.PlaybackInterruptReason.LOCAL_CANCEL,
)
```

Interrupted frames must not exceed received/sent output PCM. Reasons are
`BARGE_IN`, `LOCAL_CANCEL`, `PLAYBACK_FAILED`, `OVERFLOW`, `SHUTDOWN`. Use exact
position whenever available. False means terminal/unavailable/not accepted.

## Server Configuration

Create `ServerConfig` with these fields:

```text
port: int in 1..65535                     required
output_formats: tuple[AudioFormat, ...]   required
host: str                                 "127.0.0.1"
discovery_service_id: str | None          None
ssl_context: ssl.SSLContext | None        None
limits: ConnectionLimits                  ConnectionLimits()
max_connections: positive int             8
input_queue_frames: positive int          32000
output_queue_ms: positive int             500
waiting_timeout_s: positive float         60.0
input_timeout_s: positive float           120.0
processing_timeout_s: positive float      120.0
handshake_timeout_s: float in (0, 5]      5.0
close_timeout_s: float in (0, 10]         10.0
ping_interval_s: float in (0, 20]         20.0
ping_timeout_s: float in (0, 20]          20.0
websocket_max_queue: int in 1..16         16
websocket_write_limit: int in 1..65536    65536
agent: 1..64 UTF-8 bytes                  "lumivox-linklab"
```

Discovery with a loopback-only host is rejected. The server does not bind
non-loopback or wildcard interfaces unless explicitly configured.

`ConnectionLimits` defaults/ranges:

```text
max_message_bytes         262144   range 16384..262144
max_input_audio_frames    1600     range 160..1600
max_output_audio_frames   4800     pre-selection cap range 160..4800
max_text_bytes            16384    range 1024..65536
max_input_frames          1920000  range 16000..1920000
idle_timeout_ms           120000   range 10000..600000
```

## Server Handler And Lifecycle

Implement every method of runtime-checkable `ServerHandler`:

```python
async def on_conversation_started(self, session, event) -> None: ...
async def on_input_started(self, session, event) -> None: ...
async def on_input_audio(self, session, event) -> None: ...
async def on_input_aborted(self, session, event) -> None: ...
async def on_playback_outcome(self, session, event) -> None: ...
async def on_conversation_cancelled(self, session, event) -> None: ...
```

The factory is synchronous and non-blocking:

```python
server = linklab.VoiceServer(
    linklab.ServerConfig(port=8765, output_formats=(PCM_24K, PCM_16K)),
    lambda session: Handler(),
    logger,
)

async with server:
    await stop_event.wait()
```

`serve()` is callable once and returns after listener and optional discovery
registration are active. Repeated/concurrent calls raise `RuntimeError`.
`close()` is idempotent and bounded. `wait_closed()` waits for shutdown.

Input audio events are contiguous and validated. Implement endpointing and then:

```python
await session.close_input(input_id, linklab.InputCloseReason.ENDPOINT)
await session.update_transcript(input_id, revision=1, text="partial", language="en")
await session.finalize_transcript(input_id, text="final", language="en")
```

Transcript update revisions begin at 1 and increment exactly. Finalize only
after input closes/aborts and before starting a response. Empty final transcript
is legal only for no-speech/failed/aborted input paths.

Start and write a response:

```python
async with await session.start_response(input_id, end_conversation=False) as response:
    await response.send_text_delta("part ")
    await response.send_text_delta("two")
    await response.finalize_text("part two")
    async with await response.start_output() as output:
        await output.send_audio(pcm_chunk)
```

Text is optional. Output is optional. One response has at most one output. A
started output must receive nonempty PCM and finish with at least one frame.
Writers allocate sequence values, IDs, and frame offsets. Do not provide them.

`ResponseWriter` methods: read-only `response_id`, `send_text_delta(text)`,
`finalize_text(text)`, `start_output()`, `finish()`, `cancel(reason)`.
Application cancel reasons are restricted to `GENERATION_FAILED`, `TTS_FAILED`,
`OVERFLOW` after output start, and `SHUTDOWN`.

`OutputWriter` methods: read-only `output_id`, `send_audio(audio)`, `finish()`.
If `send_audio` raises `QueueOverflow`, Linklab has already emitted the required
output-overflow cancellation; do not retry and do not call another terminal
operation on that writer.

Both writers are async context managers. Normal exit calls `finish`. Response
body exceptions map to generation failure; output body exceptions map to TTS
failure. Original exceptions are not suppressed. A stale or terminal writer
raises `WriterClosed`.

Other `ServerSession` operations:

```text
end_conversation(reason: ConversationEndReason) -> None
fail(scope: ErrorScope, code: ErrorCode, message: str | None = None) -> None
```

`end_conversation` terminates children first. `fail` selects the sole valid
current target for input/response scope; invalid scope/code/target combinations
raise. Safe diagnostic messages are at most 512 UTF-8 bytes.

## Lifecycle Assumptions

- One connection has at most one open conversation.
- Conversations and user inputs are sequential.
- An input has at most one response; a response has zero or one output.
- Coarse states are `WAITING`, `LISTENING`, `PROCESSING`, `RESPONDING`.
- State events are projections; object events and explicit IDs are authoritative.
- Stale text/audio after barge-in is validated but not delivered to UI/playback.
- Do not cache or reuse IDs after disconnect.
- Do not replay conversations or PCM after reconnect.
- `end_conversation=True` on `start_response` ends only after successful
  response/playback. Barge-in revokes terminal intent.

## Public Enums And Exceptions

Use enum members, not arbitrary strings:

```text
ConnectionState: DISCONNECTED HANDSHAKING READY CLOSING
CoarseState: WAITING LISTENING PROCESSING RESPONDING
InputAbortReason: DISCONTINUITY OVERFLOW CAPTURE_FAILED SHUTDOWN
InputCloseReason: ENDPOINT MAX_DURATION NO_SPEECH FAILED
PlaybackPosition: EXACT ESTIMATED
PlaybackInterruptReason: BARGE_IN LOCAL_CANCEL PLAYBACK_FAILED OVERFLOW SHUTDOWN
ConversationCancelReason: USER SHUTDOWN CLIENT_FAILED
ResponseCancelReason: BARGE_IN LOCAL_CANCEL CONVERSATION_CANCELLED
                      GENERATION_FAILED TTS_FAILED PLAYBACK_FAILED OVERFLOW SHUTDOWN
ConversationEndReason: COMPLETED CANCELLED IDLE_TIMEOUT CLIENT_FAILED
                       SERVER_FAILED PLAYBACK_FAILED
ErrorScope: CONNECTION CONVERSATION INPUT RESPONSE
```

Catch `LinklabError` for all library-defined errors. Leaf classes are
`CodecError`, `ProtocolViolation`, `ConnectionClosed`, `QueueOverflow`, and
`WriterClosed`. They are independent siblings.

## Discovery

The fixed service type is `_lumivox-voice._tcp.local.`. A server advertises only
after listening. A discovery client selects the exact configured service ID and
valid protocol/scheme TXT properties. Explicit URI always disables discovery.

Discovery is unauthenticated and spoofable. It may fail across VLANs,
containers, VPNs, or networks that suppress multicast. Treat it as a routing
hint, not identity. WSS certificate validation remains required when identity is
needed.

## Security Requirements

- Deploy only on a trusted isolated LAN unless protected by an appropriate
  authenticated gateway.
- Never forward a Linklab port directly to the Internet.
- Never send secrets over unauthenticated plain `ws://`.
- Use caller-configured `wss://` contexts and trusted/pinned certificates where
  server authentication is required.
- Linklab does not authenticate clients, including when WSS is used.
- Apply firewall restrictions and keep connection/model admission limits finite.
- Log origins, admission, and protocol errors. Do not log PCM, full transcripts,
  full generated text, or unsafe peer diagnostics.

## Low-Level API

Do not use this section for normal client/server applications.

```python
encoded: bytes = linklab.encode_message(message)
message = linklab.decode_message(
    data,
    direction=linklab.MessageDirection.CLIENT_TO_SERVER,
    limits=limits,
)

validator = linklab.ProtocolValidator(linklab.EndpointRole.SERVER)
validator.transport_connected()
validator.accept(client_hello)
validator.accept(server_hello)
snapshot = validator.state
tombstones = validator.tombstones
validator.begin_close()
validator.transport_disconnected()
```

Encoding validates schema and is canonical. Decoding applies bounded structural
and schema validation but does not mutate lifecycle. Feed both sent and received
semantic messages to one validator in exact connection order. `accept` is atomic
and raises `ProtocolViolation` without partial state mutation.

Do not relax these absolute peer-input limits: WebSocket message 262,144 bytes;
first hello 16,384 bytes; nesting depth 8; map entries 64; array items 32; UTF-8
string 65,536 bytes; identifier/reason/type 64 bytes; non-audio binary 16,384
bytes. WebSocket compression is disabled.
