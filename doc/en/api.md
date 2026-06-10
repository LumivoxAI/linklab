# Public API Reference

All names below are exported from `lumivox_linklab`. Public dataclasses are
frozen and slotted. Event `type` values are class variables and are not constructor
arguments. IDs are distinct `NewType` aliases over integers.

## Formats And Configuration

`AudioFormat` has constructor `(encoding, sample_rate_hz, channels)` and accepts
only `pcm_s16le`, rates 16,000/24,000/48,000, and one channel.

`ConnectionLimits` fields:

| Field | Default | Valid range |
| --- | ---: | ---: |
| `max_message_bytes` | 262,144 | 16,384..262,144 bytes |
| `max_input_audio_frames` | 1,600 | 160..1,600 frames |
| `max_output_audio_frames` | 4,800 | 160..4,800 pre-negotiation cap |
| `max_text_bytes` | 16,384 | 1,024..65,536 bytes |
| `max_input_frames` | 1,920,000 | 16,000..1,920,000 frames |
| `idle_timeout_ms` | 120,000 | 10,000..600,000 ms |

After output negotiation, the server advertises the smaller of the configured
output cap and 100 ms of selected-rate audio; it must be at least 10 ms.

`ClientConfig` fields:

| Field | Type | Default |
| --- | --- | --- |
| `uri` | `str | None` | required; may be `None` with discovery |
| `output_formats` | `tuple[AudioFormat, ...]` | required |
| `discovery_service_id` | `str | None` | `None` |
| `discovery_timeout_s` | positive float | `10.0` |
| `ssl_context` | `ssl.SSLContext | None` | `None` |
| `input_queue_frames` | positive int | `16_000` |
| `playback_queue_ms` | positive int | `2_000` |
| `waiting_pre_roll_frames` | nonnegative int | `8_000` |
| `connect_timeout_s` | `(0, 10]` | `10.0` |
| `handshake_timeout_s` | `(0, 5]` | `5.0` |
| `close_timeout_s` | `(0, 10]` | `10.0` |
| `ping_interval_s` | `(0, 20]` | `20.0` |
| `ping_timeout_s` | `(0, 20]` | `20.0` |
| `websocket_max_queue` | int `1..16` | `16` |
| `websocket_write_limit` | int `1..65_536` | `65_536` |
| `reconnect` | bool | `False` |
| `reconnect_initial_s` | positive float | `0.5` |
| `reconnect_max_s` | positive float | `30.0` |
| `agent` | 1..64 UTF-8 bytes | `"lumivox-linklab"` |

`uri` takes precedence over discovery. Service IDs match
`[a-z0-9][a-z0-9-]{0,62}`. Output formats are a 1..3-item unique canonical
preference subset of 24/48/16 kHz and include 16 kHz.

`ServerConfig` fields:

| Field | Type | Default |
| --- | --- | --- |
| `port` | int `1..65_535` | required |
| `output_formats` | `tuple[AudioFormat, ...]` | required |
| `host` | str | `"127.0.0.1"` |
| `discovery_service_id` | `str | None` | `None` |
| `ssl_context` | `ssl.SSLContext | None` | `None` |
| `limits` | `ConnectionLimits` | protocol defaults |
| `max_connections` | positive int | `8` |
| `input_queue_frames` | positive int | `32_000` |
| `output_queue_ms` | positive int | `500` |
| `waiting_timeout_s` | positive float | `60.0` |
| `input_timeout_s` | positive float | `120.0` |
| `processing_timeout_s` | positive float | `120.0` |
| `handshake_timeout_s` | `(0, 5]` | `5.0` |
| `close_timeout_s` | `(0, 10]` | `10.0` |
| `ping_interval_s` | `(0, 20]` | `20.0` |
| `ping_timeout_s` | `(0, 20]` | `20.0` |
| `websocket_max_queue` | int `1..16` | `16` |
| `websocket_write_limit` | int `1..65_536` | `65_536` |
| `agent` | 1..64 UTF-8 bytes | `"lumivox-linklab"` |

Server formats are 1..3 unique supported formats and include 16 kHz; tuple order
is not preference order. Discovery cannot advertise a loopback-only host.

## Values And Types

- `ConversationId`, `InputId`, `ResponseId`, `OutputId`: typed protocol IDs in
  `1..4_294_967_295`.
- `ReadableBuffer`: contiguous readable Python buffer-protocol input type.
- `AnnotatedAudio`: constructor `(audio, generation, discontinuity, speech,
  activated, wake_word=None)` creates a Wakelab-like client ingress value.
  `generation` is
  `0..4_294_967_295`; audio is nonempty, frame-aligned 16 kHz mono PCM S16LE.
- `ProtocolStateSnapshot`: constructor `(connection_state, conversation_id,
  input_id, response_id, output_id, coarse_state, state_revision)` represents
  immutable validator state.
- `ProtocolTombstone`: constructor `(kind, conversation_id, input_id,
  response_id, output_id, terminal_state)` creates an immutable terminal-object
  record.

`ClientMessage`, `ServerMessage`, and `Message` are union type aliases for the
semantic message classes listed below.

## Enums

| Enum | Values |
| --- | --- |
| `AudioSubmitResult` | `accepted`, `ignored_inactive`, `ignored_waiting_silence`, `closed_input`, `overflow` |
| `ConnectionState` | `disconnected`, `handshaking`, `ready`, `closing` |
| `CoarseState` | `waiting`, `listening`, `processing`, `responding` |
| `InputStartReason` | `activation`, `speech`, `barge_in` |
| `InputAbortReason` | `discontinuity`, `overflow`, `capture_failed`, `shutdown` |
| `InputCloseReason` | `endpoint`, `max_duration`, `no_speech`, `failed` |
| `PlaybackPosition` | `exact`, `estimated` |
| `PlaybackInterruptReason` | `barge_in`, `local_cancel`, `playback_failed`, `overflow`, `shutdown` |
| `ConversationCancelReason` | `user`, `shutdown`, `client_failed` |
| `ResponseCancelReason` | `barge_in`, `local_cancel`, `conversation_cancelled`, `generation_failed`, `tts_failed`, `playback_failed`, `overflow`, `shutdown` |
| `ConversationEndReason` | `completed`, `cancelled`, `idle_timeout`, `client_failed`, `server_failed`, `playback_failed` |
| `ErrorScope` | `connection`, `conversation`, `input`, `response` |
| `MessageDirection` | `client_to_server`, `server_to_client` |
| `EndpointRole` | `client`, `server` |
| `ProtocolObjectKind` | `conversation`, `input`, `response`, `output` |

`ErrorCode` values are `malformed_message`, `message_too_large`,
`unknown_message`, `unsupported_version`, `capability_mismatch`,
`format_mismatch`, `handshake_timeout`, `protocol_state`, `id_exhausted`,
`peer_unresponsive`, `conversation_failed`, `idle_timeout`,
`input_discontinuity`, `input_overflow`, `input_too_long`, `capture_failed`,
`stt_failed`, `processing_timeout`, `generation_failed`, `tts_failed`,
`playback_failed`, `output_overflow`, and `response_cancelled`.

## Semantic Messages And Events

| Class | Constructor fields |
| --- | --- |
| `ClientHello` | `version, capabilities, input_format, output_formats, agent=None` |
| `ServerHello` | `version, capabilities, output_format, limits, agent=None` |
| `ConversationStartedEvent` | `conversation_id, activation, wake_word=None` |
| `InputStartedEvent` | `conversation_id, input_id, reason, generation, interrupts_response_id=None` |
| `InputAudioEvent` | `conversation_id, input_id, start_frame, speech, audio` |
| `InputAbortedEvent` | `conversation_id, input_id, reason` |
| `PlaybackFinishedEvent` | `conversation_id, response_id, output_id, played_frames` |
| `PlaybackInterruptedEvent` | `conversation_id, response_id, output_id, played_frames, position, reason` |
| `ConversationCancelledEvent` | `conversation_id, reason` |
| `StateEvent` | `conversation_id, revision, state, reason=None` |
| `InputClosedEvent` | `conversation_id, input_id, accepted_end_frame, reason` |
| `TranscriptUpdateEvent` | `conversation_id, input_id, revision, text, language=None` |
| `TranscriptFinalEvent` | `conversation_id, input_id, text, language=None` |
| `ResponseStartedEvent` | `conversation_id, response_id, input_id, end_conversation` |
| `ResponseTextDeltaEvent` | `conversation_id, response_id, sequence, text` |
| `ResponseTextFinalEvent` | `conversation_id, response_id, text` |
| `OutputStartedEvent` | `conversation_id, response_id, output_id` |
| `OutputAudioEvent` | `conversation_id, response_id, output_id, start_frame, audio` |
| `OutputEndedEvent` | `conversation_id, response_id, output_id, total_frames` |
| `ResponseEndedEvent` | `conversation_id, response_id` |
| `ResponseCancelledEvent` | `conversation_id, response_id, reason` |
| `ConversationEndedEvent` | `conversation_id, reason` |
| `ErrorEvent` | `scope, code, fatal, conversation_id=None, input_id=None, response_id=None, message=None` |
| `ConnectionStateEvent` | `state, reason=None`; local only |

All PCM events store immutable `bytes`. Frame positions count samples per mono
channel, not bytes. `ErrorEvent.fatal` is always true for its named scope and ID
fields are present only when required by that scope.

## Client

`VoiceClient(config, callbacks, logger)` must be constructed in a running event
loop. It is an async context manager.

- `await connect()`: connect and complete handshake; callable once.
- `await close()`: bounded, idempotent shutdown.
- `await wait_closed()`: wait for complete termination.
- `connection_state`: read-only `ConnectionState`.
- `output_format`: read-only negotiated `AudioFormat | None`.
- `submit_annotated_audio(audio) -> AudioSubmitResult`.
- `abort_input(reason: InputAbortReason) -> bool`.
- `cancel_conversation(reason: ConversationCancelReason) -> bool`.
- `playback_finished(output_id, played_frames) -> bool`.
- `playback_interrupted(output_id, played_frames, position, reason) -> bool`.

The five synchronous methods are thread-safe, bounded, and non-blocking. `bool`
means accepted into reserved capacity, not remotely completed.

`ClientCallbacks` is runtime-checkable. All methods are async and return `None`:
`on_connection_state`, `on_conversation_state`, `on_transcript_update`,
`on_transcript_final`, `on_response_started`, `on_response_text_delta`,
`on_response_text_final`, `on_response_ended`, `on_response_cancelled`,
`on_output_started`, `on_output_audio`, `on_output_ended`,
`on_conversation_ended`, and `on_error`. Each receives its correspondingly named
event.

## Server

`VoiceServer(config, handler_factory, logger)` is an async context manager.
`await serve()` starts listening and discovery; it is callable once. `close()` is
idempotent; `wait_closed()` waits for shutdown. `handler_factory(session)` is
synchronous, non-blocking, and returns one runtime-checkable `ServerHandler` per
connection.

`ServerHandler` async methods are `on_conversation_started(session, event)`,
`on_input_started`, `on_input_audio`, `on_input_aborted`,
`on_playback_outcome` (finished or interrupted), and
`on_conversation_cancelled`.

`ServerSession` async operations:

- `close_input(input_id, reason)` snapshots the committed frame boundary.
- `update_transcript(input_id, revision, text, language=None)`.
- `finalize_transcript(input_id, text, language=None)`.
- `start_response(input_id, *, end_conversation=False) -> ResponseWriter`.
- `end_conversation(reason)` terminates all children first.
- `fail(scope, code, message=None)` selects the sole valid target for that scope.

`ResponseWriter` has read-only `response_id`; async `send_text_delta(text)`,
`finalize_text(text)`, `start_output() -> OutputWriter`, `finish()`, and
`cancel(reason)`. Application cancellation accepts only `generation_failed`,
`tts_failed`, `overflow` after output start, and `shutdown`. It is an async
context manager; normal exit finishes and exceptional exit maps to generation
failure without suppressing the original exception.

`OutputWriter` has read-only `output_id`; async `send_audio(audio)` and `finish()`.
It is an async context manager; normal exit finishes and exceptional exit maps to
TTS failure. Empty output is invalid. Writers copy buffers and become unusable
after a terminal operation.

## Codec And Validator

```python
encode_message(message: Message) -> bytes
decode_message(data: ReadableBuffer, *, direction: MessageDirection,
               limits: ConnectionLimits) -> Message
```

Encoding is canonical and schema-validating. Decoding is bounded and validates
structure/schema, not lifecycle. Both raise `CodecError` for invalid codec data.

`ProtocolValidator(direction: EndpointRole)` exposes `transport_connected()`,
`begin_close()`, `transport_disconnected()`, `accept(message)`, read-only `state`,
and read-only `tombstones`. `accept` is atomic and raises `ProtocolViolation`
without partial state mutation.

## Exceptions

`LinklabError` is the common base. Independent leaf exceptions are `CodecError`,
`ProtocolViolation`, `ConnectionClosed`, `QueueOverflow`, and `WriterClosed`.
