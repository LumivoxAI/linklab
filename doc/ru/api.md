# Справочник публичного API

Все перечисленные имена экспортируются из `lumivox_linklab`. Публичные
dataclass неизменяемы и используют slots. Поле класса `type` у событий не
передаётся в конструктор. ID - отдельные `NewType` над `int`.

## Форматы и конфигурация

`AudioFormat` имеет конструктор `(encoding, sample_rate_hz, channels)` и допускает
только `pcm_s16le`, частоты 16 000/24 000/48 000 и один канал.

Поля `ConnectionLimits`:

| Поле | По умолчанию | Диапазон |
| --- | ---: | ---: |
| `max_message_bytes` | 262 144 | 16 384..262 144 байт |
| `max_input_audio_frames` | 1 600 | 160..1 600 фреймов |
| `max_output_audio_frames` | 4 800 | cap 160..4 800 до согласования |
| `max_text_bytes` | 16 384 | 1 024..65 536 байт |
| `max_input_frames` | 1 920 000 | 16 000..1 920 000 фреймов |
| `idle_timeout_ms` | 120 000 | 10 000..600 000 мс |

После согласования output сервер объявляет минимум из cap и 100 мс выбранного
формата, но не меньше 10 мс.

Поля `ClientConfig`:

| Поле | Тип | По умолчанию |
| --- | --- | --- |
| `uri` | `str | None` | обязательное; `None` при discovery |
| `output_formats` | `tuple[AudioFormat, ...]` | обязательное |
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
| `agent` | 1..64 UTF-8 байта | `"lumivox-linklab"` |

Явный `uri` отключает discovery. Service ID соответствует
`[a-z0-9][a-z0-9-]{0,62}`. Форматы - уникальное подмножество 24/48/16 кГц в
этом порядке, размером 1..3, обязательно с 16 кГц.

Поля `ServerConfig`:

| Поле | Тип | По умолчанию |
| --- | --- | --- |
| `port` | int `1..65_535` | обязательное |
| `output_formats` | `tuple[AudioFormat, ...]` | обязательное |
| `host` | str | `"127.0.0.1"` |
| `discovery_service_id` | `str | None` | `None` |
| `ssl_context` | `ssl.SSLContext | None` | `None` |
| `limits` | `ConnectionLimits` | значения протокола |
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
| `agent` | 1..64 UTF-8 байта | `"lumivox-linklab"` |

Сервер принимает 1..3 уникальных формата с обязательными 16 кГц. Discovery
нельзя включить для loopback-only host.

## Значения и типы

- `ConversationId`, `InputId`, `ResponseId`, `OutputId`: ID
  `1..4_294_967_295`.
- `ReadableBuffer`: непрерывный читаемый Python buffer.
- `AnnotatedAudio`: конструктор `(audio, generation, discontinuity, speech,
  activated, wake_word=None)` создаёт вход клиента; generation
  `0..4_294_967_295`, audio - непустой frame-aligned mono PCM S16LE 16 кГц.
- `ProtocolStateSnapshot`: конструктор `(connection_state, conversation_id,
  input_id, response_id, output_id, coarse_state, state_revision)` создаёт
  состояние validator.
- `ProtocolTombstone`: конструктор `(kind, conversation_id, input_id,
  response_id, output_id, terminal_state)` создаёт запись terminal-объекта.

`ClientMessage`, `ServerMessage` и `Message` - union aliases классов сообщений.

## Enum

| Enum | Значения |
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

Значения `ErrorCode`: `malformed_message`, `message_too_large`,
`unknown_message`, `unsupported_version`, `capability_mismatch`,
`format_mismatch`, `handshake_timeout`, `protocol_state`, `id_exhausted`,
`peer_unresponsive`, `conversation_failed`, `idle_timeout`,
`input_discontinuity`, `input_overflow`, `input_too_long`, `capture_failed`,
`stt_failed`, `processing_timeout`, `generation_failed`, `tts_failed`,
`playback_failed`, `output_overflow`, `response_cancelled`.

## Сообщения и события

| Класс | Поля конструктора |
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
| `ConnectionStateEvent` | `state, reason=None`; только локальное событие |

PCM хранится как `bytes`; frame position считает mono samples, а не байты.

## Клиент

`VoiceClient(config, callbacks, logger)` создаётся внутри работающего event loop
и является async context manager. Методы: `connect()`, идемпотентный `close()`,
`wait_closed()`. Read-only свойства: `connection_state`, `output_format`.

Синхронные методы `submit_annotated_audio(audio) -> AudioSubmitResult`,
`abort_input(reason) -> bool`, `cancel_conversation(reason) -> bool`,
`playback_finished(output_id, played_frames) -> bool`,
`playback_interrupted(output_id, played_frames, position, reason) -> bool`
потокобезопасны, неблокирующие и ограниченные. `True` означает постановку в
reserved capacity, а не удалённое завершение.

Runtime-checkable `ClientCallbacks` содержит async-методы
`on_connection_state`, `on_conversation_state`, `on_transcript_update`,
`on_transcript_final`, `on_response_started`, `on_response_text_delta`,
`on_response_text_final`, `on_response_ended`, `on_response_cancelled`,
`on_output_started`, `on_output_audio`, `on_output_ended`,
`on_conversation_ended`, `on_error`; каждый получает одноимённый event.

## Сервер

`VoiceServer(config, handler_factory, logger)` - async context manager. `serve()`
запускает listener/discovery и вызывается один раз; `close()` идемпотентен;
`wait_closed()` ждёт shutdown. Синхронная неблокирующая
`handler_factory(session)` возвращает отдельный runtime-checkable `ServerHandler`.

Async-методы `ServerHandler`: `on_conversation_started(session, event)`,
`on_input_started`, `on_input_audio`, `on_input_aborted`,
`on_playback_outcome` и `on_conversation_cancelled`.

Async-операции `ServerSession`: `close_input(input_id, reason)`,
`update_transcript(input_id, revision, text, language=None)`,
`finalize_transcript(input_id, text, language=None)`,
`start_response(input_id, *, end_conversation=False) -> ResponseWriter`,
`end_conversation(reason)`, `fail(scope, code, message=None)`.

`ResponseWriter` имеет read-only `response_id` и async-методы
`send_text_delta`, `finalize_text`, `start_output`, `finish`, `cancel`.
Приложение может отменять только с `generation_failed`, `tts_failed`, `overflow`
после начала output или `shutdown`. Нормальный выход из context manager вызывает
finish, exception превращается в generation failure и не подавляется.

`OutputWriter` имеет read-only `output_id`, async `send_audio` и `finish`.
Нормальный выход завершает output, exception превращается в TTS failure. Пустой
output запрещён. Writers копируют буферы и после terminal operation закрыты.

## Codec, validator и исключения

`encode_message(message) -> bytes` выполняет canonical schema-validating encode.
`decode_message(data, *, direction, limits) -> Message` ограниченно декодирует
структуру и схему без изменения lifecycle.

`ProtocolValidator(direction)` предоставляет `transport_connected()`,
`begin_close()`, `transport_disconnected()`, атомарный `accept(message)`,
read-only `state` и `tombstones`.

`LinklabError` - общий базовый класс. Независимые leaf-исключения:
`CodecError`, `ProtocolViolation`, `ConnectionClosed`, `QueueOverflow`,
`WriterClosed`.
