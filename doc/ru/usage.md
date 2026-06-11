# Использование Linklab

## Форматы аудио

```python
import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)
PCM_24K = linklab.AudioFormat("pcm_s16le", 24_000, 1)
PCM_48K = linklab.AudioFormat("pcm_s16le", 48_000, 1)
```

Вход имеет формат mono PCM S16LE 16 кГц. Предпочтения клиента должны быть
уникальным подмножеством в порядке 24, 48, 16 кГц и включать 16 кГц. Сервер
перечисляет уникальные поддерживаемые форматы и тоже обязан включать 16 кГц.

## Клиент

Реализуйте все async-методы `ClientCallbacks`, создайте `VoiceClient` внутри
работающего event loop и передайте совместимый `lumivox_core.logger.Logger`:

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

После `connect()` согласованный формат доступен в `client.output_format`.
`close()` идемпотентен, `wait_closed()` ожидает полное завершение.

Поток захвата или Wakelab синхронно отправляет однородные аннотации:

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

Вызов потокобезопасен, ограничен по времени, не блокирует и копирует буфер до
возврата. Обрабатывайте все значения `AudioSubmitResult`. При `OVERFLOW` нельзя
повторять или воспроизводить отклонённый PCM.

Правила адаптера Wakelab:

- вызывайте Wakelab синхронно и последовательно вне Linklab;
- отправляйте каждый результат ровно один раз и по порядку;
- один chunk содержит непрерывный mono int16 PCM 16 кГц и однородные признаки;
- активация включает сохранённое ключевое слово и pre-roll, а не начинается в
  точке обнаружения;
- точно передавайте поколения и discontinuity;
- удерживайте активацию до re-arm;
- выполняйте re-arm после `ConversationEndedEvent` или disconnected-события
  `ConnectionStateEvent`.

Если подтверждённая речь принята в состоянии `responding`, немедленно остановите
и очистите физическое воспроизведение: это barge-in, ждать callback не нужно.

## Учёт воспроизведения

`on_output_audio` получает неизменяемый PCM и абсолютную позицию во фреймах.
После output вызовите один из методов:

```python
client.playback_finished(output_id, played_frames)
client.playback_interrupted(
    output_id,
    played_frames,
    linklab.PlaybackPosition.EXACT,
    linklab.PlaybackInterruptReason.LOCAL_CANCEL,
)
```

`ESTIMATED` используйте только без точного счётчика устройства. Один output
должен получить ровно один результат, кроме разрыва соединения.

## Сервер

Реализуйте `ServerHandler`. Синхронная неблокирующая фабрика создаёт отдельный
handler на каждое соединение:

```python
config = linklab.ServerConfig(port=8765, output_formats=(PCM_24K, PCM_16K))
server = linklab.VoiceServer(config, lambda session: Handler(), logger)

async with server:
    await application_stopped.wait()
```

Приложение управляет endpoint, STT, LLM и TTS:

```python
await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
await session.update_transcript(event.input_id, 1, "частичный", "ru")
await session.finalize_transcript(event.input_id, "итог", "ru")

async with await session.start_response(event.input_id) as response:
    await response.send_text_delta("Привет")
    await response.finalize_text("Привет")
    async with await response.start_output() as output:
        await output.send_audio(pcm_s16le)
```

Ревизии transcript начинаются с 1 и растут ровно на один. Итоговый transcript
отправляется после завершения input и до response. Response может содержать
текст, output, оба или ни одного; начатый output обязан содержать хотя бы один
фрейм. `end_conversation=True` завершает диалог после успешного ответа/проигрывания.

Задачи моделей принадлежат приложению. При отмене handler или `WriterClosed`
отменяйте соответствующую работу STT/LLM/TTS.

## Discovery

`discovery_service_id` - lowercase DNS label, например `production`. Сервер
публикует `_lumivox-voice._tcp.local.`, клиент с `uri=None` ищет точный instance.
Явный `uri` всегда отключает discovery. mDNS не аутентифицирован, может быть
подменён и часто не проходит между VLAN, контейнерами и VPN.

## Низкоуровневый API

Обычным приложениям нужны фасады. Инструменты могут использовать
`encode_message`, `decode_message` и `ProtocolValidator`. Передавайте правильный
`MessageDirection`, согласованные `ConnectionLimits` и все входящие/исходящие
события одному validator соответствующей `EndpointRole` строго по порядку.
