# Разработка

## Подготовка

```bash
git clone https://github.com/LumivoxAI/linklab.git
cd linklab
just postclone
```

Проект использует `uv`, layout `src/`, Ruff, strict mypy и pytest на Linux с
CPython 3.13-3.14.

| Команда | Назначение |
| --- | --- |
| `just sync` | Синхронизировать зависимости. |
| `just fmt` | Форматировать репозиторий. |
| `just lint` | Запустить Ruff. |
| `just typecheck` | Запустить strict mypy. |
| `just test` | Запустить тесты. |
| `just fixtures_check` | Проверить golden fixtures MessagePack. |
| `just precommit` | Выполнить обязательные быстрые проверки. |
| `just build` | Собрать wheel и sdist без локальных source overrides. |
| `just release` | Выполнить полный gate на Python 3.13/3.14 с аудитом и изолированной установкой артефактов. |

Golden fixtures являются артефактами протокола. Обновляйте их явно через
`just fixtures` и проверяйте изменения байтов.

## Правила архитектуры

- Публичные async-объекты привязаны к event loop создания и не создают loop.
- Потокобезопасны только документированные синхронные методы `VoiceClient`.
- Semantic values неизменяемы и не зависят от транспорта; wire maps приватны.
- На соединение приходится один reader и writer, producers используют
  ограниченные очереди.
- Сохраняйте canonical encoding, лимиты, атомарные переходы, tombstones и
  фильтрацию stale output.
- Не добавляйте зависимости от NumPy, устройств, моделей и политики диалога.
- Не логируйте PCM и полный текст.

Перед изменением прочитайте `AGENTS.md` и durable-документы `.agent/`. Если
задача меняет установку, API, конфигурацию, lifecycle, многопоточность,
безопасность, observability или workflow, в той же задаче обновите README,
английскую, русскую и LLM-документацию.

Перед завершением обычного изменения запустите `just precommit` и `just build`.
Перед релизом выполните полный gate:

```bash
just release
```
