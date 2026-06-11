# Установка

## Требования

- Linux
- CPython `>=3.13,<3.15`
- приложение на asyncio

NumPy не требуется. PCM передаётся через непрерывные читаемые buffer-объекты и
неизменяемые `bytes`.

## Установка из репозитория

Текущая версия разработки находится в ветке `master`:

```bash
pip install "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

Для проекта на `uv`:

```bash
uv add "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

В рабочем окружении закрепляйте проверанный commit SHA в зависимости или
lock-файле вместо подвижной ветки. Импорт пакета:

```python
import lumivox_linklab as linklab
```

## Рабочая копия

```bash
git clone https://github.com/LumivoxAI/linklab.git
cd linklab
just postclone
```

Команда создаёт/синхронизирует окружение через `uv`. Остальные команды описаны в
[разделе разработки](development.md).

## TLS

`ws://` допустим только в доверенной изолированной локальной сети. Для `wss://`
создайте `ssl.SSLContext` в приложении и передайте его как `ssl_context` в
`ClientConfig` или `ServerConfig`. Linklab не выпускает сертификаты и не
аутентифицирует клиентов.
