# Lumivox Linklab

Linklab is the typed communication library for local Lumivox voice-agent clients
and servers. It provides an asyncio WebSocket client and server, strict
MessagePack messages, lifecycle validation, bounded queues, reconnect, local
service discovery, playback accounting, and safe structured observability.

Linklab transports voice-agent data. It does not capture or play audio and does
not implement wake-word detection, VAD, STT, LLM, or TTS models. Applications
connect those components through the public callback, handler, and writer APIs.

Requirements: Linux and CPython 3.13 or 3.14.

## Documentation

- [Documentation overview](doc/en/index.md)
- [Installation](doc/en/installation.md)
- [Using Linklab](doc/en/usage.md)
- [Public API reference](doc/en/api.md)
- [Operations and security](doc/en/operations.md)
- [Development](doc/en/development.md)
- [Runnable fake-pipeline examples](examples/README.md)
- [LLM integration instructions](doc/llm/linklab.md)
- [Документация на русском языке](doc/ru/index.md)

## Install From Git

```bash
pip install "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

For reproducible applications, replace `master` with a tested commit SHA in the
application lockfile.

## License

Apache-2.0. See [LICENSE](LICENSE).
