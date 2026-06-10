# Linklab Documentation

Linklab is a Python library that connects local voice-agent clients and servers.
It owns the wire protocol, IDs, connection and conversation state machines,
audio framing, bounded transport queues, and WebSocket lifecycle.

Use Linklab when an application needs to send annotated microphone PCM to a
voice-agent server and receive transcript, generated text, and synthesized PCM.
The client-facing integration normally connects Wakelab-like annotations and an
audio player. The server-facing integration connects STT, LLM, and TTS stages.

## Contents

- [Installation](installation.md): supported runtime and dependency setup.
- [Using Linklab](usage.md): client, server, Wakelab, playback, and lifecycle
  integration.
- [Public API](api.md): every exported symbol, configuration field, callback,
  handler, and writer operation.
- [Operations and security](operations.md): deployment, TLS, discovery,
  overload, errors, logging, and shutdown.
- [Development](development.md): repository setup, checks, architecture, and
  documentation maintenance.
- [LLM integration instructions](../llm/linklab.md): self-contained,
  implementation-oriented library instructions for use by another project.
- [Russian documentation](../ru/index.md).

## Scope

Linklab includes:

- immutable typed semantic values and events;
- canonical MessagePack encoding and bounded decoding;
- protocol lifecycle validation;
- an asyncio `VoiceClient` with thread-safe non-blocking audio ingress;
- an asyncio `VoiceServer` with an isolated session per connection;
- bounded queues, keepalive, reconnect, optional DNS-SD/mDNS discovery, and
  structured logging.

Linklab does not include device I/O, Wakelab, STT, LLM, TTS, VAD, wake-word
detection, AEC, resampling, authentication, pairing, or conversation policy.
Only mono PCM S16LE is supported: input is always 16 kHz; output is negotiated
from 24 kHz, 48 kHz, and 16 kHz.
