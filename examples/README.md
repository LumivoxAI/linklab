# Runnable Fake Pipeline

These examples connect the public `VoiceClient` and `VoiceServer` facades to
bounded fake Wakelab, STT, LLM, TTS, and playback components. They require no
audio device, model, multicast, or external service.

From the repository root, start the loopback server:

```bash
uv run python examples/fake_pipeline_server.py
```

Then run the deterministic client in another terminal:

```bash
uv run python examples/fake_pipeline_client.py
```

The client performs activation, receives streaming text and PCM, barges into the
first response, reports exact interrupted/finished playback positions, completes
a second response, rearms fake Wakelab, and shuts down cleanly. Use `Ctrl+C` to
stop the server. Queue capacities are finite, and normal logs contain IDs,
states, frame counts, and reasons rather than PCM or full transcript/response
text.

## Opt-In LAN Discovery

Use the same service ID on both processes. Discovery requires a non-loopback
server bind and a LAN that permits mDNS multicast:

```bash
uv run python examples/fake_pipeline_server.py --host 0.0.0.0 --service-id development
uv run python examples/fake_pipeline_client.py --service-id development
```

Change only the matching `--service-id` values to select another development or
production instance. DNS-SD is an unauthenticated routing hint; do not expose the
plain WebSocket listener to the Internet or send secrets through it.
