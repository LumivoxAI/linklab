# Development

## Setup

```bash
git clone https://github.com/LumivoxAI/linklab.git
cd linklab
just postclone
```

The project uses `uv`, a `src/` package layout, Ruff, strict mypy, and pytest.
The supported interpreter range is CPython 3.13 through 3.14 on Linux.

## Commands

| Command | Purpose |
| --- | --- |
| `just sync` | Synchronize normal and development dependencies. |
| `just fmt` | Format the repository. |
| `just lint` | Run Ruff lint checks. |
| `just typecheck` | Run strict mypy. |
| `just test` | Run tests. |
| `just fixtures_check` | Verify canonical MessagePack golden fixtures. |
| `just precommit` | Run all required fast checks. |
| `just build` | Build wheel and sdist without local source overrides. |

Golden fixtures are protocol artifacts. Regenerate them only deliberately with
`just fixtures`, then review byte changes.

## Architecture Rules

- Public async objects bind to the running event loop where they are created and
  never create or own an event loop.
- Only the documented synchronous `VoiceClient` submission/control/playback
  methods are thread-safe.
- Keep semantic values transport-neutral and frozen; wire maps remain private.
- Keep one reader and one writer task per connection and route producers through
  bounded queues.
- Preserve canonical encoding, strict decoder limits, atomic validator
  transitions, tombstones, and stale-output rejection.
- Do not add NumPy, device, model, or conversation-policy dependencies.
- Do not log PCM or full transcript/response content.

## Changing Public Behavior

Read `AGENTS.md` and the durable documents under `.agent/` before making a
change. Update tests and golden fixtures where applicable. If a task changes
installation, public API, configuration, lifecycle, concurrency, security,
observability, or development workflow, update the affected English, Russian,
README, and LLM documentation in the same task.

Before considering a change complete, run:

```bash
just precommit
just build
```
