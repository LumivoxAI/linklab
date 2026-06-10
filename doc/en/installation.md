# Installation

## Requirements

- Linux
- CPython `>=3.13,<3.15`
- an asyncio application

Linklab has no NumPy dependency. PCM is exchanged as contiguous readable buffer
objects and immutable `bytes`.

## Install From The Repository

The current development version is the `master` branch:

```bash
pip install "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

With `uv`, add the same dependency to a project:

```bash
uv add "lumivox-linklab @ git+https://github.com/LumivoxAI/linklab.git@master"
```

Tracking a moving branch is suitable for development only. Production projects
should pin the tested Git commit in their dependency declaration or lockfile.

The package is imported as:

```python
import lumivox_linklab as linklab
```

## Install A Working Copy

```bash
git clone https://github.com/LumivoxAI/linklab.git
cd linklab
just postclone
```

`just postclone` creates/synchronizes the environment using `uv`. See
[Development](development.md) for verification commands.

## TLS Prerequisites

Plain `ws://` is intended only for a trusted isolated LAN. For `wss://`, create
and configure an `ssl.SSLContext` in the application, then pass it as
`ssl_context` in `ClientConfig` or `ServerConfig`. Linklab does not provision
certificates or authenticate clients.
