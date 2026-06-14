import sys
import signal
import socket
import asyncio
from typing import cast
from pathlib import Path


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return cast(int, listener.getsockname()[1])


def test_fake_pipeline_examples_smoke() -> None:
    async def scenario() -> None:
        root = Path(__file__).parents[1]
        port = unused_port()
        server = await asyncio.create_subprocess_exec(
            sys.executable,
            "examples/fake_pipeline_server.py",
            "--port",
            str(port),
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert server.stdout is not None
        try:
            async with asyncio.timeout(5):
                while b"fake server ready" not in await server.stdout.readline():
                    if server.returncode is not None:
                        raise AssertionError("fake server exited before becoming ready")

            client = await asyncio.create_subprocess_exec(
                sys.executable,
                "examples/fake_pipeline_client.py",
                "--uri",
                f"ws://127.0.0.1:{port}",
                cwd=root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async with asyncio.timeout(10):
                client_output, _ = await client.communicate()
            assert client.returncode == 0, client_output.decode()
            assert b"fake client completed" in client_output
            assert b"fake answer" not in client_output
            assert b"turn 1" not in client_output
        finally:
            if server.returncode is None:
                server.send_signal(signal.SIGTERM)
            async with asyncio.timeout(5):
                server_output, _ = await server.communicate()
            assert server.returncode == 0, server_output.decode()
            assert b"fake answer" not in server_output
            assert b"turn 1" not in server_output

    asyncio.run(scenario(), debug=True)
