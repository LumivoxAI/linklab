"""Runnable fake STT/LLM/TTS server using only Linklab's public API."""

from __future__ import annotations

import signal
import asyncio
import argparse

from lumivox_core.logger import Logger, LoggingConfig, get_logger, shutdown_logging, configure_logging

import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def fake_pcm(value: int, frames: int = 320) -> bytes:
    return bytes((value, 0)) * frames


class FakePipeline:
    """A per-connection fake STT/LLM/TTS pipeline."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._turns: dict[linklab.InputId, int] = {}
        self._received: set[linklab.InputId] = set()
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_turn = 1

    async def on_conversation_started(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationStartedEvent,
    ) -> None:
        self._logger.info("fake_conversation_started", conversation_id=event.conversation_id)

    async def on_input_started(self, _session: linklab.ServerSession, event: linklab.InputStartedEvent) -> None:
        self._turns[event.input_id] = self._next_turn
        self._next_turn += 1
        self._logger.info("fake_input_started", input_id=event.input_id, reason=event.reason.value)

    async def on_input_audio(self, session: linklab.ServerSession, event: linklab.InputAudioEvent) -> None:
        if event.input_id in self._received:
            return
        self._received.add(event.input_id)
        turn = self._turns[event.input_id]

        # One fake Wakelab chunk is enough for deterministic endpointing and STT.
        await session.update_transcript(event.input_id, 1, f"turn {turn}")
        await session.close_input(event.input_id, linklab.InputCloseReason.ENDPOINT)
        await session.finalize_transcript(event.input_id, f"turn {turn}")
        task = asyncio.create_task(self._respond(session, event.input_id, turn), name=f"fake-response-{turn}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _respond(self, session: linklab.ServerSession, input_id: linklab.InputId, turn: int) -> None:
        try:
            response = await session.start_response(input_id, end_conversation=turn == 2)
            await response.send_text_delta("fake ")
            await response.send_text_delta(f"answer {turn}")
            await response.finalize_text(f"fake answer {turn}")
            output = await response.start_output()
            await output.send_audio(fake_pcm(0x30 + turn))
            if turn == 1:
                # Leave time for the deterministic client to demonstrate barge-in.
                await asyncio.sleep(0.5)
                await output.send_audio(fake_pcm(0x40 + turn))
            await output.finish()
            await response.finish()
        except linklab.WriterClosed:
            self._logger.info("fake_response_cancelled", input_id=input_id)

    async def on_input_aborted(self, _session: linklab.ServerSession, event: linklab.InputAbortedEvent) -> None:
        self._logger.info("fake_input_aborted", input_id=event.input_id, reason=event.reason.value)

    async def on_playback_outcome(
        self,
        _session: linklab.ServerSession,
        event: linklab.PlaybackFinishedEvent | linklab.PlaybackInterruptedEvent,
    ) -> None:
        self._logger.info("fake_playback_accounted", output_id=event.output_id, played_frames=event.played_frames)

    async def on_conversation_cancelled(
        self,
        _session: linklab.ServerSession,
        event: linklab.ConversationCancelledEvent,
    ) -> None:
        self._logger.info("fake_conversation_cancelled", reason=event.reason.value)

    async def close(self) -> None:
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def run(host: str, port: int, service_id: str | None) -> None:
    logger = get_logger(component="fake_pipeline_server")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)

    pipelines: list[FakePipeline] = []

    def make_handler(_session: linklab.ServerSession) -> FakePipeline:
        pipeline = FakePipeline(logger)
        pipelines.append(pipeline)
        return pipeline

    config = linklab.ServerConfig(
        port=port,
        output_formats=(PCM_16K,),
        host=host,
        discovery_service_id=service_id,
        max_connections=2,
        input_queue_frames=3_200,
        output_queue_ms=100,
    )
    server = linklab.VoiceServer(config, make_handler, logger)
    try:
        async with server:
            print(f"fake server ready on {host}:{port}", flush=True)
            await stop.wait()
    finally:
        await asyncio.gather(*(pipeline.close() for pipeline in pipelines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--service-id", help="advertise this DNS-SD service ID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(LoggingConfig(application="linklab-fake-server"))
    try:
        asyncio.run(run(args.host, args.port, args.service_id))
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
