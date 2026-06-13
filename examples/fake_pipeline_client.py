"""Runnable fake Wakelab/playback client using only Linklab's public API."""

from __future__ import annotations

import asyncio
import argparse

from lumivox_core.logger import Logger, LoggingConfig, get_logger, shutdown_logging, configure_logging

import lumivox_linklab as linklab

PCM_16K = linklab.AudioFormat("pcm_s16le", 16_000, 1)


def fake_pcm(value: int, frames: int = 320) -> bytes:
    return bytes((value, 0)) * frames


class FakePlayback:
    """Consumes output immediately and reports exact frame accounting."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self.client: linklab.VoiceClient | None = None
        self.wakelab: FakeWakelab | None = None
        self.state: linklab.CoarseState | None = None
        self.output_id: linklab.OutputId | None = None
        self.played_frames = 0
        self.first_audio = asyncio.Event()
        self.conversation_ended = asyncio.Event()

    async def on_connection_state(self, event: linklab.ConnectionStateEvent) -> None:
        self._logger.info("fake_connection_state", state=event.state.value)
        if event.state is linklab.ConnectionState.DISCONNECTED and self.wakelab is not None:
            self.wakelab.rearm()

    async def on_conversation_state(self, event: linklab.StateEvent) -> None:
        self.state = event.state
        self._logger.info("fake_conversation_state", state=event.state.value)

    async def on_transcript_update(self, event: linklab.TranscriptUpdateEvent) -> None:
        self._logger.info("fake_transcript_update", revision=event.revision)

    async def on_transcript_final(self, event: linklab.TranscriptFinalEvent) -> None:
        self._logger.info("fake_transcript_final", input_id=event.input_id)

    async def on_response_started(self, event: linklab.ResponseStartedEvent) -> None:
        self._logger.info("fake_response_started", response_id=event.response_id)

    async def on_response_text_delta(self, event: linklab.ResponseTextDeltaEvent) -> None:
        self._logger.debug("fake_response_text_delta", sequence=event.sequence)

    async def on_response_text_final(self, event: linklab.ResponseTextFinalEvent) -> None:
        self._logger.info("fake_response_text_final", response_id=event.response_id)

    async def on_response_ended(self, event: linklab.ResponseEndedEvent) -> None:
        self._logger.info("fake_response_ended", response_id=event.response_id)

    async def on_response_cancelled(self, event: linklab.ResponseCancelledEvent) -> None:
        self._logger.info("fake_response_cancelled", reason=event.reason.value)

    async def on_output_started(self, event: linklab.OutputStartedEvent) -> None:
        self.output_id = event.output_id
        self.played_frames = 0

    async def on_output_audio(self, event: linklab.OutputAudioEvent) -> None:
        if event.output_id != self.output_id or event.start_frame != self.played_frames:
            raise RuntimeError("fake playback received discontinuous output")
        self.played_frames += len(event.audio) // 2
        self.first_audio.set()

    async def on_output_ended(self, event: linklab.OutputEndedEvent) -> None:
        if self.client is None or not self.client.playback_finished(event.output_id, event.total_frames):
            raise RuntimeError("fake playback accounting was rejected")
        self._logger.info("fake_playback_finished", output_id=event.output_id, played_frames=event.total_frames)
        self.output_id = None

    async def on_conversation_ended(self, event: linklab.ConversationEndedEvent) -> None:
        self._logger.info("fake_conversation_ended", reason=event.reason.value)
        if self.wakelab is not None:
            self.wakelab.rearm()
        self.conversation_ended.set()

    async def on_error(self, event: linklab.ErrorEvent) -> None:
        self._logger.error("fake_protocol_error", scope=event.scope.value, code=event.code.value)

    def interrupt_for_barge_in(self) -> None:
        if self.client is None or self.output_id is None:
            raise RuntimeError("barge-in requires active playback")
        accepted = self.client.playback_interrupted(
            self.output_id,
            self.played_frames,
            linklab.PlaybackPosition.EXACT,
            linklab.PlaybackInterruptReason.BARGE_IN,
        )
        if not accepted:
            raise RuntimeError("barge-in playback accounting was rejected")
        self.output_id = None


class FakeWakelab:
    """Emits ordered annotated chunks and models activation re-arm."""

    def __init__(self, client: linklab.VoiceClient, playback: FakePlayback) -> None:
        self._client = client
        self._playback = playback
        self._armed = True
        self._generation = 0

    def activate(self) -> None:
        if not self._armed:
            raise RuntimeError("fake Wakelab is not armed")
        self._armed = False
        self._submit(fake_pcm(0x10), activated=True, wake_word="lumivox")

    def barge_in(self) -> None:
        self._submit(fake_pcm(0x20), activated=True)

    def _submit(self, audio: bytes, *, activated: bool, wake_word: str | None = None) -> None:
        result = self._client.submit_annotated_audio(
            linklab.AnnotatedAudio(audio, self._generation, False, True, activated, wake_word)
        )
        if result is not linklab.AudioSubmitResult.ACCEPTED:
            raise RuntimeError(f"fake Wakelab chunk was not accepted: {result.value}")
        if self._playback.state is linklab.CoarseState.RESPONDING and self._playback.output_id is not None:
            self._playback.interrupt_for_barge_in()

    def rearm(self) -> None:
        self._armed = True


async def run(uri: str | None, service_id: str | None) -> None:
    logger = get_logger(component="fake_pipeline_client")
    callbacks = FakePlayback(logger)
    config = linklab.ClientConfig(
        uri=uri,
        output_formats=(PCM_16K,),
        discovery_service_id=service_id,
        input_queue_frames=3_200,
        playback_queue_ms=100,
        waiting_pre_roll_frames=1_600,
    )
    client = linklab.VoiceClient(config, callbacks, logger)
    callbacks.client = client
    callbacks.wakelab = FakeWakelab(client, callbacks)

    async with client:
        callbacks.wakelab.activate()
        async with asyncio.timeout(5):
            await callbacks.first_audio.wait()
        callbacks.wakelab.barge_in()
        async with asyncio.timeout(5):
            await callbacks.conversation_ended.wait()
    print("fake client completed activation, barge-in, playback accounting, and re-arm", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--service-id", help="discover this DNS-SD service ID instead of using --uri")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uri = None if args.service_id is not None else args.uri
    configure_logging(LoggingConfig(application="linklab-fake-client"))
    try:
        asyncio.run(run(uri, args.service_id))
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
