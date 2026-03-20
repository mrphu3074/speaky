"""
Custom LiveKit STT plugin wrapping mlx-community/whisper-large-v3-turbo.

Uses mlx_whisper for Apple Silicon-optimized batch transcription.
Since mlx_whisper doesn't support streaming, the AgentSession will
automatically wrap this with a StreamAdapter.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import wave
from typing import ClassVar

import numpy as np

from livekit.agents.stt import (
    STT,
    STTCapabilities,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer


class WhisperSTT(STT):
    """Speech-to-Text using mlx_whisper (Apple Silicon optimized)."""

    _model_id: str
    _language: str
    _mlx_whisper: ClassVar = None  # lazy-loaded module

    def __init__(
        self,
        *,
        model: str = "mlx-community/whisper-large-v3-turbo",
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
            ),
        )
        self._model_id = model
        self._language = language

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "mlx-whisper"

    def _ensure_module(self):
        """Lazy-import mlx_whisper to avoid loading at module level."""
        if WhisperSTT._mlx_whisper is None:
            import mlx_whisper

            WhisperSTT._mlx_whisper = mlx_whisper

    def _audio_buffer_to_wav_bytes(self, buffer: AudioBuffer) -> bytes:
        """Convert a LiveKit AudioBuffer to WAV bytes for mlx_whisper."""
        from livekit import rtc

        # Merge all frames into one
        if isinstance(buffer, list):
            merged = rtc.combine_audio_frames(buffer)
        else:
            merged = buffer

        # Extract raw PCM samples as int16
        samples = np.frombuffer(merged.data, dtype=np.int16)

        # Write to in-memory WAV
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(merged.num_channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(merged.sample_rate)
            wf.writeframes(samples.tobytes())

        return wav_io.getvalue()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> SpeechEvent:
        self._ensure_module()

        lang = language if language is not NOT_GIVEN else self._language

        # mlx_whisper expects a file path or numpy array
        # Write buffer to a temp WAV file
        wav_bytes = self._audio_buffer_to_wav_bytes(buffer)

        # Write to temp file — mlx_whisper works best with file paths
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(wav_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        try:
            # Run transcription in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: WhisperSTT._mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo=self._model_id,
                    language=lang,
                ),
            )
        finally:
            import os

            os.unlink(tmp_path)

        text = result.get("text", "").strip()

        return SpeechEvent(
            type=SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                SpeechData(
                    language=lang,
                    text=text,
                    confidence=1.0,
                ),
            ],
        )
