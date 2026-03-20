"""
Custom LiveKit TTS plugin wrapping kyutai/pocket-tts.

Uses the official pocket_tts Python package for CPU-based synthesis.
This is a non-streaming TTS — audio is generated in full then returned.

Built-in voices: alba, marius, javert, jean, fantine, cosette, eponine, azelma
Extended voices via HuggingFace: jane, bill_boerst, caro_davy, etc.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import numpy as np

from livekit.agents.tts import (
    TTS,
    TTSCapabilities,
    ChunkedStream,
)
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid


# Sample rate for Pocket TTS output
POCKET_TTS_SAMPLE_RATE = 24000

# Built-in voices that exist as predefined embeddings in the pocket_tts package.
# These are resolved automatically by name.
BUILTIN_VOICES = {
    "alba",
    "marius",
    "javert",
    "jean",
    "fantine",
    "cosette",
    "eponine",
    "azelma",
}

# Extended voices available via HuggingFace voice cloning.
# Maps friendly names -> hf:// URLs for the kyutai/tts-voices repo.
HF_VOICES = {
    # voice-zero (CC0, LibriVox curated)
    "jane": "hf://kyutai/tts-voices/voice-zero/caro_davy.wav",
    "bill_boerst": "hf://kyutai/tts-voices/voice-zero/bill_boerst.wav",
    "caro_davy": "hf://kyutai/tts-voices/voice-zero/caro_davy.wav",
    "peter_yearsley": "hf://kyutai/tts-voices/voice-zero/peter_yearsley.wav",
    "stuart_bell": "hf://kyutai/tts-voices/voice-zero/stuart_bell.wav",
}


def resolve_voice(voice: str) -> str:
    """Resolve a friendly voice name to the value expected by pocket_tts.

    - Built-in voices are returned as-is (pocket_tts handles them natively).
    - Extended voices are mapped to their HuggingFace URL.
    - If the voice looks like a path or URL, it's returned as-is.
    """
    if voice in BUILTIN_VOICES:
        return voice
    if voice in HF_VOICES:
        return HF_VOICES[voice]
    # Assume it's a custom path/URL the user provided directly
    return voice


class PocketTTSChunkedStream(ChunkedStream):
    """ChunkedStream implementation for Pocket TTS."""

    def __init__(
        self,
        *,
        tts: "PocketTTS",
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: AudioEmitter) -> None:
        tts: PocketTTS = self._tts  # type: ignore

        request_id = shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=tts._playback_sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )

        # Run the CPU-bound TTS generation in an executor
        loop = asyncio.get_running_loop()
        audio_tensor = await loop.run_in_executor(
            None,
            lambda: tts._tts_model.generate_audio(tts._voice_state, self._input_text),
        )

        # Convert torch tensor → int16 PCM bytes
        audio_np = audio_tensor.numpy()
        if audio_np.dtype != np.int16:
            # Pocket TTS outputs float32 in [-1, 1] range
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)

        output_emitter.push(audio_np.tobytes())
        output_emitter.flush()


class PocketTTS(TTS):
    """Text-to-Speech using kyutai/pocket-tts."""

    _tts_model: ClassVar = None  # shared model instance
    _voice_state: ClassVar = None
    _initialized: ClassVar[bool] = False

    def __init__(
        self,
        *,
        voice: str = "eponine",
        temp: float = 0.7,
        speed: float = 1.0,
    ) -> None:
        # Lower speed = report lower sample rate = slower playback.
        # e.g. speed=0.85 → 24000*0.85 = 20400 Hz → plays ~15% slower.
        self._playback_sample_rate = int(POCKET_TTS_SAMPLE_RATE * speed)
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=self._playback_sample_rate,
            num_channels=1,
        )
        self._voice = voice
        self._temp = temp
        self._speed = speed
        self._resolved_voice = resolve_voice(voice)

        # Initialize model eagerly (on first instance creation)
        if not PocketTTS._initialized:
            self._load_model()

    def _load_model(self) -> None:
        """Load the Pocket TTS model and prepare the voice."""
        import logging
        from pocket_tts import TTSModel

        logger = logging.getLogger(__name__)
        logger.info("Loading Pocket TTS model...")
        PocketTTS._tts_model = TTSModel.load_model(temp=self._temp)

        logger.info(
            "Preparing voice '%s' (resolved to: %s)",
            self._voice,
            self._resolved_voice,
        )
        PocketTTS._voice_state = PocketTTS._tts_model.get_state_for_audio_prompt(
            self._resolved_voice
        )
        PocketTTS._initialized = True

    @property
    def model(self) -> str:
        return "kyutai/pocket-tts"

    @property
    def provider(self) -> str:
        return "pocket-tts"

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return PocketTTSChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )
