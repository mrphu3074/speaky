"""
Custom LiveKit turn detector plugin wrapping pipecat-ai/smart-turn-v3.

Smart Turn v3 is an audio-based turn detection model (Whisper Tiny + linear
classifier). It analyzes up to 8 seconds of 16kHz mono PCM audio and predicts
whether the user has finished their turn (probability 0–1).

Since LiveKit's built-in turn detector interface (EOUModelBase) is text-based,
we integrate Smart Turn v3 as a standalone audio analyzer that works alongside
the VAD system. The agent hooks into VAD end-of-speech events, feeds the audio
buffer to Smart Turn, and only commits the turn if Smart Turn agrees.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import WhisperFeatureExtractor

logger = logging.getLogger(__name__)

# Smart Turn v3 constants
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_ONNX = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_SAMPLE_RATE = 16000
SMART_TURN_MAX_SECONDS = 8


def _truncate_or_pad(audio: np.ndarray, n_seconds: int = 8, sample_rate: int = 16000) -> np.ndarray:
    """Truncate audio to last n seconds or pad with zeros at the beginning."""
    max_samples = n_seconds * sample_rate
    if len(audio) > max_samples:
        return audio[-max_samples:]
    elif len(audio) < max_samples:
        padding = max_samples - len(audio)
        return np.pad(audio, (padding, 0), mode="constant", constant_values=0)
    return audio


class SmartTurnDetector:
    """
    Audio-based turn detector using pipecat-ai/smart-turn-v3.

    Usage:
        detector = SmartTurnDetector()
        probability = await detector.predict(audio_int16, sample_rate=16000)
        if probability > detector.threshold:
            # User has finished speaking
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        onnx_filename: str = SMART_TURN_ONNX,
    ) -> None:
        self.threshold = threshold
        self._onnx_filename = onnx_filename
        self._session: ort.InferenceSession | None = None
        self._feature_extractor: WhisperFeatureExtractor | None = None

    def load(self) -> "SmartTurnDetector":
        """Download and load the ONNX model. Call once at startup."""
        logger.info("Downloading Smart Turn v3 model from %s", SMART_TURN_REPO)
        local_path = hf_hub_download(
            repo_id=SMART_TURN_REPO,
            filename=self._onnx_filename,
        )

        logger.info("Loading Smart Turn v3 ONNX session from %s", local_path)
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            local_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )

        self._feature_extractor = WhisperFeatureExtractor(chunk_length=SMART_TURN_MAX_SECONDS)
        logger.info("Smart Turn v3 model loaded successfully")
        return self

    def predict_sync(self, audio_float32: np.ndarray) -> float:
        """
        Predict whether the user's turn is complete.

        Args:
            audio_float32: Numpy array of audio samples at 16kHz, float32, range [-1, 1].

        Returns:
            Probability that the turn is complete (0.0 to 1.0).
        """
        if self._session is None or self._feature_extractor is None:
            raise RuntimeError("SmartTurnDetector not loaded. Call .load() first.")

        # Truncate/pad to 8 seconds
        audio = _truncate_or_pad(audio_float32, n_seconds=SMART_TURN_MAX_SECONDS)

        # Extract Whisper features
        inputs = self._feature_extractor(
            audio,
            sampling_rate=SMART_TURN_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=SMART_TURN_MAX_SECONDS * SMART_TURN_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )

        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        # Run ONNX inference
        outputs = self._session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0].item())

        logger.debug("Smart Turn prediction: probability=%.4f (threshold=%.2f)", probability, self.threshold)
        return probability

    async def predict(self, audio_float32: np.ndarray) -> float:
        """Async wrapper — runs inference in executor to avoid blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict_sync, audio_float32)

    def is_turn_complete(self, probability: float) -> bool:
        """Check if probability exceeds the threshold."""
        return probability > self.threshold
