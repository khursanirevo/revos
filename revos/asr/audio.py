"""Audio loading utility for ASR."""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def read_waveform(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Read an audio file and return mono float32 samples at target sample rate.

    Args:
        path: Path to the audio file (WAV, FLAC, etc.).
        target_sr: Target sample rate. Audio will be resampled if different.

    Returns:
        Tuple of (samples as float32 ndarray, sample_rate).
    """
    data, sr = sf.read(path, dtype="float32")

    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
        logger.debug("Converted multi-channel audio to mono")

    # Simple resampling by linear interpolation if needed
    if sr != target_sr:
        duration = len(data) / sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(data) - 1, target_len)
        data = np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        sr = target_sr
        logger.debug("Resampled from %d Hz to %d Hz", sr, target_sr)

    return data, sr
