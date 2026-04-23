"""Tests for TTS engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from revos.tts.result import Audio


def test_audio_creation():
    samples = np.zeros(16000, dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=16000)
    assert audio.sample_rate == 16000
    assert len(audio.samples) == 16000


def test_audio_save(tmp_path: Path):
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    audio = Audio(samples=samples, sample_rate=24000)
    out_path = str(tmp_path / "test_out.wav")
    audio.save(out_path)

    import soundfile as sf

    data, sr = sf.read(out_path)
    assert sr == 24000
    assert len(data) == 24000


def test_audio_dataclass():
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=16000)
    np.testing.assert_allclose(audio.samples, [0.1, 0.2, 0.3], atol=1e-6)
