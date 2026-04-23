"""Shared test fixtures for revos tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_wav(tmp_path: Path) -> str:
    """Create a small test WAV file (1-second 440Hz sine wave)."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), samples, sr)
    return str(wav_path)


@pytest.fixture
def mock_recognizer():
    """Create a mock sherpa_onnx.OfflineRecognizer."""
    recognizer = MagicMock()

    # Mock result
    result = MagicMock()
    result.text = "HELLO WORLD TEST"
    result.timestamps = [0.0, 0.3, 0.6]
    result.lang = "en"
    result.words = []

    # Mock stream
    stream = MagicMock()
    stream.result = result

    recognizer.create_stream.return_value = stream
    return recognizer


@pytest.fixture
def mock_tts_model():
    """Create a mock OmniVoice model."""
    model = MagicMock()
    # generate returns list of ndarray (24kHz, ~1 second)
    audio = np.random.randn(24000).astype(np.float32) * 0.1
    model.generate.return_value = [audio]
    return model
