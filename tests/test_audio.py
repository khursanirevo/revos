"""Tests for audio loading utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from revos.asr.audio import read_waveform


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    sf.write(str(path), samples, sr)


def test_read_mono_wav(tmp_path: Path):
    """Test reading a mono WAV file at target sample rate."""
    sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    wav_path = tmp_path / "mono.wav"
    _write_wav(wav_path, samples, sr)

    data, out_sr = read_waveform(str(wav_path))
    assert out_sr == sr
    assert data.dtype == np.float32
    assert len(data) == sr


def test_read_stereo_converts_to_mono(tmp_path: Path):
    """Test that stereo audio is converted to mono."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t)
    right = np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([left, right])
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, stereo, sr)

    data, out_sr = read_waveform(str(wav_path))
    assert out_sr == sr
    assert data.ndim == 1
    assert len(data) == sr


def test_read_resamples_to_target_sr(tmp_path: Path):
    """Test resampling from 44100 to 16000."""
    sr = 44100
    target_sr = 16000
    samples = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
    wav_path = tmp_path / "44k.wav"
    _write_wav(wav_path, samples, sr)

    data, out_sr = read_waveform(str(wav_path), target_sr=target_sr)
    assert out_sr == target_sr
    assert len(data) == target_sr
