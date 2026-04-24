"""Tests for TTS engine."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from revos.registry.manifest import ModelManifest
from revos.registry.registry import _models
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


@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    yield
    _models.clear()


def _make_mock_omnivoice():
    """Create a mock omnivoice module with OmniVoice class."""
    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_model = MagicMock()
    audio_samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_model.generate.return_value = [audio_samples]
    mock_cls.from_pretrained.return_value = mock_model
    mock_module.OmniVoice = mock_cls
    return mock_module, mock_cls, mock_model


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_revovoice_engine_synthesize(mock_hf_user, tmp_path: Path):
    """Test RevoVoiceTTS.synthesize with mocked model."""
    from revos.registry.registry import register

    register(
        ModelManifest(
            name="test-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="TestOrg/test-model",
            sample_rate=24000,
            language="en",
            description="Test TTS",
            files={},
        )
    )

    mock_module, mock_cls, mock_model = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revos.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        result = engine.synthesize("Hello world")

    assert isinstance(result, Audio)
    assert result.sample_rate == 24000
    mock_model.generate.assert_called_once_with(text="Hello world", speed=1.0)


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_revovoice_engine_save_to_file(mock_hf_user, tmp_path: Path):
    """Test RevoVoiceTTS.synthesize saves to file when output_path given."""
    from revos.registry.registry import register

    register(
        ModelManifest(
            name="test-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="TestOrg/test-model",
            sample_rate=24000,
            language="en",
            description="Test TTS",
            files={},
        )
    )

    mock_module, _, _ = _make_mock_omnivoice()
    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revos.tts.revovoice_engine import RevoVoiceTTS

        engine = RevoVoiceTTS("test-tts", device="cpu")
        out_path = str(tmp_path / "output.wav")
        engine.synthesize("Hello", output_path=out_path)

    assert (tmp_path / "output.wav").exists()


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_revovoice_engine_gated_error(mock_hf_user):
    """Test that OSError for gated repo raises clear RuntimeError."""
    from revos.registry.registry import register

    register(
        ModelManifest(
            name="gated-tts",
            task="tts",
            backend="revovoice",
            model_type="diffusion",
            model_url="Revolab/revovoice",
            sample_rate=24000,
            language="en",
            description="Gated model",
            files={},
        )
    )

    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_cls.from_pretrained.side_effect = OSError(
        "gated repo, please authenticate"
    )
    mock_module.OmniVoice = mock_cls

    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        from revos.tts.revovoice_engine import RevoVoiceTTS

        with pytest.raises(RuntimeError, match="HuggingFace authentication"):
            RevoVoiceTTS("gated-tts", device="cpu")


def test_tts_unsupported_backend():
    """Test that unsupported backend raises ValueError."""
    from revos.registry.registry import register

    register(
        ModelManifest(
            name="bad-backend",
            task="tts",
            backend="nonexistent",
            model_type="",
            model_url="",
            sample_rate=24000,
            language="en",
            description="",
        )
    )
    from revos.tts import TTS

    with pytest.raises(ValueError, match="Supported backends: revovoice"):
        TTS("bad-backend")
