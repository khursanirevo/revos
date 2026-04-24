"""Tests for text splitting, audio concatenation, and synthesize_long."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from revos.tts.base import _split_text
from revos.tts.result import Audio

# --- Text splitting ---


def test_split_text_short():
    """Short text is returned as a single chunk."""
    assert _split_text("Hello world.") == ["Hello world."]


def test_split_text_empty():
    """Empty text returns empty list."""
    assert _split_text("") == []
    assert _split_text("   ") == []


def test_split_text_multiple_sentences():
    """Text with multiple sentences splits on sentence boundaries."""
    text = "First sentence. Second sentence. Third sentence."
    chunks = _split_text(text, max_chars=30)
    assert len(chunks) >= 2
    # All chunks should be within max_chars
    for chunk in chunks:
        assert len(chunk) <= 30
    # Reassembled text should contain all original content
    assert " ".join(chunks) == text


def test_split_text_respects_max_chars():
    """No chunk exceeds max_chars."""
    text = (
        "This is sentence one. This is sentence two. "
        "This is sentence three. This is sentence four."
    )
    chunks = _split_text(text, max_chars=45)
    for chunk in chunks:
        assert len(chunk) <= 45


def test_split_text_long_sentence_splits_at_commas():
    """A single sentence exceeding max_chars splits at commas."""
    text = "A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P"
    chunks = _split_text(text, max_chars=20)
    for chunk in chunks:
        assert len(chunk) <= 20
    # All parts should be present
    joined = " ".join(chunks)
    for letter in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        assert letter in joined


def test_split_text_multilingual_punctuation():
    """Handles Chinese/Arabic sentence-ending punctuation."""
    text = "你好世界。这是第二句。这是第三句。"
    chunks = _split_text(text, max_chars=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 10


def test_split_text_single_long_word():
    """A single very long word is returned as-is."""
    text = "supercalifragilisticexpialidocious" * 5
    chunks = _split_text(text, max_chars=20)
    # Falls back to returning the chunk even if over max_chars
    assert len(chunks) >= 1


# --- Audio concatenation ---


def test_audio_concatenate_basic():
    """Two audio segments concatenate correctly."""
    sr = 16000
    a1 = Audio(samples=np.ones(sr, dtype=np.float32), sample_rate=sr)
    a2 = Audio(samples=np.ones(sr, dtype=np.float32) * 2, sample_rate=sr)
    result = Audio.concatenate([a1, a2])
    assert result.sample_rate == sr
    # 1s + 0.1s silence + 1s = 2.1s
    expected_len = sr * 2 + int(sr * 0.1)
    assert len(result.samples) == expected_len


def test_audio_concatenate_no_silence():
    """Zero silence duration joins segments directly."""
    sr = 16000
    a1 = Audio(samples=np.ones(sr, dtype=np.float32), sample_rate=sr)
    a2 = Audio(samples=np.ones(sr, dtype=np.float32) * 2, sample_rate=sr)
    result = Audio.concatenate([a1, a2], silence_duration=0.0)
    assert len(result.samples) == sr * 2


def test_audio_concatenate_single():
    """Single segment returns equivalent audio."""
    sr = 16000
    a = Audio(samples=np.ones(sr, dtype=np.float32), sample_rate=sr)
    result = Audio.concatenate([a])
    assert len(result.samples) == sr
    assert result.sample_rate == sr


def test_audio_concatenate_empty_raises():
    """Empty segment list raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        Audio.concatenate([])


def test_audio_concatenate_mismatched_rates():
    """Mismatched sample rates raises ValueError."""
    a1 = Audio(samples=np.ones(16000, dtype=np.float32), sample_rate=16000)
    a2 = Audio(samples=np.ones(24000, dtype=np.float32), sample_rate=24000)
    with pytest.raises(ValueError, match="sample rate"):
        Audio.concatenate([a1, a2])


# --- Audio duration property ---


def test_audio_duration():
    """Duration property returns correct seconds."""
    sr = 16000
    a = Audio(samples=np.zeros(sr * 5, dtype=np.float32), sample_rate=sr)
    assert a.duration == 5.0


# --- synthesize_long integration ---


@pytest.fixture(autouse=True)
def clear_registry():
    from revos.registry.registry import _models

    _models.clear()
    yield
    _models.clear()


def _make_mock_omnivoice():
    mock_module = ModuleType("omnivoice")
    mock_cls = MagicMock()
    mock_model = MagicMock()
    mock_cls.from_pretrained.return_value = mock_model
    mock_module.OmniVoice = mock_cls
    return mock_module, mock_cls, mock_model


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_synthesize_long(mock_hf_user):
    """synthesize_long splits text and returns concatenated audio."""
    from revos.registry.manifest import ModelManifest
    from revos.registry.registry import register
    from revos.tts.revovoice_engine import RevoVoiceTTS

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

    mock_module, _, mock_model = _make_mock_omnivoice()
    # Return different audio each call
    mock_model.generate.side_effect = [
        [np.random.randn(24000).astype(np.float32) * 0.1]
        for _ in range(10)
    ]

    text = (
        "First sentence here. Second sentence here. "
        "Third sentence here. Fourth sentence here."
    )

    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        engine = RevoVoiceTTS("test-tts", device="cpu")
        result = engine.synthesize_long(text, max_chars=30)

    assert isinstance(result, Audio)
    assert result.sample_rate == 24000
    # Should have been called multiple times (split into chunks)
    assert mock_model.generate.call_count >= 2


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_synthesize_long_saves_to_file(mock_hf_user, tmp_path: Path):
    """synthesize_long saves concatenated audio to file."""
    from revos.registry.manifest import ModelManifest
    from revos.registry.registry import register
    from revos.tts.revovoice_engine import RevoVoiceTTS

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

    mock_module, _, mock_model = _make_mock_omnivoice()
    mock_model.generate.side_effect = [
        [np.random.randn(24000).astype(np.float32) * 0.1]
        for _ in range(10)
    ]

    with patch.dict(sys.modules, {"omnivoice": mock_module}):
        engine = RevoVoiceTTS("test-tts", device="cpu")
        out_path = str(tmp_path / "long_output.wav")
        result = engine.synthesize_long(
            "Sentence one. Sentence two. Sentence three.",
            output_path=out_path,
            max_chars=20,
        )

    assert (tmp_path / "long_output.wav").exists()
    assert result.duration > 0


@patch("revos.tts.revovoice_engine._get_hf_user", return_value=None)
def test_synthesize_long_empty_raises(mock_hf_user):
    """synthesize_long raises on empty text."""
    from revos.registry.manifest import ModelManifest
    from revos.registry.registry import register
    from revos.tts.revovoice_engine import RevoVoiceTTS

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
        engine = RevoVoiceTTS("test-tts", device="cpu")
        with pytest.raises(ValueError, match="empty"):
            engine.synthesize_long("")
