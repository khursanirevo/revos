"""Real integration tests — download models and run inference.

These tests require network access and real model downloads.
Run with: uv run pytest tests/test_integration.py -v -s
Skip in CI: uv run pytest tests/ -v -m "not slow"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# All tests in this file are slow (require real downloads/inference)
pytestmark = pytest.mark.slow


def _make_wav(path: Path, duration: float = 2.0, sr: int = 16000) -> str:
    """Generate a test WAV file with a 440Hz sine wave."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), samples, sr)
    return str(path)


# --- ASR Integration ---


class TestASRIntegration:
    """Real ASR tests using zipformer-v2 model."""

    def test_model_downloads_and_loads(self, tmp_path: Path):
        """Model downloads from GitHub and initializes."""
        from revos.asr import ASR

        asr = ASR("zipformer-v2", device="cpu")
        assert asr.model_name == "zipformer-v2"

    def test_transcribe_sine_wave(self, tmp_path: Path):
        """Transcription runs on a generated WAV file.

        Note: A sine wave won't produce meaningful text.
        This test verifies the pipeline runs without errors
        and returns the expected result structure.
        """
        from revos.asr import ASR

        wav_path = _make_wav(tmp_path / "test.wav")
        asr = ASR("zipformer-v2", device="cpu")
        result = asr.transcribe(wav_path)

        # Check result structure
        assert isinstance(result.text, str)
        assert isinstance(result.language, str)
        assert isinstance(result.segments, list)

        # Segments should have the right shape
        for seg in result.segments:
            assert isinstance(seg.start, float)
            assert isinstance(seg.end, float)
            assert isinstance(seg.text, str)
            assert seg.end >= seg.start

    def test_transcribe_to_json_via_cli(self, tmp_path: Path):
        """CLI transcribe --json produces valid JSON output."""
        import json
        import subprocess

        wav_path = _make_wav(tmp_path / "test.wav")
        result = subprocess.run(
            [
                "uv", "run", "revos", "transcribe",
                "-m", "zipformer-v2", "--json", wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "text" in data
        assert "segments" in data
        assert "language" in data

    def test_transcribe_to_srt_via_cli(self, tmp_path: Path):
        """CLI transcribe --srt produces SRT-formatted output."""
        import subprocess

        wav_path = _make_wav(tmp_path / "test.wav")
        result = subprocess.run(
            [
                "uv", "run", "revos", "transcribe",
                "-m", "zipformer-v2", "--srt", wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        # Verify SRT format (sequence number, timestamp, text)
        assert "-->" in result.stdout

    def test_model_cached_on_second_load(self, tmp_path: Path):
        """Second load uses cached model (no re-download)."""
        from revos.asr import ASR

        asr = ASR("zipformer-v2", device="cpu")
        wav_path = _make_wav(tmp_path / "test.wav")
        result1 = asr.transcribe(wav_path)

        # Second load should be fast (cached)
        asr2 = ASR("zipformer-v2", device="cpu")
        result2 = asr2.transcribe(wav_path)

        # Results should be consistent
        assert result1.text == result2.text


# --- TTS Integration ---

class TestTTSIntegration:
    """Real TTS tests using revovoice model.

    These tests require HuggingFace authentication with access
    to the Revolab/revovoice repository.
    """

    @pytest.fixture(autouse=True)
    def check_hf_auth(self):
        """Skip tests if not authenticated to HuggingFace."""
        try:
            from huggingface_hub import HfApi
            HfApi().whoami()
        except Exception:
            pytest.skip(
                "HuggingFace not authenticated. "
                "Run 'huggingface-cli login' first."
            )

    def test_model_loads(self):
        """RevoVoice model loads without errors."""
        from revos.tts import TTS

        tts = TTS("revovoice", device="cpu")
        assert tts.model_name == "revovoice"

    def test_synthesize_basic(self, tmp_path: Path):
        """Basic synthesis produces valid audio."""
        from revos.tts import TTS

        tts = TTS("revovoice", device="cpu")
        audio = tts.synthesize("Hello, world!")

        assert audio.sample_rate == 24000
        assert len(audio.samples) > 0
        assert audio.duration > 0
        assert audio.samples.dtype == np.float32

    def test_synthesize_saves_to_file(self, tmp_path: Path):
        """Synthesized audio saves to a valid WAV file."""
        from revos.tts import TTS

        tts = TTS("revovoice", device="cpu")
        out_path = str(tmp_path / "output.wav")
        tts.synthesize("Testing file save.", output_path=out_path)

        assert Path(out_path).exists()
        data, sr = sf.read(out_path)
        assert sr == 24000
        assert len(data) > 0

    def test_synthesize_long_text(self, tmp_path: Path):
        """synthesize_long splits and concatenates audio."""
        from revos.tts import TTS

        tts = TTS("revovoice", device="cpu")

        text = (
            "This is the first sentence. "
            "Here is another sentence. "
            "And a third sentence for good measure."
        )
        audio = tts.synthesize_long(
            text, max_chars=40, silence_duration=0.05
        )

        assert audio.sample_rate == 24000
        assert audio.duration > 1.0  # Should be multiple seconds
        assert len(audio.samples) > 0

    def test_synthesize_with_speed(self, tmp_path: Path):
        """Speed parameter affects audio duration."""
        from revos.tts import TTS

        tts = TTS("revovoice", device="cpu")
        text = "This is a speed test."

        audio_normal = tts.synthesize(text)
        audio_fast = tts.synthesize(text, speed=1.5)

        # Faster speed should produce shorter audio
        assert audio_fast.duration < audio_normal.duration

    def test_cli_synthesize(self, tmp_path: Path):
        """CLI synthesize produces a valid WAV file."""
        import subprocess

        out_path = str(tmp_path / "cli_output.wav")
        result = subprocess.run(
            [
                "uv", "run", "revos", "synthesize",
                "-m", "revovoice",
                "-t", "Hello from the CLI.",
                "-o", out_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert Path(out_path).exists()

    def test_cli_synthesize_long_text(self, tmp_path: Path):
        """CLI auto-detects long text and uses synthesize_long."""
        import subprocess

        long_text = (
            "RevoS is a unified Python library for speech AI. "
            "It supports automatic speech recognition and text to speech. "
            "You can use it to transcribe audio files or synthesize speech "
            "from text. It supports multiple languages and voice cloning. "
            "This text is intentionally long to test the auto-detection."
        )
        out_path = str(tmp_path / "cli_long_output.wav")
        result = subprocess.run(
            [
                "uv", "run", "revos", "synthesize",
                "-m", "revovoice",
                "-t", long_text,
                "-o", out_path,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0
        assert Path(out_path).exists()
        data, sr = sf.read(out_path)
        assert sr == 24000
        assert len(data) > sr * 2  # At least 2 seconds


# --- Catalog Integration ---


class TestCatalogIntegration:
    """Real catalog tests hitting GitHub API."""

    def test_catalog_list_real(self):
        """Catalog list fetches real manifests from GitHub."""
        from revos.catalog import list_catalog

        models = list_catalog()
        assert len(models) >= 2  # At least zipformer-v2 + revovoice
        names = [m.name for m in models]
        assert "zipformer-v2" in names
        assert "revovoice" in names

    def test_catalog_list_filter_asr(self):
        """Catalog list filters by task."""
        from revos.catalog import list_catalog

        models = list_catalog(task="asr")
        assert len(models) >= 1
        assert all(m.task == "asr" for m in models)

    def test_catalog_pull_real(self, tmp_path: Path):
        """Catalog pull downloads and installs a manifest."""
        from revos.catalog import pull_model

        models_dir = tmp_path / "models"
        import revos.catalog as cat_mod
        original = cat_mod._USER_MODELS_DIR

        cat_mod._USER_MODELS_DIR = models_dir
        try:
            dest = pull_model("zipformer-v2")
        finally:
            cat_mod._USER_MODELS_DIR = original

        assert dest.exists()
        assert dest.name.endswith(".yaml")
