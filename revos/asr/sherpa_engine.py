"""sherpa-onnx backend for ASR (Zipformer transducer models)."""

from __future__ import annotations

import logging

import sherpa_onnx

from revos.device import auto_detect_device
from revos.registry import ensure_model, get

from .audio import read_waveform
from .base import BaseASR
from .result import Segment, Transcript

logger = logging.getLogger(__name__)


class SherpaOnnxASR(BaseASR):
    """ASR engine using sherpa-onnx OfflineRecognizer."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)

        if self.device == "auto":
            self.device = auto_detect_device()

        # Resolve provider for sherpa-onnx
        provider = "cpu"
        if self.device == "cuda":
            provider = "cuda"

        # Load manifest and ensure model is downloaded
        manifest = get(model_name, "asr")
        model_dir = ensure_model(manifest)

        # Build file paths from manifest
        files = manifest.files
        encoder = str(model_dir / files["encoder"])
        decoder = str(model_dir / files["decoder"])
        joiner = str(model_dir / files["joiner"])
        tokens = str(model_dir / files["tokens"])

        logger.info(
            "Loading ASR model %s (backend=%s, provider=%s)",
            model_name,
            manifest.backend,
            provider,
        )

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=2,
            sample_rate=manifest.sample_rate,
            provider=provider,
        )
        self._sample_rate = manifest.sample_rate
        self._model_id = manifest.model_url
        logger.info("ASR model %s loaded successfully", model_name)

        # Track gated model usage
        if manifest.hf_private or manifest.model_url.startswith("http"):
            from revos.usage import track_usage

            track_usage(
                event="model_loaded",
                model_id=str(model_dir),
                model_name=model_name,
                task="asr",
                hf_user=None,
                device=self.device,
            )

    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe an audio file using sherpa-onnx.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcript with text, segments, and detected language.
        """
        samples, sr = read_waveform(audio_path, target_sr=self._sample_rate)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(sr, samples)
        self._recognizer.decode_stream(stream)

        result = stream.result
        text = result.text.strip()

        # Build segments from timestamps
        # sherpa-onnx returns word-level timestamps as start times
        timestamps = result.timestamps
        words = text.split() if text else []

        segments: list[Segment] = []
        if timestamps and words:
            for i, word in enumerate(words):
                start = timestamps[i] if i < len(timestamps) else 0.0
                end = timestamps[i + 1] if i + 1 < len(timestamps) else start + 0.1
                segments.append(
                    Segment(start=start, end=end, text=word, confidence=0.0)
                )
        elif text:
            # No timestamps available — single segment
            segments.append(Segment(start=0.0, end=0.0, text=text, confidence=0.0))

        return Transcript(
            text=text,
            segments=segments,
            language=result.lang if result.lang else "",
        )
