"""Abstract base class for TTS engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .result import Audio


class BaseTTS(ABC):
    """Base class for text-to-speech engines."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | None = None,
        *,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            output_path: Optional path to save the audio file.
            speed: Speech speed multiplier (1.0 = normal).
            ref_audio: Optional reference audio for voice cloning.
            ref_text: Optional transcription of the reference audio.

        Returns:
            Audio object with samples and sample rate.
        """
        ...
