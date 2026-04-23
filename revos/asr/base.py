"""Abstract base class for ASR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .result import Transcript


class BaseASR(ABC):
    """Base class for automatic speech recognition engines."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def transcribe(self, audio_path: str) -> Transcript:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Transcript object with text, segments, and language.
        """
        ...
