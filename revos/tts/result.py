"""TTS result data classes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class Audio:
    """Synthesized audio result from TTS."""

    samples: np.ndarray
    sample_rate: int

    def save(self, path: str) -> None:
        """Save audio to a file.

        Args:
            path: Output file path (e.g., "output.wav").
        """
        sf.write(path, self.samples, self.sample_rate)

    @property
    def duration(self) -> float:
        """Audio duration in seconds."""
        return len(self.samples) / self.sample_rate

    @staticmethod
    def concatenate(
        segments: list[Audio],
        silence_duration: float = 0.1,
    ) -> Audio:
        """Concatenate multiple Audio segments into one.

        Args:
            segments: Audio segments to join. All must have the same
                sample rate.
            silence_duration: Seconds of silence between segments.

        Returns:
            A single Audio with all segments joined.

        Raises:
            ValueError: If segments is empty or sample rates differ.
        """
        if not segments:
            raise ValueError("Cannot concatenate empty segment list")

        rates = {s.sample_rate for s in segments}
        if len(rates) > 1:
            raise ValueError(
                f"All segments must have the same sample rate, "
                f"got: {rates}"
            )

        sr = segments[0].sample_rate
        silence = np.zeros(int(sr * silence_duration), dtype=np.float32)

        parts = [segments[0].samples]
        for seg in segments[1:]:
            parts.append(silence)
            parts.append(seg.samples)

        return Audio(samples=np.concatenate(parts), sample_rate=sr)
