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
