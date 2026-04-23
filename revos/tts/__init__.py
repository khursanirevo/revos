"""RevoS TTS — Text-to-Speech.

Usage:
    from revos.tts import TTS

    tts = TTS('omnivoice')
    audio = tts.synthesize('Hello, world!')
    audio.save('output.wav')
"""

from __future__ import annotations

from revos.registry import get

from .base import BaseTTS
from .result import Audio


def TTS(model_name: str, device: str = "auto") -> BaseTTS:
    """Create a TTS engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the TTS model (e.g. "omnivoice").
        device: Compute device — "auto", "cpu", or "cuda".

    Returns:
        A BaseTTS instance ready for synthesis.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If the model backend is not supported.
    """
    manifest = get(model_name, "tts")

    if manifest.backend == "omnivoice":
        from .omnivoice_engine import OmniVoiceTTS

        return OmniVoiceTTS(model_name, device)

    raise ValueError(
        f"Unsupported TTS backend: {manifest.backend} for model '{model_name}'"
    )


__all__ = ["TTS", "BaseTTS", "Audio"]
