"""RevoS ASR — Automatic Speech Recognition.

Usage:
    from revos.asr import ASR

    asr = ASR('zipformer-v2')
    result = asr.transcribe('audio.wav')
    print(result.text)
"""

from __future__ import annotations

from revos.registry import get

from .base import BaseASR
from .result import Segment, Transcript


def ASR(model_name: str, device: str = "auto") -> BaseASR:
    """Create an ASR engine for the given model.

    Looks up the model manifest and dispatches to the appropriate backend.

    Args:
        model_name: Name of the ASR model (e.g. "zipformer-v2").
        device: Compute device — "auto", "cpu", or "cuda".

    Returns:
        A BaseASR instance ready for transcription.

    Raises:
        KeyError: If the model is not registered.
        ValueError: If the model backend is not supported.
    """
    manifest = get(model_name, "asr")

    if manifest.backend == "sherpa-onnx":
        from .sherpa_engine import SherpaOnnxASR

        return SherpaOnnxASR(model_name, device)

    raise ValueError(
        f"Unsupported ASR backend: '{manifest.backend}' for model "
        f"'{model_name}'. Supported backends: sherpa-onnx"
    )


__all__ = ["ASR", "BaseASR", "Transcript", "Segment"]
