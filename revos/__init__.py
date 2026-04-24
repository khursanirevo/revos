"""RevoS — A unified Python library for speech AI.

Quick start:
    from revos.asr import ASR
    from revos.tts import TTS
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy re-exports for convenience: from revos import ASR, TTS."""
    if name == "ASR":
        from revos.asr import ASR

        return ASR
    if name == "TTS":
        from revos.tts import TTS

        return TTS
    if name == "configure_logging":
        from revos.logging_config import configure_logging

        return configure_logging
    raise AttributeError(f"module 'revos' has no attribute {name!r}")
