"""OmniVoice backend for TTS (diffusion-based zero-shot TTS)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from revos.registry import get

from .base import BaseTTS
from .result import Audio

logger = logging.getLogger(__name__)


def _get_hf_user() -> dict | None:
    """Get the current HuggingFace user info from their token.

    Returns:
        Dict with 'name' and 'token' keys, or None if not authenticated.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.whoami()
        return {"name": info.get("name", "unknown"), "fullname": info.get("fullname", "")}
    except Exception:
        return None


class OmniVoiceTTS(BaseTTS):
    """TTS engine using OmniVoice diffusion model.

    Supports voice cloning, voice design, and auto voice generation
    across 600+ languages.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)

        try:
            from omnivoice import OmniVoice  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "OmniVoice is required for TTS. "
                "Install it with: pip install revos[tts] or pip install omnivoice"
            ) from e

        manifest = get(model_name, "tts")
        model_id = manifest.model_url

        # Resolve device map for OmniVoice
        if self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"

        device_map = f"{self.device}:0" if self.device == "cuda" else self.device

        logger.info(
            "Loading OmniVoice TTS model %s (device=%s)", model_name, device_map
        )

        # Identify the HF user for gated model tracking
        self.hf_user = _get_hf_user()
        if self.hf_user:
            logger.info(
                "Authenticated as HuggingFace user: %s", self.hf_user["name"]
            )
        else:
            logger.warning(
                "HuggingFace user not identified. "
                "Run 'huggingface-cli login' for access to gated models."
            )

        try:
            self._model = OmniVoice.from_pretrained(
                model_id,
                device_map=device_map,
            )
        except OSError as e:
            if "gated" in str(e).lower() or "authentication" in str(e).lower() or "401" in str(e):
                raise RuntimeError(
                    f"Cannot access model '{model_id}' — it requires HuggingFace authentication.\n"
                    f"Log in with:  huggingface-cli login\n"
                    f"Or set:        export HF_TOKEN=your_token\n"
                    f"Get a token:  https://huggingface.co/settings/tokens"
                ) from e
            raise

        self._sample_rate = manifest.sample_rate
        self._model_id = model_id
        logger.info("OmniVoice TTS model %s loaded", model_name)

        # Track gated model usage
        from revos.usage import track_usage

        track_usage(
            event="model_loaded",
            model_id=model_id,
            model_name=model_name,
            task="tts",
            hf_user=self.hf_user,
            device=self.device,
        )

    def synthesize(
        self,
        text: str,
        output_path: str | None = None,
        *,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio:
        """Synthesize speech using OmniVoice.

        Args:
            text: Text to synthesize.
            output_path: Optional path to save the audio file.
            speed: Speech speed multiplier.
            ref_audio: Optional reference audio for voice cloning.
            ref_text: Optional transcription of reference audio.

        Returns:
            Audio object with synthesized samples.
        """
        kwargs: dict = {"text": text, "speed": speed}

        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text

        result = self._model.generate(**kwargs)

        # OmniVoice returns a list of np.ndarray
        if isinstance(result, list) and len(result) > 0:
            samples = np.array(result[0], dtype=np.float32)
        else:
            samples = np.array(result, dtype=np.float32)

        audio = Audio(samples=samples, sample_rate=self._sample_rate)

        if output_path:
            audio.save(output_path)
            logger.info("Saved synthesized audio to %s", output_path)

        return audio
