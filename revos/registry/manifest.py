"""Model manifest dataclass and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelManifest:
    """Describes a model's metadata and file layout.

    Loaded from YAML manifests in revos/models/ or ~/.config/revos/models/.
    """

    name: str
    task: str  # "asr" or "tts"
    backend: str  # e.g. "sherpa-onnx"
    model_type: str  # e.g. "transducer", "vits", "kokoro"
    model_url: str
    sample_rate: int
    language: str
    description: str
    files: dict[str, str] = field(default_factory=dict)
    hf_private: bool = False
    revision: str = ""


def load_manifest(path: Path) -> ModelManifest:
    """Load a model manifest from a YAML file.

    Args:
        path: Path to the YAML manifest file.

    Returns:
        Populated ModelManifest instance.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return ModelManifest(
        name=data["name"],
        task=data["task"],
        backend=data["backend"],
        model_type=data.get("model_type", ""),
        model_url=data.get("model_url", ""),
        sample_rate=data.get("sample_rate", 16000),
        language=data.get("language", ""),
        description=data.get("description", ""),
        files=data.get("files", {}),
        hf_private=data.get("hf_private", False),
        revision=data.get("revision", ""),
    )
