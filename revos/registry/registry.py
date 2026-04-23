"""Model registry — auto-loads manifests from bundled and user directories."""

from __future__ import annotations

import logging
from pathlib import Path

from .manifest import ModelManifest, load_manifest

logger = logging.getLogger(__name__)

# Internal storage: (task, name) -> ModelManifest
_models: dict[tuple[str, str], ModelManifest] = {}


def register(manifest: ModelManifest) -> None:
    """Register a model manifest.

    Args:
        manifest: The ModelManifest to register.
    """
    key = (manifest.task, manifest.name)
    _models[key] = manifest
    logger.debug("Registered model: %s/%s", manifest.task, manifest.name)


def get(name: str, task: str) -> ModelManifest:
    """Look up a registered model manifest.

    Args:
        name: Model name (e.g. "zipformer-v2").
        task: Task type ("asr" or "tts").

    Returns:
        The matching ModelManifest.

    Raises:
        KeyError: If no model is registered with that name/task.
    """
    key = (task, name)
    if key not in _models:
        available = list_models(task)
        names = [m.name for m in available]
        if not names:
            raise KeyError(
                f"Model '{name}' (task={task}) not found. "
                f"No {task} models are registered. "
                f"Add a manifest in ~/.config/revos/models/{task}/"
            )
        raise KeyError(
            f"Model '{name}' (task={task}) not found. "
            f"Available {task} models: {names}"
        )
    return _models[key]


def list_models(task: str | None = None) -> list[ModelManifest]:
    """List registered model manifests.

    Args:
        task: Optional filter by task type ("asr" or "tts").

    Returns:
        List of matching ModelManifest instances.
    """
    if task is None:
        return list(_models.values())
    return [m for m in _models.values() if m.task == task]


def _load_manifests_from_dir(directory: Path) -> None:
    """Load all YAML manifests from a directory (recursive)."""
    if not directory.is_dir():
        return

    for yaml_file in directory.rglob("*.yaml"):
        try:
            manifest = load_manifest(yaml_file)
            register(manifest)
            logger.info("Loaded manifest: %s from %s", manifest.name, yaml_file)
        except Exception as e:
            logger.warning("Failed to load manifest %s: %s", yaml_file, e)

    for yml_file in directory.rglob("*.yml"):
        try:
            manifest = load_manifest(yml_file)
            register(manifest)
            logger.info("Loaded manifest: %s from %s", manifest.name, yml_file)
        except Exception as e:
            logger.warning("Failed to load manifest %s: %s", yml_file, e)


def _load_builtin_manifests() -> None:
    """Load manifests bundled with the package (revos/models/)."""
    models_dir = Path(__file__).parent.parent / "models"
    _load_manifests_from_dir(models_dir)


def _load_user_manifests() -> None:
    """Load user manifests from ~/.config/revos/models/."""
    user_dir = Path.home() / ".config" / "revos" / "models"
    _load_manifests_from_dir(user_dir)


# Auto-load on import
_load_builtin_manifests()
_load_user_manifests()
