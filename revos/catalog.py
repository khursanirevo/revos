"""Remote model catalog — fetches available models from HuggingFace.

The catalog is a HuggingFace repository containing YAML manifests
in the same format as local manifests. Team members push new models
to the catalog repo; users browse and pull what they need.

Default catalog repo: Revolab/revos-catalog
Override with: REVOS_CATALOG_REPO env var or config.yaml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .registry.manifest import ModelManifest, load_manifest

logger = logging.getLogger(__name__)

# Default catalog repository on HuggingFace
DEFAULT_CATALOG_REPO = "Revolab/revos-catalog"

# Local cache for pulled catalog manifests
_USER_MODELS_DIR = Path.home() / ".config" / "revos" / "models"


def get_catalog_repo() -> str:
    """Get the catalog repository ID.

    Checks in order:
      1. REVOS_CATALOG_REPO environment variable
      2. ~/.config/revos/config.yaml (catalog_repo key)
      3. Default: Revolab/revos-catalog

    Returns:
        HuggingFace repository ID string.
    """
    # 1. Environment variable
    env_repo = os.environ.get("REVOS_CATALOG_REPO")
    if env_repo:
        return env_repo

    # 2. Config file
    config_path = Path.home() / ".config" / "revos" / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            if "catalog_repo" in config:
                return config["catalog_repo"]
        except Exception:
            pass

    # 3. Default
    return DEFAULT_CATALOG_REPO


def list_catalog(task: str | None = None) -> list[ModelManifest]:
    """Fetch available models from the remote catalog.

    Args:
        task: Optional filter by task type ("asr" or "tts").

    Returns:
        List of ModelManifest from the catalog.

    Raises:
        RuntimeError: If HuggingFace hub is not available or
            catalog repo cannot be reached.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required for catalog access. "
            "Install it with: pip install huggingface-hub"
        )

    repo_id = get_catalog_repo()
    api = HfApi()

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception as e:
        raise RuntimeError(
            f"Cannot fetch catalog from '{repo_id}'. "
            f"Error: {e}\n"
            f"Check that the repository exists and you have access."
        ) from e

    # Find all YAML manifest files
    yaml_files = [
        f for f in files if f.endswith((".yaml", ".yml"))
    ]

    if task:
        yaml_files = [f for f in yaml_files if f.startswith(f"{task}/")]

    manifests: list[ModelManifest] = []
    for yaml_path in yaml_files:
        try:
            local_path = _download_manifest(repo_id, yaml_path)
            manifest = load_manifest(local_path)
            # Clean up temp download
            local_path.unlink(missing_ok=True)
            manifests.append(manifest)
        except Exception as e:
            logger.warning("Failed to load catalog entry %s: %s", yaml_path, e)

    return manifests


def _download_manifest(repo_id: str, path: str) -> Path:
    """Download a single manifest file from the catalog repo."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=repo_id,
        filename=path,
        repo_type="model",
    )
    return Path(local)


def pull_model(name: str) -> Path:
    """Pull a model manifest from the catalog and install it locally.

    Downloads the YAML manifest to ~/.config/revos/models/{task}/
    and registers it.

    Args:
        name: Model name to pull (e.g. "revovoice").

    Returns:
        Path to the installed manifest file.

    Raises:
        KeyError: If the model is not found in the catalog.
        RuntimeError: If download fails.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface-hub is required for catalog access. "
            "Install it with: pip install huggingface-hub"
        )

    repo_id = get_catalog_repo()
    api = HfApi()

    # Find the manifest file for this model
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    target_file = None
    for yf in yaml_files:
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=yf, repo_type="model"
            )
            manifest = load_manifest(Path(local))
            Path(local).unlink(missing_ok=True)
            if manifest.name == name:
                target_file = yf
                target_manifest = manifest
                break
        except Exception:
            continue

    if target_file is None:
        raise KeyError(
            f"Model '{name}' not found in catalog "
            f"'{repo_id}'. "
            f"Run 'revos catalog' to see available models."
        )

    # Download to user models directory
    dest_dir = _USER_MODELS_DIR / target_manifest.task
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(target_file).name

    local = hf_hub_download(
        repo_id=repo_id, filename=target_file, repo_type="model"
    )

    # Copy to user config
    import shutil

    shutil.copy2(local, dest_path)
    Path(local).unlink(missing_ok=True)

    # Register in the live registry
    manifest = load_manifest(dest_path)
    from .registry import register

    register(manifest)

    logger.info("Pulled model '%s' to %s", name, dest_path)
    return dest_path
