"""Remote model catalog — fetches available models from this repo.

The catalog lives in the revos/models/ directory of this repository.
Team members add YAML manifests to the repo; users discover and
pull them without upgrading the package.

Default catalog: khursanirevo/revos on GitHub
Override with: REVOS_CATALOG_REPO env var or config.yaml
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

import yaml

from .registry.manifest import ModelManifest, load_manifest

logger = logging.getLogger(__name__)

# Default: this repo on GitHub
DEFAULT_CATALOG_REPO = "khursanirevo/revos"

# GitHub API base
_GITHUB_API = "https://api.github.com/repos"

# Local user models directory
_USER_MODELS_DIR = Path.home() / ".config" / "revos" / "models"


def get_catalog_repo() -> str:
    """Get the catalog repository (GitHub owner/repo format).

    Checks in order:
      1. REVOS_CATALOG_REPO environment variable
      2. ~/.config/revos/config.yaml (catalog_repo key)
      3. Default: khursanirevo/revos

    Returns:
        GitHub repository in owner/repo format.
    """
    env_repo = os.environ.get("REVOS_CATALOG_REPO")
    if env_repo:
        return env_repo

    config_path = Path.home() / ".config" / "revos" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            if "catalog_repo" in config:
                return config["catalog_repo"]
        except Exception:
            pass

    return DEFAULT_CATALOG_REPO


def _github_api_get(url: str) -> bytes:
    """Fetch data from GitHub API with proper headers."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "revos-catalog",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def _list_yaml_files(repo: str, path: str) -> list[str]:
    """List YAML files in a GitHub repo directory via API.

    Returns:
        List of file paths relative to repo root.
    """
    url = f"{_GITHUB_API}/{repo}/contents/{path}"
    data = json.loads(_github_api_get(url))

    files: list[str] = []
    for entry in data:
        if entry["type"] == "file" and entry["name"].endswith(
            (".yaml", ".yml")
        ):
            files.append(entry["path"])
        elif entry["type"] == "dir":
            # Recurse into subdirectories (asr/, tts/)
            files.extend(_list_yaml_files(repo, entry["path"]))
    return files


def _download_raw(repo: str, path: str) -> str:
    """Download a file from GitHub as raw text.

    Returns:
        File content as string.
    """
    url = (
        f"https://raw.githubusercontent.com/{repo}/"
        f"HEAD/{path}"
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "revos-catalog"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8")


def list_catalog(task: str | None = None) -> list[ModelManifest]:
    """Fetch available models from the remote catalog.

    Args:
        task: Optional filter by task type ("asr" or "tts").

    Returns:
        List of ModelManifest from the catalog.

    Raises:
        RuntimeError: If catalog cannot be reached.
    """
    repo = get_catalog_repo()

    try:
        yaml_files = _list_yaml_files(repo, "revos/models")
    except Exception as e:
        raise RuntimeError(
            f"Cannot fetch catalog from '{repo}'. "
            f"Error: {e}\n"
            f"Check that the repository exists and is accessible."
        ) from e

    if task:
        yaml_files = [
            f for f in yaml_files if f.startswith(f"revos/models/{task}/")
        ]

    manifests: list[ModelManifest] = []
    for yaml_path in yaml_files:
        try:
            content = _download_raw(repo, yaml_path)
            import tempfile

            # Write to temp file to use load_manifest
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            manifest = load_manifest(tmp_path)
            tmp_path.unlink(missing_ok=True)
            manifests.append(manifest)
        except Exception as e:
            logger.warning(
                "Failed to load catalog entry %s: %s", yaml_path, e
            )

    return manifests


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
    repo = get_catalog_repo()

    # List and search for the model
    try:
        yaml_files = _list_yaml_files(repo, "revos/models")
    except Exception as e:
        raise RuntimeError(
            f"Cannot fetch catalog from '{repo}'. "
            f"Error: {e}"
        ) from e

    target_file = None
    target_manifest = None

    for yf in yaml_files:
        try:
            content = _download_raw(repo, yf)
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            manifest = load_manifest(tmp_path)
            tmp_path.unlink(missing_ok=True)
            if manifest.name == name:
                target_file = yf
                target_manifest = manifest
                break
        except Exception:
            continue

    if target_file is None:
        raise KeyError(
            f"Model '{name}' not found in catalog "
            f"'{repo}'. "
            f"Run 'revos catalog list' to see available models."
        )

    # Install to user models directory
    dest_dir = _USER_MODELS_DIR / target_manifest.task
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(target_file).name

    content = _download_raw(repo, target_file)
    dest_path.write_text(content)

    # Register in the live registry
    manifest = load_manifest(dest_path)
    from .registry import register

    register(manifest)

    logger.info("Pulled model '%s' to %s", name, dest_path)
    return dest_path
