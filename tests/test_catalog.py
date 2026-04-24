"""Tests for remote model catalog."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from revos.catalog import (
    DEFAULT_CATALOG_REPO,
    get_catalog_repo,
    list_catalog,
    pull_model,
)
from revos.registry.registry import _models


@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    yield
    _models.clear()


def test_get_catalog_repo_default():
    """Default catalog repo is used when no config is set."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("revos.catalog.Path") as mock_path:
            mock_config = MagicMock()
            mock_config.exists.return_value = False
            mock_path.home.return_value.__truediv__ = MagicMock(
                return_value=MagicMock(
                    __truediv__=MagicMock(return_value=mock_config)
                )
            )
            assert get_catalog_repo() == DEFAULT_CATALOG_REPO


def test_get_catalog_repo_from_env():
    """Environment variable overrides default catalog repo."""
    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "myorg/models"}):
        assert get_catalog_repo() == "myorg/models"


def test_get_catalog_repo_env_beats_config():
    """Env var takes precedence over config file."""
    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "env-wins"}):
        assert get_catalog_repo() == "env-wins"


def test_list_catalog_missing_hf_hub():
    """list_catalog raises when huggingface-hub is not installed."""
    with patch.dict("sys.modules", {"huggingface_hub": None}):
        with pytest.raises(RuntimeError, match="huggingface-hub is required"):
            list_catalog()


@patch("huggingface_hub.hf_hub_download")
@patch("huggingface_hub.HfApi")
def test_list_catalog_fetches_manifests(mock_api_cls, mock_download):
    """list_catalog fetches and parses manifests from HF."""
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.return_value = [
        "tts/revovoice.yaml",
        "asr/zipformer_v2.yaml",
    ]

    manifest_tts = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: Revolab/revovoice\n"
        "sample_rate: 24000\n"
        "language: multilingual\n"
        "description: Test\n"
        "files: {}\n"
    )
    manifest_asr = (
        "name: zipformer-v2\n"
        "task: asr\n"
        "backend: sherpa-onnx\n"
        "model_type: transducer\n"
        "model_url: https://example.com/model.tar.bz2\n"
        "sample_rate: 16000\n"
        "language: en\n"
        "description: Test\n"
        "files: {}\n"
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tts_file = Path(tmpdir) / "revovoice.yaml"
        tts_file.write_text(manifest_tts)
        asr_file = Path(tmpdir) / "zipformer_v2.yaml"
        asr_file.write_text(manifest_asr)

        mock_download.side_effect = [str(tts_file), str(asr_file)]

        with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
            results = list_catalog()

    assert len(results) == 2
    names = [m.name for m in results]
    assert "revovoice" in names
    assert "zipformer-v2" in names


@patch("huggingface_hub.hf_hub_download")
@patch("huggingface_hub.HfApi")
def test_list_catalog_filters_by_task(mock_api_cls, mock_download):
    """list_catalog filters results by task type."""
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.return_value = [
        "tts/revovoice.yaml",
    ]

    import tempfile

    manifest = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: test\n"
        "sample_rate: 24000\n"
        "language: en\n"
        "description: Test\n"
        "files: {}\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "revovoice.yaml"
        f.write_text(manifest)
        mock_download.return_value = str(f)

        with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
            results = list_catalog(task="tts")

    assert len(results) == 1
    assert results[0].task == "tts"


@patch("huggingface_hub.HfApi")
def test_list_catalog_repo_not_found(mock_api_cls):
    """list_catalog raises RuntimeError when repo is unreachable."""
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.side_effect = Exception("repo not found")

    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "bad/repo"}):
        with pytest.raises(RuntimeError, match="Cannot fetch catalog"):
            list_catalog()


@patch("huggingface_hub.hf_hub_download")
@patch("huggingface_hub.HfApi")
def test_pull_model_installs_locally(
    mock_api_cls, mock_download, tmp_path
):
    """pull_model downloads and installs manifest to user dir."""
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.return_value = ["tts/revovoice.yaml"]

    manifest_content = (
        "name: revovoice\n"
        "task: tts\n"
        "backend: revovoice\n"
        "model_type: diffusion\n"
        "model_url: Revolab/revovoice\n"
        "sample_rate: 24000\n"
        "language: multilingual\n"
        "description: Test\n"
        "files: {}\n"
    )

    tmp_manifest = tmp_path / "revovoice.yaml"
    tmp_manifest.write_text(manifest_content)
    # pull_model downloads twice: once to scan, once to install.
    # The scan deletes the temp file, so we need two copies.
    tmp_manifest2 = tmp_path / "revovoice2.yaml"
    tmp_manifest2.write_text(manifest_content)
    mock_download.side_effect = [str(tmp_manifest), str(tmp_manifest2)]

    models_dir = tmp_path / "models"
    with patch("revos.catalog._USER_MODELS_DIR", models_dir):
        pull_model("revovoice")

    assert (models_dir / "tts" / "revovoice.yaml").exists()


@patch("huggingface_hub.HfApi")
def test_pull_model_not_found(mock_api_cls):
    """pull_model raises KeyError when model is not in catalog."""
    mock_api = MagicMock()
    mock_api_cls.return_value = mock_api
    mock_api.list_repo_files.return_value = []

    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
        with pytest.raises(KeyError, match="not found in catalog"):
            pull_model("nonexistent")
