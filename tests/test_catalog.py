"""Tests for remote model catalog."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from revos.catalog import (
    DEFAULT_CATALOG_REPO,
    _download_raw,
    _list_yaml_files,
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


def test_list_catalog_github_error():
    """list_catalog raises RuntimeError when GitHub is unreachable."""
    with patch(
        "revos.catalog._list_yaml_files",
        side_effect=Exception("network error"),
    ):
        with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "bad/repo"}):
            with pytest.raises(RuntimeError, match="Cannot fetch catalog"):
                list_catalog()


@patch("revos.catalog._download_raw")
@patch("revos.catalog._list_yaml_files")
def test_list_catalog_fetches_manifests(mock_list, mock_download):
    """list_catalog fetches and parses manifests from GitHub."""
    mock_list.return_value = [
        "revos/models/tts/revovoice.yaml",
        "revos/models/asr/zipformer_v2.yaml",
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

    mock_download.side_effect = [manifest_tts, manifest_asr]

    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
        results = list_catalog()

    assert len(results) == 2
    names = [m.name for m in results]
    assert "revovoice" in names
    assert "zipformer-v2" in names


@patch("revos.catalog._download_raw")
@patch("revos.catalog._list_yaml_files")
def test_list_catalog_filters_by_task(mock_list, mock_download):
    """list_catalog filters results by task type."""
    mock_list.return_value = [
        "revos/models/tts/revovoice.yaml",
    ]

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

    mock_download.return_value = manifest

    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
        results = list_catalog(task="tts")

    assert len(results) == 1
    assert results[0].task == "tts"


@patch("revos.catalog._download_raw")
@patch("revos.catalog._list_yaml_files")
def test_pull_model_installs_locally(mock_list, mock_download, tmp_path):
    """pull_model downloads and installs manifest to user dir."""
    mock_list.return_value = ["revos/models/tts/revovoice.yaml"]

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

    # pull_model downloads twice: once to scan, once to install
    mock_download.side_effect = [manifest_content, manifest_content]

    models_dir = tmp_path / "models"
    with patch("revos.catalog._USER_MODELS_DIR", models_dir):
        pull_model("revovoice")

    assert (models_dir / "tts" / "revovoice.yaml").exists()


@patch("revos.catalog._list_yaml_files")
def test_pull_model_not_found(mock_list):
    """pull_model raises KeyError when model is not in catalog."""
    mock_list.return_value = []

    with patch.dict(os.environ, {"REVOS_CATALOG_REPO": "TestOrg/catalog"}):
        with pytest.raises(KeyError, match="not found in catalog"):
            pull_model("nonexistent")


@patch("revos.catalog.urllib.request.urlopen")
def test_list_yaml_files(mock_urlopen):
    """_list_yaml_files parses GitHub API response."""
    api_response = [
        {
            "type": "dir",
            "name": "tts",
            "path": "revos/models/tts",
        },
    ]
    dir_response = [
        {
            "type": "file",
            "name": "revovoice.yaml",
            "path": "revos/models/tts/revovoice.yaml",
        },
    ]

    mock_resp = MagicMock()
    mock_resp.read.side_effect = [
        json.dumps(api_response).encode(),
        json.dumps(dir_response).encode(),
    ]
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    files = _list_yaml_files("test/repo", "revos/models")
    assert files == ["revos/models/tts/revovoice.yaml"]


@patch("revos.catalog.urllib.request.urlopen")
def test_download_raw(mock_urlopen):
    """_download_raw fetches file content from GitHub."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"name: test\n"
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    content = _download_raw("test/repo", "revos/models/test.yaml")
    assert content == "name: test\n"
