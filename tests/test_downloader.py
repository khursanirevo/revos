"""Tests for model downloader."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from revos.registry.downloader import (
    _download,
    _extract,
    _find_model_dir,
    ensure_model,
)
from revos.registry.manifest import ModelManifest


def _make_manifest(**overrides):
    defaults = dict(
        name="test-model",
        task="asr",
        backend="sherpa-onnx",
        model_type="transducer",
        model_url="http://example.com/model.tar.bz2",
        sample_rate=16000,
        language="en",
        description="Test",
        files={"encoder": "encoder.onnx", "tokens": "tokens.txt"},
    )
    defaults.update(overrides)
    return ModelManifest(**defaults)


def test_extract_tar_bz2(tmp_path: Path):
    """Test extracting a tar.bz2 archive."""
    # Create a tar.bz2 archive with test files
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "encoder.onnx").write_text("dummy encoder")
    (src_dir / "tokens.txt").write_text("hello world")

    archive_path = tmp_path / "model.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as tf:
        for f in src_dir.iterdir():
            tf.add(str(f), arcname=f.name)

    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()

    _extract(archive_path, dest_dir)

    assert (dest_dir / "encoder.onnx").exists()
    assert (dest_dir / "tokens.txt").exists()


def test_extract_zip(tmp_path: Path):
    """Test extracting a zip archive."""
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("encoder.onnx", "dummy encoder")
        zf.writestr("tokens.txt", "hello world")

    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()

    _extract(archive_path, dest_dir)

    assert (dest_dir / "encoder.onnx").exists()
    assert (dest_dir / "tokens.txt").exists()


def test_extract_zip_rejects_path_traversal(tmp_path: Path):
    """Test that zip entries with path traversal are rejected."""
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../../../etc/passwd", "evil content")

    dest_dir = tmp_path / "extracted"
    dest_dir.mkdir()

    with pytest.raises(ValueError, match="Unsafe path"):
        _extract(archive_path, dest_dir)


def test_extract_single_file(tmp_path: Path):
    """Test extracting a single file (non-archive)."""
    single_file = tmp_path / "model.bin"
    single_file.write_text("raw model data")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()

    _extract(single_file, dest_dir)

    assert (dest_dir / "model.bin").exists()


def test_find_model_dir_direct(tmp_path: Path):
    """Test finding model files directly in extract dir."""
    manifest = _make_manifest()
    (tmp_path / "encoder.onnx").write_text("data")
    (tmp_path / "tokens.txt").write_text("data")

    result = _find_model_dir(tmp_path, manifest)
    assert result == tmp_path


def test_find_model_dir_subdir(tmp_path: Path):
    """Test finding model files in a subdirectory."""
    manifest = _make_manifest()
    subdir = tmp_path / "model-subdir"
    subdir.mkdir()
    (subdir / "encoder.onnx").write_text("data")
    (subdir / "tokens.txt").write_text("data")

    result = _find_model_dir(tmp_path, manifest)
    assert result == subdir


@patch("revos.registry.downloader._download")
def test_ensure_model_cached(mock_download, tmp_path: Path):
    """Test that cached models are not re-downloaded."""
    manifest = _make_manifest()

    # Pre-populate cache
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()
    (model_dir / "encoder.onnx").write_text("cached")
    (model_dir / "tokens.txt").write_text("cached")

    with patch("revos.registry.downloader.CACHE_DIR", tmp_path):
        result = ensure_model(manifest)

    assert result == model_dir
    mock_download.assert_not_called()


@patch("revos.registry.downloader._download")
def test_ensure_model_no_url(mock_download, tmp_path: Path):
    """Test that missing URL raises ValueError."""
    manifest = _make_manifest(model_url="")

    with patch("revos.registry.downloader.CACHE_DIR", tmp_path):
        with pytest.raises(ValueError, match="no download URL"):
            ensure_model(manifest)

    mock_download.assert_not_called()


@patch("revos.registry.downloader._download")
def test_ensure_model_downloads_and_extracts(mock_download, tmp_path: Path):
    """Test full download + extract flow."""
    manifest = _make_manifest()

    # Create the archive that _download would produce
    model_dir = tmp_path / "test-model"
    model_dir.mkdir()

    archive_path = model_dir / "model.tar.bz2"
    src = tmp_path / "archive_src"
    src.mkdir()
    (src / "encoder.onnx").write_text("downloaded encoder")
    (src / "tokens.txt").write_text("hello")
    with tarfile.open(str(archive_path), "w:bz2") as tf:
        for f in src.iterdir():
            tf.add(str(f), arcname=f.name)

    # _download creates the archive file
    def fake_download(url, dest):
        # Archive is already there for this test
        pass

    mock_download.side_effect = fake_download

    with patch("revos.registry.downloader.CACHE_DIR", tmp_path):
        result = ensure_model(manifest)

    assert (result / "encoder.onnx").exists()
    assert (result / "tokens.txt").exists()


def test_download_creates_parent_dirs(tmp_path: Path):
    """Test that _download creates parent directories."""
    dest = tmp_path / "deep" / "nested" / "file.tar.bz2"
    with patch("urllib.request.urlretrieve"):
        _download("http://example.com/file.tar.bz2", dest)
    assert dest.parent.is_dir()


def test_download_rejects_file_scheme(tmp_path: Path):
    """Test that file:// URLs are rejected."""
    dest = tmp_path / "file.tar.bz2"
    with pytest.raises(ValueError, match="Invalid model URL scheme"):
        _download("file:///etc/passwd", dest)


def test_download_rejects_ftp_scheme(tmp_path: Path):
    """Test that ftp:// URLs are rejected."""
    dest = tmp_path / "file.tar.bz2"
    with pytest.raises(ValueError, match="Invalid model URL scheme"):
        _download("ftp://example.com/model.tar.bz2", dest)
