"""Model download and caching to ~/.cache/revos/."""

from __future__ import annotations

import logging
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from .manifest import ModelManifest

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "revos"


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Show download progress bar on stderr."""
    downloaded = block_num * block_size
    bar_width = 40

    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        filled = bar_width * pct // 100
        bar = "=" * filled + "-" * (bar_width - filled)
        sys.stderr.write(
            f"\r  [{bar}] {pct:3d}% {mb_down:.1f}/{mb_total:.1f} MB"
        )
        sys.stderr.flush()
        if pct >= 100:
            sys.stderr.write("\n")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stderr.write(f"\r  Downloading: {mb_down:.1f} MB")
        sys.stderr.flush()


def _download(url: str, dest: Path) -> None:
    """Download a file from URL to dest with progress logging."""
    if not url.startswith(("https://", "http://")):
        raise ValueError(
            f"Invalid model URL scheme: {url}. Only https:// and http:// are allowed."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)


def _extract(archive_path: Path, dest_dir: Path) -> None:
    """Extract an archive (tar.bz2, tar.gz, zip) to dest_dir."""
    name = archive_path.name
    if name.endswith(".tar.bz2") or name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir, filter="data")
    elif name.endswith(".zip"):
        _extract_zip_safe(archive_path, dest_dir)
    else:
        shutil.copy2(archive_path, dest_dir / archive_path.name)


def _extract_zip_safe(archive_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive with path-traversal protection."""
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            member_dest = (dest_dir / member.filename).resolve()
            if not str(member_dest).startswith(str(dest_resolved)):
                raise ValueError(
                    f"Unsafe path in zip archive: {member.filename}"
                )
            zf.extract(member, dest_dir)


def _find_model_dir(extract_dir: Path, manifest: ModelManifest) -> Path:
    """Find the actual model directory after extraction.

    Some archives extract to a subdirectory. Look for the files the manifest expects.
    """
    # Check if files exist directly in extract_dir
    expected_files = list(manifest.files.values())
    if expected_files and all((extract_dir / f).exists() for f in expected_files):
        return extract_dir

    # Check one level down (common pattern: archive extracts to subfolder)
    for subdir in extract_dir.iterdir():
        if subdir.is_dir() and all(
            (subdir / f).exists() for f in expected_files
        ):
            return subdir

    # Return extract_dir as fallback
    return extract_dir


def ensure_model(manifest: ModelManifest) -> Path:
    """Ensure model files are downloaded and cached.

    Downloads the model if not already cached. Returns the path to the
    model directory containing the files specified in the manifest.

    Args:
        manifest: Model manifest with download URL and file list.

    Returns:
        Path to the directory containing model files.
    """
    model_dir = CACHE_DIR / manifest.name

    # Check if already downloaded — all expected files exist
    expected_files = list(manifest.files.values())
    if expected_files and model_dir.is_dir():
        if all((model_dir / f).exists() for f in expected_files):
            logger.info("Model %s already cached at %s", manifest.name, model_dir)
            return model_dir

    # Download
    if not manifest.model_url:
        raise ValueError(f"Model {manifest.name} has no download URL")

    model_dir.mkdir(parents=True, exist_ok=True)
    url = manifest.model_url
    archive_name = url.split("/")[-1]
    archive_path = model_dir / archive_name

    _download(url, archive_path)

    # Extract
    if archive_path.name.endswith((".tar.bz2", ".tar.gz", ".tgz", ".zip")):
        extract_dir = model_dir / "_extracted"
        extract_dir.mkdir(exist_ok=True)
        _extract(archive_path, extract_dir)

        # Move files to model_dir
        actual_dir = _find_model_dir(extract_dir, manifest)
        for item in actual_dir.iterdir():
            dest = model_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))

        # Cleanup
        shutil.rmtree(extract_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)

    logger.info("Model %s ready at %s", manifest.name, model_dir)
    return model_dir
