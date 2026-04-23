# Registry and CLI Module Analysis

## 1. ModelManifest Dataclass

**File:** `revos/registry/manifest.py:11-28`

```python
@dataclass
class ModelManifest:
    name: str
    task: str               # "asr" or "tts"
    backend: str            # e.g. "sherpa-onnx"
    model_type: str         # e.g. "transducer", "vits", "kokoro"
    model_url: str
    sample_rate: int
    language: str
    description: str
    files: dict[str, str] = field(default_factory=dict)
    hf_private: bool = False
```

All fields are plain attributes (no `__post_init__`, no validation). The `files` dict maps logical names (e.g. `"encoder"`, `"tokens"`) to relative filenames (e.g. `"encoder.onnx"`). `hf_private` defaults `False` and is used to gate access to private Hugging Face repositories.

---

## 2. YAML Manifest Loading

**File:** `revos/registry/manifest.py:30-53`

`load_manifest(path: Path) -> ModelManifest` reads a YAML file via `yaml.safe_load` and constructs a `ModelManifest`. Required key: `name`. All others fall back to defaults:

| YAML key | Default |
|---|---|
| `model_type` | `""` |
| `model_url` | `""` |
| `sample_rate` | `16000` |
| `language` | `""` |
| `description` | `""` |
| `files` | `{}` |
| `hf_private` | `False` |

Missing `name` raises `KeyError` at runtime. No schema validation beyond what the dataclass constructor enforces.

---

## 3. Registry Storage Pattern

**File:** `revos/registry/registry.py:13`

```python
_models: dict[tuple[str, str], ModelManifest] = {}
```

Keys are `(task, name)` tuples. This means the same model name can coexist under different tasks (e.g., `"base"` for both `"asr"` and `"tts"`), but duplicate `(task, name)` pairs silently overwrite (confirmed by `test_register_overwrites`).

### Public API

| Function | Signature | Behavior |
|---|---|---|
| `register` | `(manifest: ModelManifest) -> None` | Inserts or overwrites by `(task, name)` key. Logs at DEBUG. |
| `get` | `(name: str, task: str) -> ModelManifest` | Returns manifest or raises `KeyError` with available model names listed. |
| `list_models` | `(task: str | None = None) -> list[ModelManifest]` | Returns all manifests, optionally filtered by task. |

---

## 4. Auto-Loading on Import

**File:** `revos/registry/registry.py:99-101`

```python
_load_builtin_manifests()
_load_user_manifests()
```

These execute at module import time. Both call `_load_manifests_from_dir`, which recursively globs for `*.yaml` and `*.yml` files.

| Source | Directory | Notes |
|---|---|---|
| Builtin | `<package_root>/models/` | Ships with the installed package. |
| User | `~/.config/revos/models/` | User-installed custom models. |

Loading errors are non-fatal: each file failure is logged at WARNING level and skipped. This means a broken user manifest does not prevent the application from starting.

---

## 5. Download System

**File:** `revos/registry/downloader.py`

### 5.1 Download (`_download`)

```python
def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
```

Uses `urllib.request.urlretrieve` with a progress hook. No retry logic, no checksum verification, no authentication headers.

### 5.2 Progress Hook (`_progress_hook`)

```python
def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
```

Logs at INFO level:
- When `total_size` is known: logs at every 20% milestone with MB downloaded / MB total.
- When `total_size` is unknown: logs every 5 MB.

### 5.3 Extraction (`_extract`)

```python
def _extract(archive_path: Path, dest_dir: Path) -> None:
```

Supports three archive formats based on filename extension:

| Extension | Handler |
|---|---|
| `.tar.bz2`, `.tar.gz`, `.tgz` | `tarfile.open` + `extractall` with `filter="data"` |
| `.zip` | `zipfile.ZipFile` + `extractall` |
| Anything else | `shutil.copy2` (treats as a raw single file) |

### 5.4 Model Directory Discovery (`_find_model_dir`)

After extraction, some archives produce a subdirectory. `_find_model_dir` resolves the actual location:

1. Check if all `manifest.files` values exist directly in `extract_dir`.
2. Check one level of subdirectories for the same condition.
3. Fallback to `extract_dir` itself.

### 5.5 Top-Level Orchestrator (`ensure_model`)

```python
def ensure_model(manifest: ModelManifest) -> Path:
```

Flow:
1. Compute `model_dir = CACHE_DIR / manifest.name`.
2. If all `manifest.files` values exist under `model_dir`, return immediately (cached).
3. If `manifest.model_url` is empty, raise `ValueError`.
4. Create `model_dir`, derive archive filename from URL's last path segment.
5. Call `_download`.
6. If archive is a known format: extract to `model_dir/_extracted/`, move files up to `model_dir/`, delete `_extracted/` and the archive.
7. Return `model_dir`.

---

## 6. Cache Management

**File:** `revos/registry/downloader.py:16`

```python
CACHE_DIR = Path.home() / ".cache" / "revos"
```

Each model is stored at `~/.cache/revos/<model_name>/`. Cache validity is determined by checking that all files listed in `manifest.files` exist. There is no expiration, no size limit, and no explicit cache-clearing API. Manual cleanup requires deleting the directory.

---

## 7. Security Measures

**File:** `revos/registry/downloader.py:46`

```python
tf.extractall(dest_dir, filter="data")
```

Tarball extraction uses `filter="data"` (Python 3.12+), which blocks path traversal attacks (e.g., files with `../` components or absolute paths). This is the recommended mitigation for CVE-2007-4559.

No equivalent filter is applied to `zipfile.ZipFile.extractall` (which has a different, less severe traversal risk). No checksum/hash verification is performed on downloaded archives.

---

## 8. Click CLI Structure

**File:** `revos/cli/main.py`

### Top-Level Group

```python
@click.group()
@click.version_option()
def cli() -> None:
    """RevoS -- A unified library for speech AI (ASR & TTS)."""
```

### 8.1 `transcribe` Command

```python
@cli.command()
@click.option("--model", "-m", required=True, help="ASR model name (e.g. zipformer-v2)")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--srt", "as_srt", is_flag=True, help="Output as SRT subtitles")
def transcribe(model, audio_path, as_json, as_srt):
```

- Lazily imports `revos.asr.ASR` (deferred import to avoid loading ONNX runtime at CLI startup).
- Constructs `ASR(model)`, calls `asr.transcribe(audio_path)`.
- Output format dispatch:
  - **Plain** (default): prints `result.text`.
  - **JSON** (`--json`): prints `{text, segments: [{start, end, text, confidence}], language}` with `indent=2`, `ensure_ascii=False`.
  - **SRT** (`--srt`): numbered segments with `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamps.

### 8.2 `synthesize` Command

```python
@cli.command()
@click.option("--model", "-m", required=True, help="TTS model name (e.g. omnivoice)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option("--file", "-f", type=click.Path(exists=True), help="Text file to synthesize")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output audio path")
@click.option("--speed", default=1.0, help="Speech speed (default: 1.0)")
@click.option("--ref-audio", type=click.Path(exists=True), help="Reference audio for voice cloning")
@click.option("--ref-text", help="Transcription of reference audio")
def synthesize(model, text, file, output, speed, ref_audio, ref_text):
```

- Requires at least one of `--text` or `--file`; otherwise raises `click.UsageError`.
- When `--file` is provided, reads and strips file content as text.
- Calls `tts.synthesize(text, output, speed=speed, ref_audio=ref_audio, ref_text=ref_text)`.
- On success, prints: `"Saved {n_samples} samples ({duration:.1f}s) to {output}"`.

### 8.3 SRT Timestamp Formatter

```python
def _format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

Standard SRT format: `HH:MM:SS,mmm`.

---

## 9. Output Formats

### 9.1 Plain Text

Emits `result.text` directly via `click.echo`. Single line.

### 9.2 JSON

```json
{
  "text": "HELLO WORLD",
  "segments": [
    {"start": 0.0, "end": 0.5, "text": "HELLO", "confidence": 0.9},
    {"start": 0.5, "end": 1.0, "text": "WORLD", "confidence": 0.8}
  ],
  "language": "en"
}
```

Two-space indented, Unicode-preserving (`ensure_ascii=False`).

### 9.3 SRT Subtitles

```
1
00:00:00,000 --> 00:00:00,500
HELLO

2
00:00:00,500 --> 00:00:01,000
WORLD
```

Blank line separates entries. Sequence numbers start at 1.

---

## 10. Entry Point Configuration

**File:** `pyproject.toml:44-45`

```toml
[project.scripts]
revos = "revos.cli.main:cli"
```

Installs a `revos` console script that invokes the Click group at `revos.cli.main:cli`. The CLI module also supports `python -m revos.cli.main` execution via the `if __name__ == "__main__": cli()` guard.

**File:** `pyproject.toml:47-48`

```toml
[project.entry-points."revos.models"]
```

An empty entry point group is declared (no entries yet), reserved for plugin-discovered model manifests.

---

## Test Coverage Summary

| Test file | Tests | Key patterns |
|---|---|---|
| `test_registry.py` | 5 | Clear/restore `_models` dict via autouse fixture; verifies register, get, list, overwrite, YAML loading |
| `test_cli.py` | 7 | Uses `CliRunner` + `unittest.mock.patch` on `ASR`/`TTS`; tests help output, all three output formats, synthesize from text and file, missing-input error |
| `test_downloader.py` | 7 | Creates real tar.bz2/zip archives; patches `CACHE_DIR` for isolation; verifies extraction, subdirectory discovery, caching, missing URL error, parent dir creation |

Test fixture `sample_wav` (`tests/conftest.py`) generates a 1-second 440 Hz sine wave WAV at 16 kHz using numpy + soundfile.
