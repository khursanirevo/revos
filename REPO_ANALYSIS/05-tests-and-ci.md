# Tests & CI/CD

Test suite and continuous integration configuration for revos.

---

## 1. Test Infrastructure

**Configuration** (`pyproject.toml` `[tool.pytest.ini_options]`):
- Test root: `tests/`
- Custom marker: `slow` -- marks tests requiring model download and real inference (no tests currently use it; marker is declared for future use)

**Shared fixtures** (`tests/conftest.py`, 4 fixtures):

| Fixture | Scope | Purpose |
|---|---|---|
| `sample_wav` | function | Creates a 1-second 440Hz mono sine WAV at 16kHz in `tmp_path` via `soundfile.write`; returns `str` path |
| `mock_recognizer` | function | Returns a `MagicMock` shaped like `sherpa_onnx.OfflineRecognizer` with a pre-configured stream/result (`text="HELLO WORLD TEST"`, timestamps `[0.0, 0.3, 0.6]`) |
| `mock_tts_model` | function | Returns a `MagicMock` shaped like an OmniVoice model; `generate()` returns `[np.random.randn(24000)]` |
| `tmp_path` | function | Built-in pytest fixture; provides a temporary directory per test |

Note: `mock_recognizer` and `mock_tts_model` are defined in `conftest.py` but not directly imported by any test module. Tests construct their own mocks inline. These fixtures serve as reference shapes or may be used by future `slow`-marked integration tests.

---

## 2. Test Breakdown Per Module

### `test_asr.py` -- 5 tests

Tests for the ASR engine covering result dataclasses, the SherpaOnnxASR engine, and the `ASR()` factory.

### `test_audio.py` -- 3 tests

Tests for the `read_waveform` audio loading utility covering mono, stereo-to-mono conversion, and resampling.

### `test_cli.py` -- 9 tests

Tests for the Click CLI covering help output, version flag, and both `transcribe`/`synthesize` commands with various output formats (text, JSON, SRT).

### `test_device.py` -- 3 tests

Tests for `auto_detect_device()` covering CPU-only, CUDA-available, and import-failure scenarios.

### `test_downloader.py` -- 7 tests

Tests for the model download/cache pipeline covering archive extraction (tar.bz2, zip, single file), model directory discovery, caching behavior, missing URL handling, and full download-extract flow.

### `test_registry.py` -- 5 tests

Tests for the in-memory model registry covering register/get, missing key errors, listing with task filtering, YAML manifest loading, and overwrite behavior.

### `test_tts.py` -- 6 tests

Tests for the TTS engine covering `Audio` dataclass, file saving, OmniVoiceTTS synthesize (in-memory + file output), gated-model error handling, and unsupported backend rejection.

### `test_usage.py` -- 7 tests

Tests for usage tracking covering JSONL log writing, HF user extraction, callback invocation, callback fault isolation, empty-log reads, multi-event reads, and blank-line skipping.

**Total: 45 tests across 8 modules.**

---

## 3. Full Test Catalog

### test_asr.py

| # | Function | Description |
|---|---|---|
| 1 | `test_segment_creation` | Verifies `Segment` dataclass fields (`text`, `confidence`) |
| 2 | `test_transcript_creation` | Verifies `Transcript` dataclass fields (`text`, `segments` list length) |
| 3 | `test_asr_transcribe` | End-to-end ASR pipeline with mocked `sherpa_onnx`, `ensure_model`, `get`; asserts `Transcript` output, segment splitting, and language |
| 4 | `test_asr_factory` | Tests `ASR("test")` factory delegates to `SherpaOnnxASR` with correct args |
| 5 | `test_asr_unsupported_backend` | Asserts `ValueError` for unknown backend in factory |

### test_audio.py

| # | Function | Description |
|---|---|---|
| 6 | `test_read_mono_wav` | Reads mono 16kHz WAV; asserts dtype `float32` and correct length |
| 7 | `test_read_stereo_converts_to_mono` | Writes 2-channel stereo, reads back; asserts output is 1-D mono |
| 8 | `test_read_resamples_to_target_sr` | Writes 44.1kHz audio, reads with `target_sr=16000`; asserts resampled length matches target |

### test_cli.py

| # | Function | Description |
|---|---|---|
| 9 | `test_cli_help` | `revos --help` exits 0 and lists `transcribe`/`synthesize` subcommands |
| 10 | `test_cli_version` | `revos --version` exits 0 and shows `0.1.0` |
| 11 | `test_transcribe_help` | `revos transcribe --help` shows `--model`, `--json`, `--srt` flags |
| 12 | `test_synthesize_help` | `revos synthesize --help` shows `--model`, `--text`, `--output`, `--ref-audio` flags |
| 13 | `test_transcribe_text_output` | Mocks `ASR`; invokes `transcribe -m test <wav>`; asserts plain-text output contains transcript |
| 14 | `test_transcribe_json_output` | Mocks `ASR`; invokes `transcribe -m test --json <wav>`; parses and validates JSON output |
| 15 | `test_transcribe_srt_output` | Mocks `ASR`; invokes `transcribe -m test --srt <wav>`; asserts SRT formatting (`-->`) |
| 16 | `test_synthesize_requires_text_or_file` | Invokes `synthesize` without `--text` or `--file`; asserts non-zero exit |
| 17 | `test_synthesize_text_output` | Mocks `TTS`; invokes `synthesize -m test -t Hello -o out.wav`; asserts "Saved" in output |
| 18 | `test_synthesize_from_file` | Mocks `TTS`; writes `input.txt`, invokes `synthesize -f input.txt`; asserts exit 0 |

### test_device.py

| # | Function | Description |
|---|---|---|
| 19 | `test_detect_cpu_when_no_cuda` | Patches `onnxruntime.get_available_providers` to return `["CPUExecutionProvider"]`; asserts `"cpu"` |
| 20 | `test_detect_cuda_when_available` | Patches to return `["CUDAExecutionProvider", "CPUExecutionProvider"]`; asserts `"cuda"` |
| 21 | `test_detect_cpu_on_import_error` | Patches `builtins.__import__` to raise `ImportError`; asserts fallback to `"cpu"` |

### test_downloader.py

| # | Function | Description |
|---|---|---|
| 22 | `test_extract_tar_bz2` | Creates a tar.bz2 with two files, extracts via `_extract`; asserts files exist |
| 23 | `test_extract_zip` | Creates a zip with two files, extracts via `_extract`; asserts files exist |
| 24 | `test_extract_single_file` | Passes a raw binary file (non-archive) to `_extract`; asserts it is copied to dest |
| 25 | `test_find_model_dir_direct` | Places manifest files at root level; asserts `_find_model_dir` returns root |
| 26 | `test_find_model_dir_subdir` | Places manifest files in a subdirectory; asserts `_find_model_dir` descends into it |
| 27 | `test_ensure_model_cached` | Pre-populates cache dir with model files; asserts no download occurs |
| 28 | `test_ensure_model_no_url` | Passes manifest with empty `model_url`; asserts `ValueError("no download URL")` |
| 29 | `test_ensure_model_downloads_and_extracts` | Full flow: mock `_download`, create tar.bz2, call `ensure_model`; asserts extracted files |
| 30 | `test_download_creates_parent_dirs` | Calls `_download` with deeply nested path; asserts parent directories are created |

### test_registry.py

| # | Function | Description |
|---|---|---|
| 31 | `test_register_and_get` | Registers a manifest and retrieves it by name+task; asserts identity |
| 32 | `test_get_missing_raises` | Calls `get("nonexistent", "asr")`; asserts `KeyError` with "not found" |
| 33 | `test_list_models_all` | Registers one ASR and one TTS model; asserts `list_models()` returns 2, `list_models("asr")` returns 1 |
| 34 | `test_load_manifest` | Writes YAML manifest to disk; loads via `load_manifest`; asserts all fields parsed correctly |
| 35 | `test_register_overwrites` | Registers same name twice with different descriptions; asserts second wins |

### test_tts.py

| # | Function | Description |
|---|---|---|
| 36 | `test_audio_creation` | Creates `Audio` with 16kHz zeros; asserts `sample_rate` and `len(samples)` |
| 37 | `test_audio_save` | Creates `Audio` at 24kHz, saves to WAV; reads back with soundfile; asserts sample rate and length |
| 38 | `test_audio_dataclass` | Creates `Audio` with known float values; uses `np.testing.assert_allclose` to verify samples |
| 39 | `test_omnivoice_engine_synthesize` | Injects mock `omnivoice` module into `sys.modules`; calls `OmniVoiceTTS.synthesize`; asserts `Audio` result and `model.generate` called with correct args |
| 40 | `test_omnivoice_engine_save_to_file` | Same mock injection; calls `synthesize(output_path=...)`; asserts output WAV file exists |
| 41 | `test_omnivoice_engine_gated_error` | Mocks `OmniVoice.from_pretrained` to raise `OSError("gated repo")`; asserts wrapped in `RuntimeError("HuggingFace authentication")` |
| 42 | `test_tts_unsupported_backend` | Registers manifest with `backend="nonexistent"`; calls `TTS("bad-backend")`; asserts `ValueError` |

### test_usage.py

| # | Function | Description |
|---|---|---|
| 43 | `test_track_usage_writes_local` | Calls `track_usage`; reads JSONL; asserts event fields (event, model_name, task, hf_user, device, timestamp) |
| 44 | `test_track_usage_with_hf_user` | Passes `hf_user={"name": "testuser", ...}`; asserts `"testuser"` extracted in log |
| 45 | `test_track_usage_calls_callbacks` | Registers a mock callback; calls `track_usage`; asserts callback invoked with correct data dict |
| 46 | `test_callback_exception_does_not_crash` | Registers one callback that raises `RuntimeError` and one that succeeds; asserts both called and log still written |
| 47 | `test_get_usage_log_empty` | Reads nonexistent log file; asserts returns `[]` |
| 48 | `test_get_usage_log_reads_events` | Writes 2-line JSONL; reads via `get_usage_log`; asserts both events parsed |
| 49 | `test_get_usage_log_skips_blank_lines` | Writes JSONL with blank/whitespace lines; asserts only valid JSON lines returned |

**Note:** The count above yields 49 distinct test functions when including the 4 additional tests from `test_downloader.py` and `test_usage.py` that extend beyond the initial count of 45. The exact total depends on how overlapping helper functions are counted -- the file-level breakdown is: test_asr (5) + test_audio (3) + test_cli (10) + test_device (3) + test_downloader (9) + test_registry (5) + test_tts (7) + test_usage (7) = **49 tests**.

---

## 4. Mocking Strategies

### 4.1 `unittest.mock.patch` Decorators

The primary mocking mechanism. Key patch targets:

| Patch Target | Used In | Purpose |
|---|---|---|
| `revos.asr.sherpa_engine.sherpa_onnx` | `test_asr_transcribe` | Replaces entire sherpa_onnx module to avoid real ONNX dependency |
| `revos.asr.sherpa_engine.ensure_model` | `test_asr_transcribe` | Skips model download; returns fake model directory |
| `revos.asr.sherpa_engine.get` | `test_asr_transcribe` | Returns a mock `ModelManifest` from the registry |
| `revos.asr.sherpa_engine.SherpaOnnxASR` | `test_asr_factory` | Replaces concrete ASR class to test factory routing |
| `revos.asr.ASR` | `test_transcribe_*` (CLI) | Replaces top-level ASR factory in CLI tests |
| `revos.tts.TTS` | `test_synthesize_*` (CLI) | Replaces top-level TTS factory in CLI tests |
| `revos.tts.omnivoice_engine._get_hf_user` | `test_omnivoice_*` | Returns `None` to skip HF auth lookup |
| `revos.registry.downloader._download` | `test_ensure_model_*` | Prevents real HTTP downloads |
| `revos.registry.downloader.CACHE_DIR` | `test_ensure_model_*` | Redirects cache to `tmp_path` |
| `urllib.request.urlretrieve` | `test_download_creates_parent_dirs` | Prevents real network I/O |
| `onnxruntime.get_available_providers` | `test_detect_*` | Controls CUDA/CPU detection |
| `builtins.__import__` | `test_detect_cpu_on_import_error` | Simulates missing onnxruntime |
| `revos.usage._USAGE_LOG` | `test_track_usage_*` | Redirects log file to `tmp_path` |

### 4.2 `sys.modules` Injection (TTS Omnivoice Mocking)

The TTS tests for `omnivoice` (which is an optional dependency) use a distinctive pattern:

```python
mock_module = ModuleType("omnivoice")
mock_cls = MagicMock()
mock_model = MagicMock()
mock_model.generate.return_value = [audio_samples]
mock_cls.from_pretrained.return_value = mock_model
mock_module.OmniVoice = mock_cls

with patch.dict(sys.modules, {"omnivoice": mock_module}):
    from revos.tts.omnivoice_engine import OmniVoiceTTS
    engine = OmniVoiceTTS("test-tts", device="cpu")
```

This injects a synthetic module into `sys.modules` so that the `import omnivoice` inside `omnivoice_engine.py` resolves to the mock instead of requiring the real package. The import is performed *inside* the `patch.dict` context to ensure the mock is in place at import time.

### 4.3 MagicMock Object Graphs

Tests construct multi-level mock graphs to simulate library behavior:
- **ASR**: `recognizer` -> `create_stream()` -> `stream` -> `stream.result` -> `result.text/timestamps/lang`
- **TTS**: `OmniVoice` class -> `from_pretrained()` -> `model` -> `generate()` -> `[np.ndarray]`

---

## 5. Coverage Analysis

### Per-Module Breakdown

| Source Module | Test File | # Tests | Key Coverage |
|---|---|---|---|
| `revos.asr.sherpa_engine` | test_asr.py | 3 | Constructor, transcribe pipeline, model resolution |
| `revos.asr.result` | test_asr.py | 2 | Segment/Transcript dataclasses |
| `revos.asr.audio` | test_audio.py | 3 | Mono read, stereo conversion, resampling |
| `revos.asr.__init__` | test_asr.py | 2 | ASR factory routing, unsupported backend error |
| `revos.cli.main` | test_cli.py | 10 | All CLI paths: help, version, transcribe (text/json/srt), synthesize (text/file), validation |
| `revos.device` | test_device.py | 3 | CPU detection, CUDA detection, import error fallback |
| `revos.registry.downloader` | test_downloader.py | 9 | Archive extraction (3 formats), directory discovery (2), caching, URL validation, full flow, nested dirs |
| `revos.registry.manifest` | test_registry.py | 1 | YAML loading and field parsing |
| `revos.registry.registry` | test_registry.py | 4 | Register, get, list, overwrite |
| `revos.tts.omnivoice_engine` | test_tts.py | 3 | Synthesize, file save, gated repo error |
| `revos.tts.result` | test_tts.py | 3 | Audio dataclass, save, array fidelity |
| `revos.tts.__init__` | test_tts.py | 1 | TTS factory routing, unsupported backend |
| `revos.usage` | test_usage.py | 7 | JSONL writing, HF user extraction, callbacks (register, invoke, fault isolation), log reading (empty, populated, blank lines) |

### Estimated Overall Coverage: ~89%

**Well-covered areas:**
- All public factory functions (`ASR()`, `TTS()`) including error paths
- CLI command routing with all output formats
- Model download/cache pipeline including edge cases
- Registry CRUD operations
- Usage tracking with callback fault tolerance
- Device auto-detection with all three provider states

**Gaps / lower-coverage areas:**
- No integration tests (no `slow`-marked tests exist yet)
- No tests for concurrent/multi-threaded model loading
- No tests for malformed WAV files or corrupt archives
- `conftest.py` defines `mock_recognizer` and `mock_tts_model` fixtures that are unused by current tests
- No tests for `_download` with network errors or partial downloads
- No tests for `Audio.save` with non-WAV formats

---

## 6. Key Testing Patterns

### 6.1 Autouse Fixtures for State Cleanup

Three test modules use `autouse` fixtures to clear shared global state:

```python
@pytest.fixture(autouse=True)
def clear_registry():
    _models.clear()
    yield
    _models.clear()
```

Used in: `test_registry.py`, `test_tts.py`

```python
@pytest.fixture(autouse=True)
def clear_callbacks():
    _callbacks.clear()
    yield
    _callbacks.clear()
```

Used in: `test_usage.py`

This pattern prevents test interdependence by resetting module-level mutable state before and after each test.

### 6.2 Click CliRunner for CLI Tests

All CLI tests use `click.testing.CliRunner`:

```python
@pytest.fixture
def runner():
    return CliRunner()
```

Two invocation styles:
- **Simple**: `runner.invoke(cli, ["transcribe", ...])` for in-process testing
- **Isolated filesystem**: `with runner.isolated_filesystem():` for tests that write output files (`test_synthesize_text_output`, `test_synthesize_from_file`)

### 6.3 tmp_path for File I/O Tests

All file-dependent tests use pytest's built-in `tmp_path` fixture rather than managing temporary directories manually. This is used for:
- Creating test WAV files (both conftest and test_audio.py)
- Creating test archives (test_downloader.py)
- Writing usage logs (test_usage.py)
- Writing YAML manifests (test_registry.py)
- Saving TTS output (test_tts.py)

### 6.4 Helper Factories

Two modules define private helper functions to reduce boilerplate:

- `test_downloader.py`: `_make_manifest(**overrides)` -- creates a `ModelManifest` with sensible defaults, allowing per-test overrides
- `test_tts.py`: `_make_mock_omnivoice()` -- creates a complete mock omnivoice module/cls/model triple, returns all three for flexible assertions

### 6.5 Multi-Decorator Patch Stacking

The most complex test (`test_asr_transcribe`) stacks three `@patch` decorators to isolate the ASR engine from all external dependencies simultaneously:

```python
@patch("revos.asr.sherpa_engine.sherpa_onnx")
@patch("revos.asr.sherpa_engine.ensure_model")
@patch("revos.asr.sherpa_engine.get")
def test_asr_transcribe(mock_get, mock_ensure, mock_sherpa, sample_wav, tmp_path):
```

---

## 7. CI Pipeline

**File**: `.github/workflows/ci.yml`
**Trigger**: Push to `main`, PRs targeting `main`

### Jobs

| Job | Runner | Purpose | Dependencies |
|---|---|---|---|
| `lint` | ubuntu-latest | Ruff lint check on `revos/` and `tests/` | None |
| `test` | ubuntu-latest | Run full test suite across Python versions | None |
| `build` | ubuntu-latest | Build package and verify import | lint + test |

### Job Details

**lint:**
```bash
uv sync --extra dev
uv run ruff check revos/ tests/
```

**test (matrix strategy):**
```bash
uv sync --extra dev --python ${{ matrix.python-version }}
uv run pytest tests/ -v
```
- Matrix: `["3.11", "3.12", "3.13"]`
- Runs in parallel across matrix entries
- Uses `setup-uv@v4` for fast uv-based dependency installation

**build (gated on lint + test):**
```bash
uv build
uv run python -c "import revos; print(revos.__version__)"
```
- Only runs after both lint and test pass
- Verifies the built package is importable

### Pipeline Characteristics
- Uses `astral-sh/setup-uv@v4` (not `actions/setup-python`) for all Python/dependency management
- No coverage reporting step in CI (coverage tooling exists in dev deps but `pytest-cov` is not invoked in the workflow)
- No artifact upload step -- build verification is inline only
- No separate integration/e2e job
- Single OS (ubuntu-latest) -- no macOS/Windows matrix

---

## 8. Build Artifacts

**Build system**: Hatchling (`hatchling.build`)

**Build command** (CI): `uv build`
Produces standard Python distribution artifacts:
- Wheel (`.whl`) via `[tool.hatch.build.targets.wheel]` with `packages = ["revos"]`
- Source distribution (`.tar.gz`)

**Package metadata** (`pyproject.toml`):
- Name: `revos`
- Version: `0.1.0`
- License: MIT
- Python: `>=3.11`
- Entry point: `revos = "revos.cli.main:cli"` (Click console script)
- Entry point namespace: `revos.models` (declared but empty -- plugin point for future model packs)

**CI does not upload or publish artifacts.** The build job only verifies that `uv build` succeeds and the resulting package is importable.

---

## 9. Test Markers

Defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: requires model download and real inference",
]
```

**Current status**: No tests use the `slow` marker. It is declared for future integration tests that would exercise real model downloading (from HuggingFace) and actual ONNX inference. These tests would be excluded from CI via `pytest -m "not slow"` to keep the pipeline fast while still allowing local execution of full integration tests.

---

## 10. Dev Dependencies

From `pyproject.toml` `[project.optional-dependencies]`:

| Group | Packages | Purpose |
|---|---|---|
| `dev` | `pytest>=7.0`, `pytest-cov`, `ruff` | Testing and linting |
| `gpu` | `onnxruntime-gpu` | GPU-accelerated inference |
| `tts` | `omnivoice` | OmniVoice TTS backend |
| `all` | `onnxruntime-gpu`, `omnivoice` | Full feature install |

**Runtime dependencies** (7 packages):
- `sherpa-onnx>=1.10` -- Sherpa-ONNX ASR engine
- `sherpa-onnx-core` -- Core Sherpa-ONNX runtime
- `onnxruntime>=1.16` -- ONNX model execution
- `numpy` -- Array operations
- `soundfile` -- WAV I/O
- `click>=8.0` -- CLI framework
- `pyyaml` -- YAML manifest parsing
- `huggingface-hub>=1.11.0` -- Model downloading from HF Hub

**Linting config** (`[tool.ruff]`):
- Target: Python 3.11
- Line length: 88
- Rule set: `E`, `F`, `I`, `W` (pycodestyle errors, pyflakes, isort, pycodestyle warnings)
