# RevoS Architecture Overview

Generated: 2026-04-24 | Version analyzed: 0.1.0

---

## 1. Project Purpose and Scope

RevoS is a unified Python library for speech AI, providing both Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) through open models. It wraps inference backends (sherpa-onnx for ASR, OmniVoice for TTS) behind a clean, model-agnostic API. The core design goal: adding a new model requires zero changes to core code -- only a YAML manifest.

**Scope boundaries:**
- Inference only (no training, no fine-tuning).
- Two task types currently: ASR and TTS. Architecture explicitly supports new task types (diarization, VAD, enhancement) via the same pattern.
- Models are either bundled as public downloads or accessed via private HuggingFace repositories.

---

## 2. Package Structure

```
revos/
  __init__.py                    # Version only: __version__ = "0.1.0"
  device.py                      # CUDA/CPU auto-detection via onnxruntime
  usage.py                       # Usage tracking for gated models (callbacks + local log)

  asr/
    __init__.py                  # ASR() factory function, dispatches on manifest.backend
    base.py                      # BaseASR ABC with abstract transcribe()
    result.py                    # Segment, Transcript dataclasses
    audio.py                     # read_waveform() -- mono conversion + resampling
    sherpa_engine.py             # SherpaOnnxASR -- concrete engine using sherpa-onnx

  tts/
    __init__.py                  # TTS() factory function, dispatches on manifest.backend
    base.py                      # BaseTTS ABC with abstract synthesize()
    result.py                    # Audio dataclass with save()
    omnivoice_engine.py          # OmniVoiceTTS -- concrete engine using omnivoice + torch

  registry/
    __init__.py                  # Public API: get, register, list_models, ensure_model, load_manifest, ModelManifest
    manifest.py                  # ModelManifest dataclass + load_manifest() YAML loader
    registry.py                  # In-memory model registry, auto-loads manifests on import
    downloader.py                # Model download/extract/cache to ~/.cache/revos/

  cli/
    __init__.py                  # Empty package marker
    main.py                      # Click CLI: "revos transcribe" and "revos synthesize"

  models/
    asr/
      zipformer_v2.yaml          # Zipformer small English transducer via sherpa-onnx
    tts/
      omnivoice.yaml             # OmniVoice multilingual diffusion TTS (HF private)
```

### Module Roles

| Module | Responsibility | Key Exports |
|--------|---------------|-------------|
| `revos/__init__` | Version identity | `__version__` |
| `revos.device` | Hardware detection | `auto_detect_device()` |
| `revos.usage` | Telemetry for gated models | `track_usage()`, `register_callback()`, `get_usage_log()` |
| `revos.asr` | ASR factory + engine | `ASR()`, `BaseASR`, `Transcript`, `Segment` |
| `revos.asr.sherpa_engine` | sherpa-onnx ONNX inference | `SherpaOnnxASR` |
| `revos.asr.audio` | Audio I/O preprocessing | `read_waveform()` |
| `revos.tts` | TTS factory + engine | `TTS()`, `BaseTTS`, `Audio` |
| `revos.tts.omnivoice_engine` | OmniVoice diffusion inference | `OmniVoiceTTS` |
| `revos.registry` | Model discovery + download | `get()`, `register()`, `list_models()`, `ensure_model()` |
| `revos.registry.manifest` | YAML manifest parsing | `ModelManifest`, `load_manifest()` |
| `revos.registry.downloader` | Archive download + extraction | `ensure_model()` |
| `revos.cli.main` | Command-line interface | `cli` (Click group) |

---

## 3. Build System and Dependencies

**Build backend:** `hatchling` (Hatch build system).

### Required Dependencies (installed with `pip install revos`)

| Package | Constraint | Used By |
|---------|-----------|---------|
| `sherpa-onnx` | `>=1.10` | `revos/asr/sherpa_engine.py` |
| `sherpa-onnx-core` | unversioned | companion to sherpa-onnx |
| `onnxruntime` | `>=1.16` | `revos/device.py` (provider detection), ASR inference |
| `numpy` | unversioned | `revos/asr/audio.py`, `revos/tts/result.py` |
| `soundfile` | unversioned | `revos/asr/audio.py`, `revos/tts/result.py` |
| `click` | `>=8.0` | `revos/cli/main.py` |
| `pyyaml` | unversioned | `revos/registry/manifest.py` |
| `huggingface-hub` | `>=1.11.0` | `revos/tts/omnivoice_engine.py` (HF auth), model download |

### Optional Dependencies (extras)

| Extra | Packages | Purpose |
|-------|----------|---------|
| `gpu` | `onnxruntime-gpu` | CUDA-accelerated ONNX inference |
| `tts` | `omnivoice` | OmniVoice TTS backend (also requires PyTorch) |
| `all` | `onnxruntime-gpu`, `omnivoice` | Full stack |
| `dev` | `pytest>=7.0`, `pytest-cov`, `ruff` | Development tooling |

**Notable:** PyTorch (`torch`) is not declared as a dependency anywhere in `pyproject.toml`, yet `revos/tts/omnivoice_engine.py` imports it at runtime for CUDA detection. The `omnivoice` package presumably pulls it in transitively, but this is an implicit dependency.

---

## 4. Entry Points

### CLI

Defined in `pyproject.toml` line 45:

```
revos = "revos.cli.main:cli"
```

Two subcommands:

- **`revos transcribe`** -- ASR transcription
  - Options: `--model/-m` (required), `--json` (JSON output), `--srt` (SRT subtitle output)
  - Argument: `audio_path` (must exist on disk)
  - Implementation: `revos/cli/main.py:26-57`

- **`revos synthesize`** -- TTS synthesis
  - Options: `--model/-m` (required), `--text/-t`, `--file/-f`, `--output/-o` (required), `--speed` (default 1.0), `--ref-audio`, `--ref-text`
  - Requires exactly one of `--text` or `--file`
  - Implementation: `revos/cli/main.py:60-108`

### Programmatic API

```python
from revos.asr import ASR
asr = ASR('zipformer-v2', device='auto')
result = asr.transcribe('audio.wav')
# result.text, result.segments, result.language

from revos.tts import TTS
tts = TTS('omnivoice')
audio = tts.synthesize('Hello', output_path='out.wav', ref_audio='ref.wav', ref_text='...')
# audio.samples, audio.sample_rate, audio.save()
```

### Plugin Entry Point

`pyproject.toml` declares an empty `revos.models` entry point group (line 47-48), reserved for future plugin-based model registration.

---

## 5. Python Version Support

- **Minimum:** Python 3.11 (declared in `requires-python = ">=3.11"`)
- **Classifiers target:** 3.11, 3.12, 3.13
- **Ruff target:** `py311`

The codebase uses Python 3.10+ syntax features:
- `from __future__ import annotations` in most modules (deferring type evaluation)
- `dict | None` union syntax (3.10+) in `revos/tts/omnivoice_engine.py:57`, `revos/registry/registry.py:51`
- `list[...]`, `tuple[...]` generic syntax without `typing` imports

---

## 6. Device Detection Logic

**File:** `revos/device.py`

`auto_detect_device()` performs a single detection strategy:

1. Attempts `import onnxruntime`
2. Calls `onnxruntime.get_available_providers()`
3. If `"CUDAExecutionProvider"` is in the returned list, returns `"cuda"`
4. Falls back to `"cpu"` (also falls back if onnxruntime is not installed)

**Usage sites:**
- `SherpaOnnxASR.__init__` (`revos/asr/sherpa_engine.py:26-27`) -- called when `device="auto"`
- `OmniVoiceTTS.__init__` (`revos/tts/omnivoice_engine.py:59-67`) -- uses its own parallel detection via `torch.cuda.is_available()` rather than calling `auto_detect_device()`, since OmniVoice runs on PyTorch, not ONNX

**Design note:** The two engines use different CUDA detection methods. ASR checks ONNX providers; TTS checks PyTorch CUDA availability. This means it is possible for the two engines to disagree about device availability if one runtime is installed without the corresponding GPU support package.

---

## 7. Usage Tracking System

**File:** `revos/usage.py`

### Event Schema

Each usage event is a dict with these fields:

| Field | Source | Example |
|-------|--------|---------|
| `event` | Passed as arg | `"model_loaded"`, `"model_synthesized"` |
| `model_id` | HF model ID or local path | `"Revolab/omnivoice"`, `"/home/user/.cache/revos/zipformer-v2"` |
| `model_name` | RevoS model name | `"omnivoice"`, `"zipformer-v2"` |
| `task` | Task type | `"asr"`, `"tts"` |
| `hf_user` | HF username or None | `"jdoe"` |
| `device` | Compute device | `"cpu"`, `"cuda"` |
| `timestamp` | ISO 8601 UTC | `"2026-04-24T10:30:00+00:00"` |
| Additional | `**extra` kwargs | Any |

### Callback System

- `register_callback(callback: Callable[[dict], None])` -- appends to module-level `_callbacks` list.
- On each `track_usage()` call, all registered callbacks are invoked with the usage dict.
- Callbacks that raise exceptions are caught and logged at WARNING level (line 89-91). Other callbacks still execute.

### Local Logging

- `_log_to_local()` appends JSONL entries to `~/.cache/revos/usage.jsonl`.
- This happens unconditionally on every `track_usage()` call, before callbacks.
- `get_usage_log()` reads the full JSONL file and returns a list of dicts (oldest-first order).

### Invocation Points

1. **ASR engine** (`revos/asr/sherpa_engine.py:65-75`): Tracks `model_loaded` only if `manifest.hf_private` is true or `model_url` starts with `"http"`.
2. **TTS engine** (`revos/tts/omnivoice_engine.py:109-118`): Always tracks `model_loaded` (OmniVoice is a gated HF model).
3. Neither engine currently tracks `model_synthesized` events, though the schema supports it.

### HF User Detection

`OmniVoiceTTS` has a private helper `_get_hf_user()` (`revos/tts/omnivoice_engine.py:17-33`) that calls `HfApi().whoami()` to resolve the current HF user. The ASR engine passes `hf_user=None`.

---

## 8. Key Design Patterns

### Pattern 1: Factory Functions with Backend Dispatch

Both `ASR()` and `TTS()` are module-level factory functions (not classes). They:

1. Call `registry.get(model_name, task)` to resolve the manifest.
2. Inspect `manifest.backend` to select the concrete engine class.
3. Lazy-import the engine module (avoiding import of unused backends).
4. Return a `BaseASR` / `BaseTTS` instance.

```python
# revos/asr/__init__.py:19-44
def ASR(model_name: str, device: str = "auto") -> BaseASR:
    manifest = get(model_name, "asr")
    if manifest.backend == "sherpa-onnx":
        from .sherpa_engine import SherpaOnnxASR
        return SherpaOnnxASR(model_name, device)
    raise ValueError(...)
```

Adding a new backend requires adding one `elif` branch in the factory and creating the engine file.

### Pattern 2: YAML Manifest Registry (Data-Driven Model Discovery)

The model registry (`revos/registry/registry.py`) is an in-memory dict keyed by `(task, name)` tuples. On import:

1. `_load_builtin_manifests()` scans `revos/models/` recursively for `.yaml` and `.yml` files.
2. `_load_user_manifests()` scans `~/.config/revos/models/` recursively for the same.

User manifests override bundled ones if they share the same `(task, name)` key (last-write-wins). The registry is a module-level singleton, populated once at import time (lines 99-101).

### Pattern 3: Lazy Imports for Optional Dependencies

Heavy or optional dependencies are imported inside function bodies and constructor methods, not at module level:

- `sherpa_onnx` imported in `SherpaOnnxASR.__init__`
- `omnivoice` imported in `OmniVoiceTTS.__init__`
- `torch` imported in `OmniVoiceTTS.__init__` (for CUDA check)
- `onnxruntime` imported in `auto_detect_device()`
- `huggingface_hub.HfApi` imported in `_get_hf_user()`

This ensures the library can be imported without optional backends installed. Missing backends produce actionable `ImportError` messages with install instructions (e.g., `revos/tts/omnivoice_engine.py:48-52`).

### Pattern 4: Abstract Base Classes with Uniform Interface

- `BaseASR` (`revos/asr/base.py`): ABC with `transcribe(audio_path) -> Transcript`
- `BaseTTS` (`revos/tts/base.py`): ABC with `synthesize(text, output_path, *, speed, ref_audio, ref_text) -> Audio`

Both store `model_name` and `device` as instance attributes. Both use keyword-only parameters for optional args. Return types are frozen dataclasses (`Segment`, `Transcript`, `Audio`).

### Pattern 5: Download-and-Cache with Idempotent Guard

`ensure_model()` (`revos/registry/downloader.py:75-125`) implements a check-before-download pattern:

1. Check if all expected files exist in `~/.cache/revos/{model_name}/`.
2. If yes, return immediately (cached).
3. If no, download archive from `model_url`, extract (tar.bz2, tar.gz, zip), locate model files (handles subdirectory extraction), move to cache dir, cleanup temp files.

The guard is file-presence-based, not version-aware. There is no cache invalidation or update mechanism.

### Pattern 6: Self-Save in Result Objects

The `Audio` dataclass (`revos/tts/result.py`) includes a `save(path)` method that writes samples via `soundfile.write()`. This follows the principle that result objects are self-contained and can serialize themselves.

---

## 9. Distribution Strategy

### PyPI

The package is configured for PyPI publication via hatchling. The package name is `revos`.

### Installation Modes

```bash
pip install revos              # Core: ASR only (sherpa-onnx + onnxruntime)
pip install revos[tts]         # + OmniVoice TTS (requires PyTorch implicitly)
pip install revos[gpu]         # + onnxruntime-gpu for CUDA ASR
pip install revos[all]         # Full stack
pip install revos[dev]         # Dev tooling (pytest, ruff)
```

### Model Distribution

Models are NOT packaged with the library. They are:

1. **Public models** (e.g., zipformer-v2): Downloaded on first use from GitHub releases URLs via `urllib.request`. Cached at `~/.cache/revos/{model_name}/`.
2. **Private/gated models** (e.g., omnivoice): Accessed via HuggingFace Hub (`OmniVoice.from_pretrained(model_id, device_map=...)`). Requires HF authentication (`huggingface-cli login` or `HF_TOKEN` env var).

User-defined models are supported by placing YAML manifests in `~/.config/revos/models/` -- no code changes required.

### Git Install

Standard editable install supported:

```bash
pip install -e .
```

---

## 10. Notable Architecture Decisions

### 10.1 Factory Functions over Class Constructors

The public API uses `ASR('model-name')` and `TTS('model-name')` as factory functions rather than requiring users to know the concrete engine class. This encapsulates the dispatch logic and keeps the user-facing surface minimal (2 entry points for the entire library).

### 10.2 Manifest-Driven Model Registry

Models are defined declaratively in YAML rather than in Python code. This enables:
- Non-developers to add models (ops, researchers).
- Runtime discovery of user-installed models without code changes.
- Clean separation of model metadata from inference logic.

### 10.3 Split Device Detection Across Engines

ASR uses onnxruntime provider detection; TTS uses PyTorch CUDA detection. This is architecturally intentional (each engine checks its own runtime's GPU support) but could lead to inconsistent device reporting if only one runtime has CUDA support installed.

### 10.4 Implicit PyTorch Dependency

PyTorch is never declared as a direct dependency in `pyproject.toml`. The `omnivoice` extra package presumably brings it in, but the relationship is implicit. This means `pip install revos[tts]` may fail if `omnivoice` does not declare a `torch` dependency.

### 10.5 No Streaming or Online Recognition

Both ASR and TTS engines operate in offline/batch mode only. `SherpaOnnxASR` uses `OfflineRecognizer` (not streaming). There is no API surface for streaming audio input or real-time transcription.

### 10.6 Resampling via Linear Interpolation

`read_waveform()` in `revos/asr/audio.py:30-36` uses `np.interp` (linear interpolation) for sample rate conversion. This is simple and dependency-free but lower quality than proper resampling libraries (e.g., `librosa.resample`, `scipy.signal.resample`). May introduce artifacts for non-trivial rate conversions.

### 10.7 Callback-Based Usage Tracking

The usage tracking system uses a register-and-notify pattern rather than a fixed telemetry endpoint. This allows consumers to hook into usage events without modifying the library. The local JSONL log is always active, providing a fallback audit trail.

### 10.8 Empty Plugin Entry Point

`pyproject.toml` declares `[project.entry-points."revos.models"]` with no entries. This reserves the namespace for future plugin-based model registration via setuptools entry points, but the current implementation loads manifests from filesystem paths instead.

### 10.9 No Version-Pinned Model Downloads

The downloader checks for file presence but not version or integrity (no checksums, no ETag matching). If a model URL changes or files are corrupted, the user must manually clear `~/.cache/revos/` to force re-download.

---

## Appendix: Bundled Model Manifests

### zipformer-v2 (ASR)

| Field | Value |
|-------|-------|
| Name | `zipformer-v2` |
| Task | `asr` |
| Backend | `sherpa-onnx` |
| Model type | `transducer` |
| URL | `https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2` |
| Sample rate | 16000 |
| Language | `en` |
| Files | `encoder-epoch-99-avg-1.onnx`, `decoder-epoch-99-avg-1.onnx`, `joiner-epoch-99-avg-1.onnx`, `tokens.txt` |
| HF private | false |

### omnivoice (TTS)

| Field | Value |
|-------|-------|
| Name | `omnivoice` |
| Task | `tts` |
| Backend | `omnivoice` |
| Model type | `diffusion` |
| URL | `Revolab/omnivoice` (HuggingFace repo ID) |
| Sample rate | 24000 |
| Language | `multilingual` (600+) |
| Files | `{}` (no explicit files; downloaded via HF Hub) |
| HF private | true |
