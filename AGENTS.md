# AGENTS.md — Contributing Guide for AI Agents & Humans

This document describes how to extend the RevoS library with new models, backends, and tasks. Follow it precisely when adding support for a new speech AI model.

---

## Architecture Overview

RevoS has a layered architecture:

```
CLI (click) → Factory Functions (ASR/TTS) → Base Classes (BaseASR/BaseTTS) → Concrete Engines
                                    ↕
                              Model Registry (YAML manifests)
                                    ↕
                              Model Downloader (~/.cache/revos/)
                              Remote Catalog (GitHub repo)
```

**Key principle:** Adding a new model should require ZERO changes to core code if the backend is already supported. Only a new YAML manifest is needed.

**Adding a new backend** (e.g., HuggingFace, ONNX Direct, Triton) requires a new engine file + registering it in the factory.

---

## Task 1: Add a New Model (Same Backend)

If the new model uses an already-supported backend (sherpa-onnx for ASR, revovoice for TTS), you only need a YAML manifest.

### Steps

1. **Create a YAML manifest** in `revos/models/{task}/` or `~/.config/revos/models/{task}/`:

```yaml
name: my-new-model          # Unique identifier, used as ASR('my-new-model')
task: asr                    # "asr" or "tts"
backend: sherpa-onnx         # Must match an existing backend
model_type: transducer       # sherpa-onnx: transducer, whisper, paraformer, etc.
model_url: "https://..."     # Download URL (tar.bz2, tar.gz, zip)
# revision: "a1b2c3d"        # Pin to HF commit hash or tag
sample_rate: 16000           # Model's expected sample rate
language: en                 # Supported language(s)
description: "Human-readable description"
hf_private: false            # True for gated HF models
files:                       # Filenames inside the archive
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
```

2. **Test it:**
```python
from revos.asr import ASR
asr = ASR('my-new-model')
result = asr.transcribe('test.wav')
print(result.text)
```

3. **Add tests** in `tests/test_registry.py` for manifest loading.

### Rules
- The `name` must be unique within the same `task`.
- The `files` dict keys must match what the engine expects (see engine source).
- The `model_url` must be a direct download link.
- The archive must extract to contain the files listed in `files`.

---

## Task 2: Add a New Backend for ASR

To support a new inference backend for ASR (e.g., HuggingFace Transformers, faster-whisper, TensorRT).

### Steps

1. **Research the backend's Python API** before writing any code:
   - How to instantiate the model?
   - How to run inference?
   - What does the result object look like (text, timestamps, confidence)?
   - What dependencies does it need?

2. **Create engine file** at `revos/asr/{backend}_engine.py`:

```python
"""Backend name backend for ASR."""

from __future__ import annotations
import logging
from .base import BaseASR
from .result import Segment, Transcript

logger = logging.getLogger(__name__)

class MyBackendASR(BaseASR):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        # Load manifest, download model, initialize backend
        from revos.registry import get, ensure_model
        manifest = get(model_name, "asr")
        # ... initialize your backend

    def transcribe(self, audio_path: str) -> Transcript:
        # Run inference, parse results into Transcript
        text = "..."
        segments = [Segment(start=0.0, end=1.0, text="hello", confidence=0.9)]
        return Transcript(text=text, segments=segments, language="en")
```

3. **Register in factory** — edit `revos/asr/__init__.py`:

```python
# In the ASR() function, add a new branch:
if manifest.backend == "my-backend":
    from .my_backend_engine import MyBackendASR
    return MyBackendASR(model_name, device)
```

4. **Add dependency** to `pyproject.toml` (preferably as an optional extra):

```toml
[project.optional-dependencies]
my-backend = ["my-backend-package"]
```

5. **Add tests** in `tests/test_asr.py` with mocked backend.

6. **Add a YAML manifest** for at least one model using this backend.

### Rules
- Must inherit from `BaseASR` and implement `transcribe()`.
- Must return a `Transcript` object with `text`, `segments`, and `language`.
- Must use `revos.registry` for manifest lookup and `ensure_model()` for downloads.
- Must handle device selection ("auto", "cpu", "cuda").
- Must lazy-import backend dependencies (so the library works without them installed).
- Must raise `ImportError` with install instructions if backend is missing.

---

## Task 3: Add a New Backend for TTS

Same pattern as ASR, but for TTS.

### Steps

1. **Research the backend's Python API.**

2. **Create engine file** at `revos/tts/{backend}_engine.py`:

```python
from .base import BaseTTS
from .result import Audio

class MyBackendTTS(BaseTTS):
    def __init__(self, model_name: str, device: str = "auto") -> None:
        super().__init__(model_name, device)
        # ... initialize backend

    def synthesize(self, text: str, output_path: str | None = None, *,
                   speed: float = 1.0, ref_audio: str | None = None,
                   ref_text: str | None = None) -> Audio:
        # Run inference, return Audio(samples=np.ndarray, sample_rate=int)
        audio = Audio(samples=samples, sample_rate=sr)
        if output_path:
            audio.save(output_path)
        return audio
```

3. **Register in factory** — edit `revos/tts/__init__.py`.

4. **Add dependency, tests, manifest** — same as ASR.

### Rules
- Must inherit from `BaseTTS` and implement `synthesize()`.
- Must return an `Audio` object with `samples` (np.ndarray float32) and `sample_rate` (int).
- Must support `output_path` parameter for direct file saving.
- Must lazy-import backend dependencies.

### Long Text Support

`BaseTTS` provides `synthesize_long()` which automatically splits
long text into sentences, synthesizes each chunk, and concatenates
the audio with short silence gaps.

```python
audio = tts.synthesize_long(
    "Very long text with many sentences...",
    output_path="output.wav",
    max_chars=500,           # max chars per chunk
    silence_duration=0.1,    # silence between chunks
)
print(f"Duration: {audio.duration:.1f}s")
```

Text splitting handles English and CJK punctuation. Falls back to
comma/word boundaries for very long sentences. `Audio.concatenate()`
joins segments with configurable silence gaps.

---

## Task 4: Add a New Task Type

To add a completely new task (e.g., speaker diarization, voice activity detection, speech enhancement).

### Steps

1. **Create task package** at `revos/{task}/`:
   - `__init__.py` — factory function
   - `base.py` — abstract base class
   - `result.py` — result dataclasses
   - `{backend}_engine.py` — concrete engine

2. **Add result dataclasses** specific to the task (e.g., `DiarizationResult` with speakers and segments).

3. **Create the abstract base class** following the pattern of `BaseASR`/`BaseTTS`.

4. **Implement at least one backend engine.**

5. **Add CLI commands** in `revos/cli/main.py`.

6. **Add model manifests** in `revos/models/{task}/`.

7. **Add tests.**

### Rules
- Each task gets its own package under `revos/`.
- Each task defines its own result types.
- The factory function pattern must match ASR/TTS (name + device lookup).

---

## Key File Locations

| Purpose | Location |
|---------|----------|
| ASR engine (sherpa-onnx) | `revos/asr/sherpa_engine.py` |
| TTS engine (RevoVoice) | `revos/tts/revovoice_engine.py` |
| ASR base class | `revos/asr/base.py` |
| TTS base class | `revos/tts/base.py` (includes `synthesize_long`) |
| ASR result types | `revos/asr/result.py` (Segment, Transcript) |
| TTS result types | `revos/tts/result.py` (Audio, Audio.concatenate) |
| Model registry | `revos/registry/registry.py` |
| Manifest loader | `revos/registry/manifest.py` (ModelManifest dataclass) |
| Model downloader | `revos/registry/downloader.py` |
| Remote catalog | `revos/catalog.py` |
| Device detection | `revos/device.py` |
| CLI entry point | `revos/cli/main.py` |
| Bundled manifests | `revos/models/{asr,tts}/*.yaml` |
| User manifests | `~/.config/revos/models/**/*.yaml` |
| Model cache | `~/.cache/revos/{model_name}/` |

---

## Remote Catalog

Users can discover and install models from this repo without upgrading
the package. The catalog fetches YAML manifests from the GitHub repo's
`revos/models/` directory via the GitHub API.

### How It Works

1. Team member adds a YAML manifest to `revos/models/{task}/` and pushes
2. User runs `revos catalog list` to see available models
3. User runs `revos catalog pull <name>` to install locally

### CLI Commands

```bash
revos catalog list              # List models from GitHub
revos catalog list -t tts       # Filter by task
revos catalog pull revovoice    # Install a model locally
```

### Adding Models to the Catalog

Just add a YAML file to `revos/models/{task}/` in this repo. No
separate catalog repo or service needed. Users get it automatically
on the next `revos catalog list`.

The catalog source is configurable via `REVOS_CATALOG_REPO` env var
or `~/.config/revos/config.yaml` (`catalog_repo` key).

---

## Testing Checklist

When adding any new model or backend, verify:

- [ ] `uv run pytest tests/ -v` — all tests pass
- [ ] Factory function returns correct engine type
- [ ] Manifest loads and registers correctly
- [ ] CLI works: `uv run revos transcribe -m {model} test.wav`
- [ ] JSON output valid: `uv run revos transcribe -m {model} --json test.wav`
- [ ] SRT output valid: `uv run revos transcribe -m {model} --srt test.wav`
- [ ] GPU fallback to CPU works (set `device="cpu"`)
- [ ] `from revos.asr import ASR` / `from revos.tts import TTS` works without optional deps
- [ ] ImportErrors are helpful when optional backend is missing

---

## Current Backends

### ASR Backends

| Backend | Engine File | Dependencies | Notes |
|---------|------------|-------------|-------|
| sherpa-onnx | `revos/asr/sherpa_engine.py` | sherpa-onnx, onnxruntime | Zipformer transducer models via ONNX |

### TTS Backends

| Backend | Engine File | Dependencies | Notes |
|---------|------------|-------------|-------|
| revovoice | `revos/tts/revovoice_engine.py` | omnivoice, torch | Diffusion-based zero-shot TTS, 600+ languages |

### Manifest `backend` Values

The `backend` field in YAML manifests must exactly match one of the registered backends above. The factory functions dispatch on this value.
