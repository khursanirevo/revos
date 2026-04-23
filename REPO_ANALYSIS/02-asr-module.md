# ASR Module Analysis

## 1. Module Architecture

The ASR module follows a **base class -> factory -> engine** pattern:

```
revos/asr/
  __init__.py       # Factory function ASR()
  base.py           # Abstract base class BaseASR
  result.py         # Dataclasses: Segment, Transcript
  audio.py          # Waveform loading: read_waveform()
  sherpa_engine.py  # Concrete engine: SherpaOnnxASR
```

**Class hierarchy:**

```
BaseASR (ABC)                    # revos/asr/base.py:10
  |
  +-- SherpaOnnxASR              # revos/asr/sherpa_engine.py:19
```

**Public API surface** (`__all__`):

```python
# revos/asr/__init__.py:47
["ASR", "BaseASR", "Transcript", "Segment"]
```

**Instantiation flow:**

1. User calls `ASR("zipformer-v2", device="auto")`
2. Factory calls `get(model_name, "asr")` to look up the manifest in the registry
3. Dispatches on `manifest.backend` -- only `"sherpa-onnx"` is supported
4. Lazy-imports `SherpaOnnxASR` and constructs it with `(model_name, device)`
5. Returns a `BaseASR`-typed instance

```python
# revos/asr/__init__.py:19
def ASR(model_name: str, device: str = "auto") -> BaseASR:
```

Unsupported backends raise `ValueError` with message `"Unsupported ASR backend: {backend} for model '{model_name}'"`.

---

## 2. SherpaOnnxASR Engine Implementation

**File:** `revos/asr/sherpa_engine.py:19`

### Constructor

```python
def __init__(self, model_name: str, device: str = "auto") -> None:
```

Initialization sequence:

1. **Call `super().__init__(model_name, device)`** -- stores `model_name` and `device` on the base class.
2. **Device resolution** -- if `device == "auto"`, replaces it with `auto_detect_device()` (returns `"cuda"` or `"cpu"`).
3. **Provider mapping** -- maps resolved device to sherpa-onnx provider string: `"cuda"` -> `"cuda"`, anything else -> `"cpu"`.
4. **Manifest lookup** -- calls `get(model_name, "asr")` to retrieve the `ModelManifest`.
5. **Model download** -- calls `ensure_model(manifest)` to download/cache model files, returns `Path` to model directory.
6. **Build file paths** -- reads `manifest.files` dict to resolve four ONNX file paths:
   - `encoder` -> `encoder-epoch-99-avg-1.onnx`
   - `decoder` -> `decoder-epoch-99-avg-1.onnx`
   - `joiner` -> `joiner-epoch-99-avg-1.onnx`
   - `tokens` -> `tokens.txt`
7. **Create recognizer** -- calls `sherpa_onnx.OfflineRecognizer.from_transducer(...)` with the resolved paths.
8. **Gated model tracking** -- if `manifest.hf_private` is truthy OR `manifest.model_url` starts with `"http"`, calls `track_usage(event="model_loaded", ...)`. This means all models with HTTP URLs trigger tracking, not just HuggingFace gated ones.

### Instance state

| Attribute | Type | Source |
|---|---|---|
| `self.model_name` | `str` | From base class |
| `self.device` | `str` | Resolved from `"auto"` |
| `self._recognizer` | `sherpa_onnx.OfflineRecognizer` | Built from transducer files |
| `self._sample_rate` | `int` | `manifest.sample_rate` |
| `self._model_id` | `str` | `manifest.model_url` |

---

## 3. End-to-End `transcribe()` Flow

**Signature:**

```python
# revos/asr/sherpa_engine.py:77
def transcribe(self, audio_path: str) -> Transcript:
```

**Step-by-step:**

1. **Load audio** -- `read_waveform(audio_path, target_sr=self._sample_rate)` returns `(samples: np.ndarray[float32], sr: int)`.
2. **Create stream** -- `self._recognizer.create_stream()` produces a new sherpa-onnx offline stream.
3. **Feed waveform** -- `stream.accept_waveform(sr, samples)` loads samples into the stream.
4. **Decode** -- `self._recognizer.decode_stream(stream)` runs inference synchronously.
5. **Extract result** -- `stream.result` yields an object with `.text`, `.timestamps`, and `.lang`.
6. **Build segments** -- from word-level timestamps (see Section 4).
7. **Return Transcript** -- `(text=text, segments=segments, language=result.lang or "")`.

---

## 4. Segment Building Logic (Word-Level Timestamps)

**File:** `revos/asr/sherpa_engine.py:96-116`

The segment builder operates on two parallel sources:
- `timestamps`: a list of floats from `stream.result.timestamps` (one per word, representing word *start* times)
- `words`: `text.split()` -- the whitespace-split transcript

**Algorithm:**

```python
if timestamps and words:
    for i, word in enumerate(words):
        start = timestamps[i] if i < len(timestamps) else 0.0
        end = timestamps[i + 1] if i + 1 < len(timestamps) else start + 0.1
        segments.append(Segment(start=start, end=end, text=word, confidence=0.0))
elif text:
    # Fallback: single segment covering the entire text, no timing
    segments.append(Segment(start=0.0, end=0.0, text=text, confidence=0.0))
```

**Key behaviors:**
- Each word becomes its own `Segment` (word-level granularity, not sentence-level).
- Word *end* time is the next word's *start* time.
- For the last word, end time defaults to `start + 0.1` (100ms fallback).
- If timestamps are missing entirely but text exists, a single segment with `start=0.0, end=0.0` covers the whole text.
- `confidence` is always `0.0` -- sherpa-onnx does not return per-word confidence in the offline API.
- If `timestamps` and `words` lengths diverge, the shorter list controls the loop (index-bounded).

---

## 5. sherpa-onnx API Usage

### OfflineRecognizer Construction

```python
# revos/asr/sherpa_engine.py:51-59
sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder=encoder,    # str path to encoder ONNX
    decoder=decoder,    # str path to decoder ONNX
    joiner=joiner,      # str path to joiner ONNX
    tokens=tokens,      # str path to tokens.txt
    num_threads=2,      # hardcoded thread count
    sample_rate=16000,  # from manifest
    provider="cpu",     # or "cuda"
)
```

This is the `from_transducer` classmethod -- a factory for building a transducer-based offline (non-streaming) recognizer. The model architecture is encoder-decoder-joint (RNN-T / transducer).

### Stream API (Offline Inference)

The engine uses sherpa-onnx's **offline** stream API (not the online/streaming API):

```python
stream = recognizer.create_stream()       # Allocate a new inference stream
stream.accept_waveform(sample_rate, samples)  # Feed float32 PCM data
recognizer.decode_stream(stream)          # Run full inference (blocking)
result = stream.result                    # Access OfflineRecognitionResult
```

`result` exposes:
- `.text` -- full transcription string
- `.timestamps` -- list of float word-start times in seconds
- `.lang` -- detected language string (may be empty)

---

## 6. Device Handling

### Auto-Detection

**File:** `revos/device.py:8`

```python
def auto_detect_device() -> str:
    """Returns "cuda" if CUDAExecutionProvider is available, "cpu" otherwise."""
```

Mechanism:
- Imports `onnxruntime` and checks `onnxruntime.get_available_providers()`.
- If `"CUDAExecutionProvider"` is present, returns `"cuda"`.
- If onnxruntime is not installed, logs a warning and returns `"cpu"`.
- Otherwise returns `"cpu"`.

### Provider Selection in SherpaOnnxASR

```python
# revos/asr/sherpa_engine.py:29-31
provider = "cpu"
if self.device == "cuda":
    provider = "cuda"
```

The provider string is passed directly to `OfflineRecognizer.from_transducer()`. There is no handling for other providers (e.g., TensorRT, CoreML). Only `"cpu"` and `"cuda"` are mapped.

### Accepted device values

| User input | Resolution |
|---|---|
| `"auto"` | Calls `auto_detect_device()` |
| `"cpu"` | Passed through |
| `"cuda"` | Passed through |
| Any other string | Falls through to `"cpu"` provider (no validation) |

---

## 7. Model Manifest Structure for ASR

### Dataclass

**File:** `revos/registry/manifest.py:12`

```python
@dataclass
class ModelManifest:
    name: str              # e.g. "zipformer-v2"
    task: str              # "asr" or "tts"
    backend: str           # e.g. "sherpa-onnx"
    model_type: str        # e.g. "transducer"
    model_url: str         # Download URL
    sample_rate: int       # e.g. 16000
    language: str          # e.g. "en"
    description: str       # Human-readable description
    files: dict[str, str]  # {"encoder": "file.onnx", ...}
    hf_private: bool       # Whether model is a gated HF model
```

### Manifest Loading

Manifests are loaded from two locations at import time:

1. **Built-in:** `revos/models/` -- shipped with the package, loaded via `_load_builtin_manifests()`.
2. **User:** `~/.config/revos/models/` -- user-provided overrides, loaded via `_load_user_manifests()`.

Both scan recursively for `*.yaml` and `*.yml` files. The registry auto-loads on import of `revos.registry`.

### Registry Lookup

```python
# revos/registry/registry.py:27
def get(name: str, task: str) -> ModelManifest:
```

Uses `(task, name)` tuple as the key. Raises `KeyError` with available model names if not found.

### Model Download and Cache

```python
# revos/registry/downloader.py:75
def ensure_model(manifest: ModelManifest) -> Path:
```

- Cache location: `~/.cache/revos/{model_name}/`
- Checks if all expected files already exist before downloading.
- Supports `.tar.bz2`, `.tar.gz`, `.tgz`, `.zip` archives.
- Handles archives that extract to a subdirectory (searches one level down).
- Downloads via `urllib.request.urlretrieve` with progress logging.

---

## 8. Zipformer v2 Model Details

**File:** `revos/models/asr/zipformer_v2.yaml`

```yaml
name: zipformer-v2
task: asr
backend: sherpa-onnx
model_type: transducer
model_url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-small-en-2023-06-26.tar.bz2"
sample_rate: 16000
language: en
description: "Zipformer small English transducer ASR model via sherpa-onnx"
files:
  encoder: "encoder-epoch-99-avg-1.onnx"
  decoder: "decoder-epoch-99-avg-1.onnx"
  joiner: "joiner-epoch-99-avg-1.onnx"
  tokens: "tokens.txt"
```

| Field | Value |
|---|---|
| Model name | `zipformer-v2` |
| Architecture | Zipformer (small) transducer |
| Backend | sherpa-onnx |
| Language | English (`en`) |
| Sample rate | 16000 Hz |
| Source | k2-fsa/sherpa-onnx GitHub releases |
| ONNX files | encoder, decoder, joiner (epoch 99, avg 1) + tokens.txt |
| Archive format | `.tar.bz2` |

The model is the "small" English Zipformer, dated 2023-06-26, trained for 99 epochs with averaged checkpoint 1. It is a non-streaming (offline) transducer model.

---

## 9. Gated Model Tracking in ASR

**File:** `revos/asr/sherpa_engine.py:64-75`

Tracking is triggered in `SherpaOnnxASR.__init__()` after model loading:

```python
if manifest.hf_private or manifest.model_url.startswith("http"):
    from revos.usage import track_usage
    track_usage(
        event="model_loaded",
        model_id=str(model_dir),
        model_name=model_name,
        task="asr",
        hf_user=None,
        device=self.device,
    )
```

**Trigger conditions:** `manifest.hf_private == True` OR `manifest.model_url` starts with `"http"`. For the default `zipformer-v2` manifest, the URL is an HTTPS GitHub release link, so tracking **always fires**.

**What is tracked:**

| Field | Value |
|---|---|
| `event` | `"model_loaded"` |
| `model_id` | `str(model_dir)` -- local cache path, not HF model ID |
| `model_name` | e.g. `"zipformer-v2"` |
| `task` | `"asr"` |
| `hf_user` | Always `None` (not resolved at this point) |
| `device` | `"cpu"` or `"cuda"` |
| `timestamp` | ISO 8601 UTC (added by `track_usage`) |

**Where it goes:**

1. **Local log:** `~/.cache/revos/usage.jsonl` -- always written, one JSON object per line.
2. **Registered callbacks:** Any callbacks registered via `revos.usage.register_callback()` are invoked. Exceptions in callbacks are logged as warnings but do not propagate.

**Notable:** `hf_user` is always passed as `None` -- there is no HuggingFace authentication integration in the ASR path. The tracking is effectively a local-only usage log for models downloaded via HTTP.

---

## 10. Test Coverage

### Test files

- `tests/test_asr.py` -- 5 test functions
- `tests/test_audio.py` -- 3 test functions

### `tests/test_asr.py`

| Test | Lines | What it covers |
|---|---|---|
| `test_segment_creation` | 12-15 | `Segment` dataclass instantiation and field access |
| `test_transcript_creation` | 18-22 | `Transcript` dataclass instantiation and segment list |
| `test_asr_transcribe` | 25-79 | Full `SherpaOnnxASR` transcription with mocked sherpa_onnx. Mocks `get`, `ensure_model`, and `sherpa_onnx`. Verifies transcript text, segment count (2 words -> 2 segments), segment text, and language. |
| `test_asr_factory` | 82-107 | `ASR()` factory dispatch. Registers a test manifest, patches `SherpaOnnxASR`, verifies constructor called with `("test", "auto")`. |
| `test_asr_unsupported_backend` | 110-129 | `ASR()` factory raises `ValueError` for unsupported backend. |

**Mocking strategy in `test_asr_transcribe`:**
- `sherpa_onnx` module is fully mocked at import path.
- `OfflineRecognizer.from_transducer` returns a mock recognizer.
- `recognizer.create_stream()` returns a mock stream.
- `stream.result` returns a mock with `text="HELLO WORLD"`, `timestamps=[0.0, 0.5]`, `lang="en"`.
- Model files are created as dummy text files in `tmp_path`.
- Uses a `sample_wav` fixture (likely from conftest.py).

### `tests/test_audio.py`

| Test | Lines | What it covers |
|---|---|---|
| `test_read_mono_wav` | 17-27 | Mono 16kHz WAV: verifies sample rate, dtype float32, correct length |
| `test_read_stereo_converts_to_mono` | 30-43 | Stereo WAV: verifies mono conversion via mean, correct length |
| `test_read_resamples_to_target_sr` | 46-56 | 44100 Hz -> 16000 Hz resampling: verifies output sample rate and length |

**Audio test strategy:** Generates synthetic sine waves using numpy, writes to WAV via `soundfile`, then reads back with `read_waveform`. No external audio files needed.

### Coverage gaps

- **No integration test** with real sherpa-onnx inference (all sherpa calls are mocked).
- **No test for device auto-detection** (`auto_detect_device`).
- **No test for gated model tracking** (usage logging).
- **No test for the download/cache path** (`ensure_model`).
- **No test for edge cases in segment building:** mismatched `timestamps`/`words` lengths, empty audio, very short audio.
- **No test for `read_waveform` with FLAC or non-WAV formats** (soundfile supports them).
- **No test for resampling quality** (linear interpolation is lossy).

---

## Cross-File Dependency Map

```
User call: ASR("zipformer-v2", device="auto")
  |
  +-- revos/asr/__init__.py::ASR()
  |     |
  |     +-- revos/registry/registry.py::get()
  |     |     |
  |     |     +-- revos/registry/manifest.py::ModelManifest  (loaded from YAML at import time)
  |     |
  |     +-- revos/asr/sherpa_engine.py::SherpaOnnxASR.__init__()
  |           |
  |           +-- revos/device.py::auto_detect_device()
  |           |     +-- onnxruntime.get_available_providers()
  |           |
  |           +-- revos/registry/__init__.py::ensure_model()
  |           |     +-- revos/registry/downloader.py::ensure_model()
  |           |           +-- urllib.request.urlretrieve()
  |           |           +-- tarfile / zipfile extraction
  |           |
  |           +-- sherpa_onnx.OfflineRecognizer.from_transducer()
  |           |
  |           +-- revos/usage.py::track_usage()  (conditional)
  |
  +-- .transcribe("audio.wav")
        |
        +-- revos/asr/audio.py::read_waveform()
        |     +-- soundfile.read()
        |     +-- numpy.interp()  (resampling)
        |
        +-- sherpa_onnx stream API:
              create_stream() -> accept_waveform() -> decode_stream()
              -> stream.result -> Segment building -> Transcript
```
