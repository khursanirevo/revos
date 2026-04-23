# TTS Module Analysis

## 1. Module Architecture

The TTS module follows a factory pattern with three layers:

```
revos/tts/__init__.py          -- TTS() factory function
revos/tts/base.py              -- BaseTTS abstract base class
revos/tts/result.py            -- Audio dataclass (output container)
revos/tts/omnivoice_engine.py  -- OmniVoiceTTS concrete engine
```

`__init__.py` exports three names: `TTS`, `BaseTTS`, `Audio`.

**Factory dispatch** (`TTS()` in `__init__.py:19`):
- Looks up the model manifest via `revos.registry.get(model_name, "tts")`.
- Inspects `manifest.backend` and routes to the matching engine class.
- Currently only `backend == "omnivoice"` is wired (line 37), importing `OmniVoiceTTS` lazily.
- Any other backend raises `ValueError(f"Unsupported TTS backend: {manifest.backend} for model '{model_name}'")`.

**Public usage pattern:**
```python
from revos.tts import TTS
tts = TTS("omnivoice")            # factory -> OmniVoiceTTS
audio = tts.synthesize("Hello!")  # returns Audio
audio.save("output.wav")          # writes via soundfile
```

---

## 2. OmniVoiceTTS Engine Implementation

**File:** `revos/tts/omnivoice_engine.py`

**Class signature:**
```python
class OmniVoiceTTS(BaseTTS):
    def __init__(self, model_name: str, device: str = "auto") -> None: ...
    def synthesize(
        self,
        text: str,
        output_path: str | None = None,
        *,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio: ...
```

**Constructor flow (`__init__`, lines 43-118):**

1. Calls `super().__init__(model_name, device)`.
2. Attempts `from omnivoice import OmniVoice` -- on `ImportError`, raises:
   ```
   ImportError: "OmniVoice is required for TTS. Install it with: pip install revos[tts] or pip install omnivoice"
   ```
3. Resolves manifest via `get(model_name, "tts")` and reads `manifest.model_url` as the HuggingFace model ID.
4. Resolves device (see section 6).
5. Identifies the HuggingFace user via `_get_hf_user()` (see section 5).
6. Calls `OmniVoice.from_pretrained(model_id, device_map=device_map)`.
7. Wraps `OSError` from gated repo access into a descriptive `RuntimeError` (see section 10).
8. Stores `self._sample_rate` and `self._model_id` from the manifest.
9. Calls `track_usage(event="model_loaded", ...)` (see section 11).

**Key instance attributes after construction:**
- `self.model_name` -- the revos model name (e.g. `"omnivoice"`)
- `self.device` -- resolved to `"cpu"` or `"cuda"` (never left as `"auto"`)
- `self._model` -- the loaded OmniVoice model instance
- `self._sample_rate` -- from manifest (e.g. `24000`)
- `self._model_id` -- HuggingFace model ID (e.g. `"Revolab/omnivoice"`)
- `self.hf_user` -- dict with `"name"` and `"fullname"`, or `None`

---

## 3. synthesize() End-to-End Flow

**Method body** (`omnivoice_engine.py:120-162`):

```
Input: text, output_path?, speed, ref_audio?, ref_text?
  |
  v
Build kwargs dict: {"text": text, "speed": speed}
  |-- if ref_audio is truthy: kwargs["ref_audio"] = ref_audio
  |     if ref_text is truthy: kwargs["ref_text"] = ref_text
  |
  v
self._model.generate(**kwargs)
  |
  v
result normalization:
  - if result is a non-empty list: samples = np.array(result[0], dtype=np.float32)
  - otherwise: samples = np.array(result, dtype=np.float32)
  |
  v
Audio(samples, sample_rate=self._sample_rate)
  |
  v
if output_path: audio.save(output_path)
  |
  v
return audio
```

The `generate()` call returns either a `list[np.ndarray]` or a single `np.ndarray`. The normalization logic at lines 151-154 handles both cases, always producing a contiguous float32 array.

---

## 4. OmniVoice API Surface (as consumed)

The engine uses only two methods from the `omnivoice` package:

**`OmniVoice.from_pretrained(model_id: str, device_map: str) -> OmniVoice`**
- Called at construction time.
- `model_id` is the HuggingFace repo ID (e.g. `"Revolab/omnivoice"`).
- `device_map` is `"cuda:0"`, `"cpu"`, or a resolved device string.

**`self._model.generate(**kwargs) -> list[np.ndarray] | np.ndarray`**
- Called during `synthesize()`.
- Accepted keyword arguments:
  - `text: str` -- always provided.
  - `speed: float` -- always provided, defaults to `1.0`.
  - `ref_audio: str` -- optional, path to a reference audio file for voice cloning.
  - `ref_text: str` -- optional, only forwarded if `ref_audio` is also provided.

Return format: a list of numpy arrays (each array is one utterance's waveform), or a single numpy array. The engine takes `result[0]` when a list is returned.

---

## 5. HuggingFace Authentication Flow

**`_get_hf_user()` helper** (`omnivoice_engine.py:17-33`):

```python
def _get_hf_user() -> dict | None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.whoami()
        return {
            "name": info.get("name", "unknown"),
            "fullname": info.get("fullname", ""),
        }
    except Exception:
        return None
```

- Uses `huggingface_hub.HfApi().whoami()` which reads the token from `HF_TOKEN` env var or the local HF token cache.
- On success, returns a dict with `"name"` and `"fullname"`.
- On any failure (no token, invalid token, missing `huggingface_hub`), returns `None` silently.
- The constructor logs the username at INFO level if authenticated, or a WARNING if not:
  ```
  WARNING: HuggingFace user not identified. Run 'huggingface-cli login' for access to gated models.
  ```

---

## 6. Device Resolution

**Logic in `OmniVoiceTTS.__init__` (lines 58-69):**

```
if self.device == "auto":
    try:
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    except ImportError:
        self.device = "cpu"
```

Then `device_map` is constructed (line 69):
```python
device_map = f"{self.device}:0" if self.device == "cuda" else self.device
```

| Input `device` | torch available? | CUDA available? | Resolved `device` | `device_map`    |
|-----------------|------------------|-----------------|--------------------|-----------------|
| `"auto"`        | yes              | yes             | `"cuda"`           | `"cuda:0"`      |
| `"auto"`        | yes              | no              | `"cpu"`            | `"cpu"`         |
| `"auto"`        | no               | --              | `"cpu"`            | `"cpu"`         |
| `"cpu"`         | --               | --              | `"cpu"`            | `"cpu"`         |
| `"cuda"`        | --               | --              | `"cuda"`           | `"cuda:0"`      |

Note: if the user explicitly passes `device="cuda"` but CUDA is unavailable, the constructor does not guard against this -- the error would surface later inside `OmniVoice.from_pretrained`.

---

## 7. Voice Cloning Parameters

Voice cloning is controlled by two optional keyword arguments to `synthesize()`:

- **`ref_audio: str | None`** -- Path to a reference audio file. When provided, it is forwarded to `generate()` as `ref_audio`, enabling the model to clone the speaker's voice.
- **`ref_text: str | None`** -- Transcription of the reference audio. Only forwarded to `generate()` when `ref_audio` is also truthy (guard at line 146: `if ref_text: kwargs["ref_text"] = ref_text`).

**Code** (`omnivoice_engine.py:141-146`):
```python
kwargs: dict = {"text": text, "speed": speed}
if ref_audio:
    kwargs["ref_audio"] = ref_audio
    if ref_text:
        kwargs["ref_text"] = ref_text
```

This means `ref_text` without `ref_audio` is silently ignored. The model is described as "zero-shot TTS with voice cloning and design" supporting 600+ languages.

---

## 8. Speed Control

The `speed` parameter is a float multiplier passed directly to `OmniVoice.generate()`:

- **Default:** `1.0` (normal speed)
- **Range:** Not enforced at the engine level -- delegated to the OmniVoice model
- **Propagation:** Always included in the `kwargs` dict, even when no reference audio is provided

---

## 9. Model Manifest for OmniVoice

**File:** `revos/models/tts/omnivoice.yaml`

```yaml
name: omnivoice
task: tts
backend: omnivoice
model_type: diffusion
model_url: "Revolab/omnivoice"
sample_rate: 24000
language: multilingual
description: "OmniVoice multilingual zero-shot TTS with voice cloning and design (600+ languages)"
hf_private: true
files: {}
```

Key fields:
- **`backend: omnivoice`** -- matches the dispatch condition in `TTS()` factory.
- **`model_url: "Revolab/omnivoice"`** -- HuggingFace repo ID, passed to `OmniVoice.from_pretrained()`.
- **`sample_rate: 24000`** -- stored as `self._sample_rate`, used for `Audio` construction and `soundfile.write()`.
- **`hf_private: true`** -- marks this as a gated/private model on HuggingFace (used by manifest metadata; the actual gated-repo detection happens via OSError at load time).
- **`model_type: diffusion`** -- indicates the underlying architecture is diffusion-based.
- **`files: {}`** -- no local file downloads needed; the model is loaded entirely via `from_pretrained()`.

The manifest is loaded into the global registry at startup via `revos.registry.load_manifest()` / `register()`.

---

## 10. Error Handling

### ImportError (missing omnivoice package)

At `omnivoice_engine.py:46-52`:
```python
try:
    from omnivoice import OmniVoice
except ImportError as e:
    raise ImportError(
        "OmniVoice is required for TTS. "
        "Install it with: pip install revos[tts] or pip install omnivoice"
    ) from e
```

### OSError (gated HuggingFace repo)

At `omnivoice_engine.py:92-102`:
```python
try:
    self._model = OmniVoice.from_pretrained(model_id, device_map=device_map)
except OSError as e:
    err = str(e).lower()
    if "gated" in err or "authentication" in err or "401" in err:
        raise RuntimeError(
            f"Cannot access model '{model_id}' -- "
            f"it requires HuggingFace authentication.\n"
            f"Log in with:  huggingface-cli login\n"
            f"Or set:        export HF_TOKEN=your_token\n"
            f"Get a token:  https://huggingface.co/settings/tokens"
        ) from e
    raise  # re-raise non-auth OSErrors unchanged
```

The detection checks three substrings in the lowercase error message: `"gated"`, `"authentication"`, `"401"`. Non-auth `OSError` exceptions are re-raised as-is.

### Unsupported backend

At `__init__.py:42-44`: raises `ValueError(f"Unsupported TTS backend: {manifest.backend} for model '{model_name}'")`.

---

## 11. Usage Tracking on Model Load

At the end of `OmniVoiceTTS.__init__` (lines 109-118):

```python
from revos.usage import track_usage

track_usage(
    event="model_loaded",
    model_id=model_id,
    model_name=model_name,
    task="tts",
    hf_user=self.hf_user,
    device=self.device,
)
```

**`track_usage()`** (in `revos/usage.py:52-93`) performs two actions:

1. **Local JSONL log** -- Appends a JSON event record to `~/.cache/revos/usage.jsonl` with fields: `event`, `model_id`, `model_name`, `task`, `hf_user` (username string or `None`), `device`, `timestamp` (ISO 8601 UTC).

2. **Callback dispatch** -- Iterates registered callbacks (`_callbacks` list) and calls each with the usage dict. Callback failures are logged as warnings but do not raise.

Callbacks can be registered via `revos.usage.register_callback(fn)`. This is an extension point for external telemetry or analytics.

---

## 12. Test Coverage and Mocking Strategy

**File:** `tests/test_tts.py` (176 lines, 6 tests)

### Test fixtures

- **`clear_registry`** (autouse, lines 44-49): Clears `_models` dict before and after every test, ensuring test isolation.

### Helper

- **`_make_mock_omnivoice()`** (lines 51-60): Creates a synthetic `omnivoice` module using `types.ModuleType`. Returns a 3-tuple of `(mock_module, mock_cls, mock_model)`. The mock model's `generate()` returns a list with one random float32 array of 24000 samples.

### Test cases

| Test | Lines | What it verifies |
|------|-------|------------------|
| `test_audio_creation` | 18-22 | `Audio` dataclass stores samples and sample_rate correctly |
| `test_audio_save` | 25-36 | `Audio.save()` writes a valid WAV that `soundfile` can read back with correct sr/length |
| `test_audio_dataclass` | 39-42 | Array values are preserved exactly (within tolerance) |
| `test_omnivoice_engine_synthesize` | 64-91 | Full synthesis flow: registers manifest, injects mock omnivoice module via `sys.modules` patch, calls `synthesize("Hello world")`, verifies returned `Audio` has `sample_rate=24000` and that `generate` was called with `text="Hello world", speed=1.0` |
| `test_omnivoice_engine_save_to_file` | 94-121 | Verifies that passing `output_path` causes the WAV file to be written to disk |
| `test_omnivoice_engine_gated_error` | 124-154 | Mocks `from_pretrained` to raise `OSError("gated repo, please authenticate")`, verifies `OmniVoiceTTS()` constructor re-raises as `RuntimeError` matching `"HuggingFace authentication"` |
| `test_tts_unsupported_backend` | 157-175 | Registers a manifest with `backend="nonexistent"`, verifies `TTS()` factory raises `ValueError` matching `"Unsupported TTS backend"` |

### Mocking strategy

1. **`omnivoice` package injection**: Tests use `patch.dict(sys.modules, {"omnivoice": mock_module})` to inject a synthetic module, avoiding any real dependency on the `omnivoice` package.

2. **HuggingFace auth mock**: All engine tests use `@patch("revos.tts.omnivoice_engine._get_hf_user", return_value=None)` to bypass real HF authentication.

3. **Registry isolation**: The autouse `clear_registry` fixture ensures each test starts with a clean global model registry.

4. **No network calls**: All external interactions (HF hub, model download) are fully mocked. Tests run offline.

### Coverage gaps

- No test for `speed` values other than the default `1.0`.
- No test for voice cloning parameters (`ref_audio`, `ref_text`) being forwarded to `generate()`.
- No test for the device resolution logic (auto -> cuda/cpu).
- No test for non-auth `OSError` being re-raised unchanged.
- No test for `_get_hf_user()` returning a non-None user dict.
- No negative test for `Audio.save()` (invalid path, permissions).
