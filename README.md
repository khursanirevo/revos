# RevoS

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/revos/revos/ci.yml?branch=main)](.github/workflows/ci.yml)

A unified Python library for speech AI — ASR and TTS using open models.

## Installation

```bash
# Core (ASR support)
pip install revos

# With TTS support (OmniVoice — requires PyTorch)
pip install revos[tts]

# With GPU support
pip install revos[gpu]

# Everything (GPU + TTS)
pip install revos[all]

# Or with uv
uv add revos
```

### HuggingFace Login (Required for TTS)

> **Note:** The OmniVoice TTS model is hosted on a private HuggingFace repository. You **must** log in before using TTS.

```bash
pip install huggingface-hub
huggingface-cli login
```

Get your token at https://huggingface.co/settings/tokens

### Important Notes

- `revos[gpu]` and `revos[all]` install `onnxruntime-gpu`, which **conflicts** with `onnxruntime`. If you already have `revos` installed, uninstall it first before installing the GPU variant.
- Audio formats supported: WAV, FLAC, OGG, and any format supported by `libsndfile`.

## Quick Start

### ASR (Automatic Speech Recognition)

```python
from revos.asr import ASR

asr = ASR('zipformer-v2')
result = asr.transcribe('meeting.wav')

print(result.text)        # Full transcript
print(result.language)    # Detected language
for seg in result.segments:
    print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
```

### TTS (Text-to-Speech)

```python
from revos.tts import TTS

# Basic synthesis
tts = TTS('omnivoice')
audio = tts.synthesize('Hello, how are you?')
audio.save('greeting.wav')

# Voice cloning (with reference audio)
audio = tts.synthesize(
    'This will sound like the reference speaker.',
    ref_audio='speaker.wav',
    ref_text='Sample of the speaker talking.',
)
audio.save('cloned.wav')
```

### CLI

```bash
# Transcribe audio
revos transcribe -m zipformer-v2 audio.wav

# JSON output
revos transcribe -m zipformer-v2 --json audio.wav

# SRT subtitles
revos transcribe -m zipformer-v2 --srt audio.wav

# Synthesize speech
revos synthesize -m omnivoice -t "Hello, world!" -o output.wav

# From text file
revos synthesize -m omnivoice -f script.txt -o audiobook.wav

# List available models
revos models

# Show environment info
revos info
```

## Available Models

| Model | Task | Backend | Languages | Description |
|-------|------|---------|-----------|-------------|
| `zipformer-v2` | ASR | sherpa-onnx | English | Zipformer small transducer model |
| `omnivoice` | TTS | OmniVoice | 600+ | Zero-shot multilingual TTS with voice cloning |

## Adding Custom Models

Add a YAML manifest to `~/.config/revos/models/`:

```yaml
# ~/.config/revos/models/asr/my-model.yaml
name: my-custom-model
task: asr
backend: sherpa-onnx
model_type: transducer
model_url: "https://example.com/models/my-model.tar.bz2"
sample_rate: 16000
language: en
description: "My custom ASR model"
files:
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
```

Then use it: `from revos.asr import ASR; asr = ASR('my-custom-model')`

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture guide for AI agents and contributors
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute

## Project Structure

```
revos/
├── revos/
│   ├── asr/           # ASR engine (sherpa-onnx backend)
│   ├── tts/           # TTS engine (OmniVoice backend)
│   ├── registry/      # Model manifest registry + downloader
│   ├── cli/           # Click CLI (revos transcribe / synthesize / models / info)
│   ├── device.py      # GPU/CPU auto-detection
│   └── models/        # Bundled YAML manifests
├── tests/
├── pyproject.toml
├── AGENTS.md
└── CONTRIBUTING.md
```

## License

MIT
