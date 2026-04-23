# RevoS

A unified Python library for speech AI — ASR and TTS using open models.

## Installation

```bash
# Core (ASR support)
pip install revos

# With TTS support (OmniVoice — requires PyTorch)
pip install revos[tts]

# With GPU support
pip install revos[gpu]

# Everything
pip install revos[all]

# Or with uv
uv add revos
```

### HuggingFace Login (Required for TTS)

The OmniVoice TTS model is hosted on a private HuggingFace repository. Before using TTS, log in to HuggingFace:

```bash
# Install HF CLI (if not already installed)
pip install huggingface-hub

# Log in — you'll be prompted for your access token
huggingface-cli login

# Or set it via environment variable
export HF_TOKEN=hf_your_token_here
```

Get your token at https://huggingface.co/settings/tokens

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

# Voice design (describe the voice)
audio = tts.synthesize(
    'Good morning everyone.',
    # instruct parameter available via OmniVoice backend
)
```

### CLI

```bash
# Transcribe audio
revos transcribe -m zipformer-v2 audio.wav

# With JSON output
revos transcribe -m zipformer-v2 --json audio.wav

# With SRT subtitles
revos transcribe -m zipformer-v2 --srt audio.wav

# Synthesize speech
revos synthesize -m omnivoice -t "Hello, world!" -o output.wav

# From text file
revos synthesize -m omnivoice -f script.txt -o audiobook.wav
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

## Project Structure

```
revos/
├── revos/
│   ├── asr/           # ASR engine (sherpa-onnx backend)
│   ├── tts/           # TTS engine (OmniVoice backend)
│   ├── registry/      # Model manifest registry + downloader
│   ├── cli/           # Click CLI (revos transcribe / synthesize)
│   ├── device.py      # GPU/CPU auto-detection
│   └── models/        # Bundled YAML manifests
├── tests/
├── pyproject.toml
└── AGENTS.md          # Guide for AI agents / contributors
```

## License

MIT
