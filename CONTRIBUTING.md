# Contributing to RevoS

Thank you for your interest in contributing! This guide covers everything you need.

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd revos
uv sync --extra dev

# Run checks
uv run ruff check revos/ tests/
uv run pytest tests/ -v

# Build
uv build
```

## Adding a New Model (Zero Code Changes)

If the model uses an existing backend (sherpa-onnx for ASR, revovoice for TTS), just add a YAML manifest:

```yaml
# ~/.config/revos/models/asr/my-model.yaml
name: my-model
task: asr
backend: sherpa-onnx
model_type: transducer
model_url: "https://example.com/model.tar.bz2"
sample_rate: 16000
language: en
description: "My custom ASR model"
files:
  encoder: "encoder.onnx"
  decoder: "decoder.onnx"
  joiner: "joiner.onnx"
  tokens: "tokens.txt"
```

Then use it: `from revos.asr import ASR; ASR('my-model')`

## Adding a New Backend

1. Create `revos/{task}/{backend}_engine.py` inheriting from the base class
2. Register in the factory function (`revos/{task}/__init__.py`)
3. Add optional dependency to `pyproject.toml`
4. Add tests with mocked backend
5. Add at least one YAML manifest

See [AGENTS.md](AGENTS.md) for detailed instructions.

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes with tests
3. Run lint and tests:
   ```bash
   uv run ruff check revos/ tests/
   uv run pytest tests/ -v
   ```
4. Commit with clear messages
5. Open a pull request

## Code Style

- Python 3.11+, formatted by ruff (line length 88)
- Lazy imports for optional dependencies (omnivoice, torch)  # omnivoice is the pip package; revovoice is the model/backend name
- Factory functions as public API (not classes)
- YAML manifests for model configuration

## Testing

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=revos --cov-report=term-missing

# Only fast tests (exclude real inference)
uv run pytest tests/ -v -m "not slow"
```

## Project Structure

```
revos/
  asr/           # ASR engine
  tts/           # TTS engine
  registry/      # Model manifest registry + downloader
  cli/           # Click CLI
  models/        # Bundled YAML manifests
tests/           # Test suite
```

## Need Help?

Check [AGENTS.md](AGENTS.md) for detailed architecture docs and extension guides.
