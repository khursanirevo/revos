# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-24

### Added

- ASR engine with sherpa-onnx backend (zipformer-v2 model)
- TTS engine with RevoVoice backend (revovoice model)
- Model registry with YAML manifests (bundled + user)
- Model downloader with security hardening
  (tarball filter, zip path traversal protection, URL validation)
- CLI: `revos transcribe`, `revos synthesize`, `revos models`, `revos info`
- CLI: `revos catalog list`, `revos catalog pull`
- Remote model catalog fetching from GitHub repo
- `synthesize_long()` for automatic text splitting and audio concatenation
- `Audio.concatenate()` for joining audio segments
- `Audio.duration` property
- Model version pinning via `revision` field in YAML manifests
- Gated model access with clear error messages (401/403)
- Usage tracking for gated models (local JSONL log)
- Device auto-detection (GPU/CPU)
- HuggingFace authentication check
- Pre-commit hooks (ruff lint + format)
- CI workflow with coverage reporting
- MIT license

### Changed

- Renamed model from omnivoice to revovoice (backend and model name)
- Catalog fetches from this GitHub repo instead of separate HF repo

### Security

- Zip path traversal protection in model extraction
- URL scheme validation (only http/https)
- Usage log file permissions set to 0o600
- Tarball extraction uses `filter="data"`
