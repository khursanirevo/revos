"""Usage tracking for gated models.

Provides hooks to track which users load and use gated models.
Configure via ~/.config/revos/config.yaml or environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Type for usage callbacks
UsageCallback = Callable[[dict], None]

# Registered callbacks
_callbacks: list[UsageCallback] = []

# Local usage log path
_USAGE_LOG = Path.home() / ".cache" / "revos" / "usage.jsonl"


def register_callback(callback: UsageCallback) -> None:
    """Register a callback to be called when a gated model is loaded.

    The callback receives a dict with:
        - model_id: HuggingFace model ID
        - model_name: revos model name
        - task: "asr" or "tts"
        - hf_user: HuggingFace username (or None)
        - device: "cpu" or "cuda"
        - timestamp: ISO 8601 UTC timestamp
        - event: "model_loaded" or "model_synthesized"

    Args:
        callback: Function that takes a usage dict.
    """
    _callbacks.append(callback)


def _log_to_local(usage: dict) -> None:
    """Append usage event to local JSONL log."""
    _USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_USAGE_LOG, "a") as f:
        f.write(json.dumps(usage, default=str) + "\n")


def track_usage(
    event: str,
    model_id: str,
    model_name: str,
    task: str,
    hf_user: dict | None,
    device: str,
    **extra: object,
) -> None:
    """Record a usage event and notify all registered callbacks.

    Args:
        event: Event type ("model_loaded", "model_synthesized").
        model_id: HuggingFace model ID.
        model_name: RevoS model name.
        task: "asr" or "tts".
        hf_user: HF user info dict (or None).
        device: Device used ("cpu" or "cuda").
        **extra: Additional data to include.
    """
    usage = {
        "event": event,
        "model_id": model_id,
        "model_name": model_name,
        "task": task,
        "hf_user": hf_user["name"] if hf_user else None,
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }

    # Always log locally
    _log_to_local(usage)

    # Notify registered callbacks
    for callback in _callbacks:
        try:
            callback(usage)
        except Exception as e:
            logger.warning("Usage callback failed: %s", e)

    logger.debug("Usage tracked: %s %s by %s", event, model_id, usage.get("hf_user"))


def get_usage_log() -> list[dict]:
    """Read all usage events from the local log.

    Returns:
        List of usage event dicts, most recent last.
    """
    if not _USAGE_LOG.exists():
        return []
    events = []
    with open(_USAGE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
