"""Tests for usage tracking."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from revos.usage import (
    _callbacks,
    get_usage_log,
    register_callback,
    track_usage,
)


@pytest.fixture(autouse=True)
def clear_callbacks():
    """Clear callbacks before each test."""
    _callbacks.clear()
    yield
    _callbacks.clear()


def test_track_usage_writes_local(tmp_path: Path):
    """Test that track_usage writes to local JSONL file."""
    log_path = tmp_path / "usage.jsonl"
    with patch("revos.usage._USAGE_LOG", log_path):
        track_usage(
            event="model_loaded",
            model_id="test-model",
            model_name="test",
            task="asr",
            hf_user=None,
            device="cpu",
        )

    assert log_path.exists()
    events = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
    assert len(events) == 1
    assert events[0]["event"] == "model_loaded"
    assert events[0]["model_name"] == "test"
    assert events[0]["task"] == "asr"
    assert events[0]["hf_user"] is None
    assert events[0]["device"] == "cpu"
    assert "timestamp" in events[0]


def test_track_usage_with_hf_user(tmp_path: Path):
    """Test that HF user name is extracted from dict."""
    log_path = tmp_path / "usage.jsonl"
    with patch("revos.usage._USAGE_LOG", log_path):
        track_usage(
            event="model_loaded",
            model_id="revos/test",
            model_name="test",
            task="tts",
            hf_user={"name": "testuser", "fullname": "Test User"},
            device="cuda",
        )

    events = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
    assert events[0]["hf_user"] == "testuser"


def test_track_usage_calls_callbacks(tmp_path: Path):
    """Test that registered callbacks are called."""
    log_path = tmp_path / "usage.jsonl"
    cb = MagicMock()
    register_callback(cb)

    with patch("revos.usage._USAGE_LOG", log_path):
        track_usage(
            event="model_loaded",
            model_id="test",
            model_name="test",
            task="asr",
            hf_user=None,
            device="cpu",
        )

    cb.assert_called_once()
    usage = cb.call_args[0][0]
    assert usage["model_name"] == "test"


def test_callback_exception_does_not_crash(tmp_path: Path):
    """Test that failing callbacks don't prevent other callbacks or tracking."""
    log_path = tmp_path / "usage.jsonl"
    bad_cb = MagicMock(side_effect=RuntimeError("boom"))
    good_cb = MagicMock()
    register_callback(bad_cb)
    register_callback(good_cb)

    with patch("revos.usage._USAGE_LOG", log_path):
        track_usage(
            event="model_loaded",
            model_id="test",
            model_name="test",
            task="asr",
            hf_user=None,
            device="cpu",
        )

    # Good callback still called despite bad one failing
    good_cb.assert_called_once()
    # Local log still written
    assert log_path.exists()


def test_get_usage_log_empty(tmp_path: Path):
    """Test reading usage log when file doesn't exist."""
    log_path = tmp_path / "nonexistent.jsonl"
    with patch("revos.usage._USAGE_LOG", log_path):
        result = get_usage_log()
    assert result == []


def test_get_usage_log_reads_events(tmp_path: Path):
    """Test reading usage log with existing events."""
    log_path = tmp_path / "usage.jsonl"
    log_path.write_text(
        '{"event": "model_loaded", "model_name": "a"}\n'
        '{"event": "model_synthesized", "model_name": "b"}\n'
    )

    with patch("revos.usage._USAGE_LOG", log_path):
        result = get_usage_log()

    assert len(result) == 2
    assert result[0]["model_name"] == "a"
    assert result[1]["model_name"] == "b"


def test_get_usage_log_skips_blank_lines(tmp_path: Path):
    """Test that blank lines in log are skipped."""
    log_path = tmp_path / "usage.jsonl"
    log_path.write_text(
        '{"event": "model_loaded"}\n\n  \n{"event": "model_synthesized"}\n'
    )

    with patch("revos.usage._USAGE_LOG", log_path):
        result = get_usage_log()

    assert len(result) == 2
