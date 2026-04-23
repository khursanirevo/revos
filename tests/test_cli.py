"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from revos.cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner: CliRunner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "synthesize" in result.output


def test_cli_version(runner: CliRunner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_transcribe_help(runner: CliRunner):
    result = runner.invoke(cli, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--json" in result.output
    assert "--srt" in result.output


def test_synthesize_help(runner: CliRunner):
    result = runner.invoke(cli, ["synthesize", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--text" in result.output
    assert "--output" in result.output
    assert "--ref-audio" in result.output


@patch("revos.asr.ASR")
def test_transcribe_text_output(mock_asr_cls, runner: CliRunner, sample_wav):
    from revos.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO WORLD",
        segments=[
            Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9),
            Segment(start=0.5, end=1.0, text="WORLD", confidence=0.8),
        ],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", sample_wav])
    assert result.exit_code == 0
    assert "HELLO WORLD" in result.output


@patch("revos.asr.ASR")
def test_transcribe_json_output(mock_asr_cls, runner: CliRunner, sample_wav):
    import json

    from revos.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=0.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--json", sample_wav])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["text"] == "HELLO"
    assert len(data["segments"]) == 1


@patch("revos.asr.ASR")
def test_transcribe_srt_output(mock_asr_cls, runner: CliRunner, sample_wav):
    from revos.asr.result import Segment, Transcript

    mock_asr = MagicMock()
    mock_asr.transcribe.return_value = Transcript(
        text="HELLO",
        segments=[Segment(start=0.0, end=1.5, text="HELLO", confidence=0.9)],
        language="en",
    )
    mock_asr_cls.return_value = mock_asr

    result = runner.invoke(cli, ["transcribe", "-m", "test", "--srt", sample_wav])
    assert result.exit_code == 0
    assert "-->" in result.output
    assert "HELLO" in result.output


def test_synthesize_requires_text_or_file(runner: CliRunner):
    result = runner.invoke(cli, ["synthesize", "-m", "test", "-o", "out.wav"])
    assert result.exit_code != 0
