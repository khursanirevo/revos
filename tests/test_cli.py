"""Tests for CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


@patch("revos.tts.TTS")
def test_synthesize_text_output(mock_tts_cls, runner: CliRunner):
    """Test synthesize command with text input and output."""
    import numpy as np

    from revos.tts.result import Audio

    mock_tts = MagicMock()
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_tts.synthesize.return_value = Audio(samples=samples, sample_rate=24000)
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli, ["synthesize", "-m", "test", "-t", "Hello", "-o", "out.wav"]
        )
    assert result.exit_code == 0
    assert "Saved" in result.output


@patch("revos.tts.TTS")
def test_synthesize_from_file(mock_tts_cls, runner: CliRunner):
    """Test synthesize command reading from text file."""
    import numpy as np

    from revos.tts.result import Audio

    mock_tts = MagicMock()
    samples = np.random.randn(24000).astype(np.float32) * 0.1
    mock_tts.synthesize.return_value = Audio(samples=samples, sample_rate=24000)
    mock_tts_cls.return_value = mock_tts

    with runner.isolated_filesystem():
        with open("input.txt", "w") as f:
            f.write("Hello from file")

        result = runner.invoke(
            cli, ["synthesize", "-m", "test", "-f", "input.txt", "-o", "out.wav"]
        )
    assert result.exit_code == 0
