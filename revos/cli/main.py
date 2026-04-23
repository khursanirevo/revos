"""RevoS CLI — Command-line interface for speech AI.

Usage:
    revos transcribe --model zipformer-v2 audio.wav
    revos synthesize --model omnivoice --text "Hello" -o output.wav
"""

from __future__ import annotations

import json
import sys

import click


@click.group()
@click.version_option()
def cli() -> None:
    """RevoS — A unified library for speech AI (ASR & TTS)."""


@cli.command()
@click.option("--model", "-m", required=True, help="ASR model name (e.g. zipformer-v2)")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--srt", "as_srt", is_flag=True, help="Output as SRT subtitles")
def transcribe(model: str, audio_path: str, as_json: bool, as_srt: bool) -> None:
    """Transcribe an audio file to text."""
    from revos.asr import ASR

    asr = ASR(model)
    result = asr.transcribe(audio_path)

    if as_json:
        data = {
            "text": result.text,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
            "language": result.language,
        }
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
    elif as_srt:
        for i, seg in enumerate(result.segments, 1):
            start_ts = _format_srt_time(seg.start)
            end_ts = _format_srt_time(seg.end)
            click.echo(f"{i}")
            click.echo(f"{start_ts} --> {end_ts}")
            click.echo(seg.text)
            click.echo()
    else:
        click.echo(result.text)


@cli.command()
@click.option("--model", "-m", required=True, help="TTS model name (e.g. omnivoice)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Text file to synthesize"
)
@click.option("--output", "-o", required=True, type=click.Path(), help="Output audio path")
@click.option("--speed", default=1.0, help="Speech speed (default: 1.0)")
@click.option(
    "--ref-audio", type=click.Path(exists=True), help="Reference audio for voice cloning"
)
@click.option("--ref-text", help="Transcription of reference audio")
def synthesize(
    model: str,
    text: str | None,
    file: str | None,
    output: str,
    speed: float,
    ref_audio: str | None,
    ref_text: str | None,
) -> None:
    """Synthesize speech from text."""
    from revos.tts import TTS

    if text is None and file is None:
        raise click.UsageError("Either --text or --file must be provided")

    if text is None and file is not None:
        with open(file) as f:
            text = f.read().strip()

    assert text is not None

    tts = TTS(model)
    audio = tts.synthesize(
        text,
        output,
        speed=speed,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    click.echo(f"Saved {len(audio.samples)} samples ({len(audio.samples) / audio.sample_rate:.1f}s) to {output}")


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    cli()
