"""RevoS CLI — Command-line interface for speech AI.

Usage:
    revos transcribe --model zipformer-v2 audio.wav
    revos synthesize --model revovoice --text "Hello" -o output.wav
"""

from __future__ import annotations

import json
from pathlib import Path

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
@click.option("--model", "-m", required=True, help="TTS model name (e.g. revovoice)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option(
    "--file", "-f", type=click.Path(exists=True), help="Text file to synthesize"
)
@click.option(
    "--output", "-o", required=True, type=click.Path(), help="Output audio path"
)
@click.option("--speed", default=1.0, help="Speech speed (default: 1.0)")
@click.option(
    "--ref-audio",
    type=click.Path(exists=True),
    help="Reference audio for voice cloning",
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

    # Auto-detect long text and use synthesize_long
    if len(text) > 500:
        audio = tts.synthesize_long(
            text,
            output,
            speed=speed,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    else:
        audio = tts.synthesize(
            text,
            output,
            speed=speed,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    click.echo(
        f"Saved {len(audio.samples)} samples "
        f"({len(audio.samples) / audio.sample_rate:.1f}s) to {output}"
    )


@cli.command()
@click.option("--task", "-t", help="Filter by task type (asr or tts)")
def models(task: str | None) -> None:
    """List available models."""
    from revos.registry import list_models

    results = list_models(task)
    if not results:
        click.echo("No models found.")
        return

    click.echo(f"{'Name':<20} {'Task':<6} {'Backend':<15} {'Language':<12}")
    click.echo("-" * 53)
    for m in results:
        click.echo(f"{m.name:<20} {m.task:<6} {m.backend:<15} {m.language:<12}")


@cli.command()
def info() -> None:
    """Show environment and configuration info."""
    import sys

    click.echo(f"RevoS version:   {_get_version()}")
    click.echo(f"Python:          {sys.version.split()[0]}")

    # Device
    from revos.device import auto_detect_device

    click.echo(f"Device:          {auto_detect_device()}")

    # Models
    from revos.registry import list_models

    click.echo(f"Models loaded:   {len(list_models())}")

    # Cache dir
    cache_dir = Path.home() / ".cache" / "revos"
    click.echo(f"Cache dir:       {cache_dir}")

    # Catalog repo
    from revos.catalog import get_catalog_repo

    click.echo(f"Catalog repo:    {get_catalog_repo()}")

    # HF auth
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami()
        click.echo(f"HuggingFace:     {user.get('name', 'unknown')}")
    except Exception:
        click.echo("HuggingFace:     not logged in")


@cli.group()
def catalog() -> None:
    """Browse and pull models from the remote catalog."""


@catalog.command("list")
@click.option("--task", "-t", help="Filter by task type (asr or tts)")
def catalog_list(task: str | None) -> None:
    """List models available in the remote catalog."""
    from revos.catalog import get_catalog_repo, list_catalog

    click.echo(f"Fetching catalog from {get_catalog_repo()}...")
    try:
        results = list_catalog(task)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    if not results:
        click.echo("No models found in catalog.")
        return

    click.echo(
        f"{'Name':<20} {'Task':<6} {'Backend':<15} "
        f"{'Language':<12} {'Version':<12}"
    )
    click.echo("-" * 65)
    for m in results:
        rev = m.revision or "latest"
        click.echo(
            f"{m.name:<20} {m.task:<6} {m.backend:<15} "
            f"{m.language:<12} {rev:<12}"
        )
    click.echo("\nUse 'revos catalog pull <name>' to install.")


@catalog.command("pull")
@click.argument("model_name")
def catalog_pull(model_name: str) -> None:
    """Pull a model from the catalog and install it locally."""
    from revos.catalog import get_catalog_repo, pull_model

    click.echo(f"Pulling '{model_name}' from {get_catalog_repo()}...")
    try:
        dest = pull_model(model_name)
    except (KeyError, RuntimeError) as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Installed to {dest}")
    click.echo(f"Use: from revos.tts import TTS; TTS('{model_name}')")


def _get_version() -> str:
    """Get revos version without triggering heavy imports."""
    from importlib.metadata import version

    try:
        return version("revos")
    except Exception:
        return "unknown"


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    cli()
