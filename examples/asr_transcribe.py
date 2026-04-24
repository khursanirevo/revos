"""ASR example — Transcribe audio to text.

Usage:
    uv run python examples/asr_transcribe.py audio.wav
"""

import sys

from revos.asr import ASR

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python asr_transcribe.py <audio.wav>")
        sys.exit(1)

    asr = ASR("zipformer-v2")
    result = asr.transcribe(sys.argv[1])

    print(f"Language: {result.language}")
    print(f"Text: {result.text}")
    print(f"\nSegments:")
    for seg in result.segments:
        print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")
