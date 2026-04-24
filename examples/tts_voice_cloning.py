"""Voice cloning example — Synthesize with a reference voice.

Usage:
    uv run python examples/tts_voice_cloning.py
"""

from revos.tts import TTS

if __name__ == "__main__":
    tts = TTS("revovoice")

    # Synthesize with a reference audio for voice cloning
    # Replace with your own reference audio file
    audio = tts.synthesize(
        "This will sound like the reference speaker.",
        ref_audio="examples/reference_speaker.wav",
        ref_text="This is a sample of the speaker talking.",
    )
    audio.save("examples/output_cloned.wav")
    print(f"Saved cloned voice ({audio.duration:.1f}s)")
