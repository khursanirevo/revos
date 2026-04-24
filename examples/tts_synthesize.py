"""TTS example — Synthesize speech from text.

Usage:
    uv run python examples/tts_synthesize.py
"""

from revos.tts import TTS

if __name__ == "__main__":
    tts = TTS("revovoice")

    # Basic synthesis
    audio = tts.synthesize("Hello, how are you today?")
    audio.save("examples/output_hello.wav")
    print(f"Saved hello ({audio.duration:.1f}s)")

    # Long text (auto-splits and concatenates)
    long_text = (
        "RevoS is a unified Python library for speech AI. "
        "It supports automatic speech recognition and text to speech. "
        "You can use it to transcribe audio files or synthesize speech "
        "from text. It supports multiple languages and voice cloning."
    )
    audio = tts.synthesize_long(long_text)
    audio.save("examples/output_long.wav")
    print(f"Saved long text ({audio.duration:.1f}s)")
