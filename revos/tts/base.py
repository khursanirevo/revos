"""Abstract base class for TTS engines."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from .result import Audio

logger = logging.getLogger(__name__)

# Sentence-ending punctuation followed by whitespace or CJK punctuation
# Handles both English (. ! ? + space) and CJK (。！？ no space needed)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")

# Default max chars per chunk (most TTS models handle up to ~500 well)
DEFAULT_MAX_CHARS = 500


def _split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Tries to keep sentences together. If a single sentence exceeds
    max_chars, it is split at comma/semicolon boundaries, then at
    word boundaries as a last resort.

    Args:
        text: Input text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks, each within max_chars.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Split on sentence boundaries
    sentences = _SENTENCE_RE.split(text)
    # Filter empty strings from split
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    # Handle any chunk that still exceeds max_chars — split at commas
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_long_chunk(chunk, max_chars))

    return final_chunks


def _split_long_chunk(text: str, max_chars: int) -> list[str]:
    """Split a chunk that exceeds max_chars at comma/word boundaries."""
    # Try splitting at commas or semicolons
    parts = re.split(r"(?<=[,;，；])\s*", text)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        # Last resort: split at word boundaries
        words = text.split()
        parts = []
        current = ""
        for word in words:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_chars:
                current = current + " " + word
            else:
                parts.append(current)
                current = word
        if current:
            parts.append(current)
        return parts if parts else [text]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= max_chars:
            current = current + " " + part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


class BaseTTS(ABC):
    """Base class for text-to-speech engines."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: str | None = None,
        *,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            output_path: Optional path to save the audio file.
            speed: Speech speed multiplier (1.0 = normal).
            ref_audio: Optional reference audio for voice cloning.
            ref_text: Optional transcription of the reference audio.

        Returns:
            Audio object with samples and sample rate.
        """
        ...

    def synthesize_long(
        self,
        text: str,
        output_path: str | None = None,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        silence_duration: float = 0.1,
        speed: float = 1.0,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> Audio:
        """Synthesize long text by splitting into chunks and
        concatenating audio.

        Automatically splits text at sentence boundaries, synthesizes
        each chunk, and joins the audio with short silence gaps.

        Args:
            text: Long text to synthesize.
            output_path: Optional path to save the concatenated audio.
            max_chars: Maximum characters per chunk (default 500).
            silence_duration: Seconds of silence between chunks
                (default 0.1).
            speed: Speech speed multiplier.
            ref_audio: Optional reference audio for voice cloning.
            ref_text: Optional transcription of reference audio.

        Returns:
            A single Audio with all chunks concatenated.
        """
        chunks = _split_text(text, max_chars=max_chars)

        if not chunks:
            raise ValueError("Text is empty — nothing to synthesize")

        logger.info(
            "synthesize_long: splitting into %d chunk(s)", len(chunks)
        )

        segments: list[Audio] = []
        for i, chunk in enumerate(chunks):
            logger.debug("synthesizing chunk %d/%d", i + 1, len(chunks))
            audio = self.synthesize(
                chunk,
                speed=speed,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            segments.append(audio)

        result = Audio.concatenate(segments, silence_duration=silence_duration)

        if output_path:
            result.save(output_path)
            logger.info("Saved long audio (%.1fs) to %s", result.duration, output_path)

        return result
