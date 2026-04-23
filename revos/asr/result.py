"""ASR result data classes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    """A single transcription segment with timing and confidence."""

    start: float
    end: float
    text: str
    confidence: float


@dataclass
class Transcript:
    """Full transcription result from ASR."""

    text: str
    segments: list[Segment]
    language: str
