"""Interfaces that keep extraction backends replaceable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from sara_audio.domain import FeatureValue, Transcript


class FeatureExtractor(ABC):
    """Extract a self-contained collection of feature values from an input."""

    @abstractmethod
    def extract(self, input_path: Path) -> tuple[FeatureValue, ...]:
        """Return the extractor's feature values for one prepared input."""


class AsrTranscriber(ABC):
    """Produce a timestamped transcript for one audio file."""

    @abstractmethod
    def transcribe(self, input_path: Path) -> Transcript:
        """Transcribe one audio file."""
