"""openSMILE adapter with an explicit semantic coverage report."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sara_audio.domain import FeatureValue
from sara_audio.errors import DependencyUnavailableError


@dataclass(frozen=True)
class OpenSmileSettings:
    """Configuration for the high-coverage ComParE LLD extractor."""

    feature_set: str = "ComParE_2016"
    formant_feature_set: str = "eGeMAPSv02"
    frame_step_seconds: float = 0.01
    aggregate_statistics: tuple[str, ...] = ("mean", "std", "min", "max")


@dataclass(frozen=True)
class OpenSmileOutput:
    """Frame-level values and the exact columns emitted by openSMILE."""

    frames: Any
    columns: tuple[str, ...]
    semantic_coverage: dict[str, tuple[str, ...]]


class OpenSmileExtractor:
    """Extract openSMILE low-level descriptors from normalized WAV input.

    ``ComParE_2016`` is selected because it exposes broad frame-level acoustic
    descriptors. ``eGeMAPSv02`` is additionally run because its LLD contract
    exports LPC-derived F1--F3 frequencies and bandwidths. The returned
    coverage maps requested semantic groups to exact emitted column names,
    making a missing descriptor observable rather than silently substituting a
    different feature.
    """

    method_version = "opensmile-compare-2016-lld-v1"
    _SEMANTIC_MATCHERS = {
        "frame_energy": ("pcm_RMSenergy",),
        "frame_intensity_loudness": ("audspec_lengthL1norm",),
        "mfcc": ("mfcc",),
        "f0_shs": ("F0final", "F0semitone"),
        "f0_acf_cepstrum": ("F0acf", "F0cepstr"),
        "jitter": ("jitter",),
        "shimmer": ("shimmer",),
        "formant_frequency": ("F1frequency", "F2frequency", "F3frequency"),
        "formant_bandwidth": ("F1bandwidth", "F2bandwidth", "F3bandwidth"),
        "psychoacoustic_sharpness": ("sharpness",),
        "spectral_harmonicity": ("HNR",),
        "f0_harmonics_ratios": ("harmonic", "H1", "H2"),
    }

    def __init__(self, settings: OpenSmileSettings | None = None) -> None:
        self._settings = settings or OpenSmileSettings()

    def extract_frames(self, wav_path: Path) -> OpenSmileOutput:
        try:
            import opensmile
        except ImportError as error:
            raise DependencyUnavailableError(
                "openSMILE is not installed. Run: pip install -e ."
            ) from error

        try:
            feature_set = getattr(opensmile.FeatureSet, self._settings.feature_set)
        except AttributeError as error:
            raise ValueError(
                f"Unknown openSMILE feature set: {self._settings.feature_set}"
            ) from error

        smile = opensmile.Smile(
            feature_set=feature_set,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        formant_smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, self._settings.formant_feature_set),
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        frames = smile.process_file(str(wav_path))
        formant_frames = formant_smile.process_file(str(wav_path))
        frames = self._combine_frame_streams(
            frames,
            formant_frames,
            base_name=self._settings.feature_set,
            extra_name=self._settings.formant_feature_set,
        )
        columns = tuple(str(column) for column in frames.columns)
        coverage = {
            semantic: tuple(
                column
                for column in columns
                if any(token.lower() in column.lower() for token in tokens)
            )
            for semantic, tokens in self._SEMANTIC_MATCHERS.items()
        }
        return OpenSmileOutput(frames=frames, columns=columns, semantic_coverage=coverage)

    def aggregate(self, output: OpenSmileOutput) -> tuple[FeatureValue, ...]:
        """Summarize selected acoustic LLDs into auditable scalar features."""
        selected = {
            column
            for columns in output.semantic_coverage.values()
            for column in columns
        }
        values: list[FeatureValue] = []
        for column in sorted(selected):
            numeric = output.frames[column].dropna()
            if numeric.empty:
                continue
            for statistic in self._settings.aggregate_statistics:
                value = self._statistic(numeric, statistic)
                if value is None:
                    continue
                values.append(
                    FeatureValue(
                        code=f"ACOUSTIC_{self._safe_code(column)}_{statistic}",
                        value=value,
                        unit="openSMILE-LLD",
                        source="opensmile-aggregate",
                        confidence=0.7,
                        method_version=f"{self.method_version}-aggregate-v1",
                        evidence={
                            "column": column,
                            "statistic": statistic,
                            "frame_count": int(numeric.shape[0]),
                        },
                    )
                )
        return tuple(values)

    @staticmethod
    def _safe_code(column: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", column).strip("_")

    @staticmethod
    def _statistic(values: Any, statistic: str) -> float | None:
        methods = {
            "mean": values.mean,
            "std": values.std,
            "min": values.min,
            "max": values.max,
        }
        if statistic not in methods:
            raise ValueError(
                "opensmile.aggregate_statistics accepts mean, std, min, and max"
            )
        value = float(methods[statistic]())
        return value if math.isfinite(value) else None

    @staticmethod
    def _combine_frame_streams(
        base_frames: Any,
        extra_frames: Any,
        base_name: str,
        extra_name: str,
    ) -> Any:
        """Preserve each LLD stream's exact analysis-window timestamps.

        ComParE uses 60 ms windows while eGeMAPS uses 20 ms windows. Both
        advance in 10 ms steps, but their ``end`` timestamps differ. A wide
        join would produce sparse, ambiguous rows or relabel one stream's
        timestamps. Stacking the streams and labeling their source retains the
        exact frame interval for every descriptor.
        """
        base = base_frames.copy()
        base.insert(0, "frame_source", base_name)
        extra = extra_frames.copy()
        extra.insert(0, "frame_source", extra_name)
        try:
            import pandas as pd
        except ImportError as error:
            raise DependencyUnavailableError(
                "pandas is required to combine openSMILE frame streams."
            ) from error
        return pd.concat([base, extra], axis=0, sort=False).sort_index()
