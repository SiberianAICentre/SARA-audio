"""Isolated feature-extraction worker used by the Gradio service.

The audio and ML libraries can terminate a Windows process at native level.
Keeping the computation in this child process lets the web server survive and
retain a useful per-job log when that happens.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from sara_audio.asr import FasterWhisperTranscriber
from sara_audio.asr_features import AsrFeatureExtractor
from sara_audio.audio import AudioPreprocessor
from sara_audio.config import ApplicationSettings
from sara_audio.linguistic import LinguisticFeatureExtractor
from sara_audio.mic_features import MicScalarFeatureExtractor
from sara_audio.opensmile_extractor import OpenSmileExtractor
from sara_audio.pipeline import FeaturePipeline
from sara_audio.predictions import JsonLinearModel
from sara_audio.runner import BatchProcessor
from sara_audio.speech_rate import SpeechRateExtractor
from sara_audio.text_embeddings import TextEmbeddingExtractor
from sara_audio.vad import EnergyVadPauseExtractor


def _trace_method(cls: type[Any], method_name: str) -> None:
    original = getattr(cls, method_name)

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        label = f"{cls.__name__}.{method_name}"
        print(f"BEGIN {label}", flush=True)
        result = original(self, *args, **kwargs)
        print(f"END   {label}", flush=True)
        return result

    setattr(cls, method_name, wrapped)


def _enable_stage_tracing() -> None:
    for target, method in (
        (AudioPreprocessor, "prepare"),
        (EnergyVadPauseExtractor, "analyze"),
        (AudioPreprocessor, "write_speech_only_wav"),
        (OpenSmileExtractor, "extract_frames"),
        (FasterWhisperTranscriber, "transcribe"),
        (AsrFeatureExtractor, "extract"),
        (SpeechRateExtractor, "extract"),
        (MicScalarFeatureExtractor, "extract"),
        (TextEmbeddingExtractor, "extract"),
        (LinguisticFeatureExtractor, "extract"),
        (FeaturePipeline, "extract"),
        (BatchProcessor, "_write_record"),
    ):
        _trace_method(target, method)

    original_load = JsonLinearModel.load.__func__

    def traced_load(cls: type[JsonLinearModel], *args: Any, **kwargs: Any) -> JsonLinearModel:
        print("BEGIN JsonLinearModel.load", flush=True)
        result = original_load(cls, *args, **kwargs)
        print("END   JsonLinearModel.load", flush=True)
        return result

    JsonLinearModel.load = classmethod(traced_load)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sara-web-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    faulthandler.enable(all_threads=True)
    _enable_stage_tracing()
    print(f"CONFIG {args.config.resolve()}", flush=True)
    print(f"INPUT  {args.input.resolve()}", flush=True)
    print(f"OUTPUT {args.output.resolve()}", flush=True)
    try:
        settings = ApplicationSettings.from_yaml(args.config)
        # Keep native model owners alive until the process is deliberately
        # terminated. On Windows, tearing down CTranslate2 and PyTorch in the
        # same interpreter can abort after every result has already been written.
        processor = BatchProcessor(settings)
        result = processor.process(args.input, args.output, show_progress=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    print(f"SUCCESS {result.directory}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
