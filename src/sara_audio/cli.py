"""Minimal command-line entry points for SARA-audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sara_audio.asr import asr_runtime_diagnostics
from sara_audio.audio import AudioPreprocessor
from sara_audio.config import ApplicationSettings
from sara_audio.corpus_prepare import CorpusPreparer
from sara_audio.errors import SaraAudioError
from sara_audio.registry import LinguisticFeatureRegistry
from sara_audio.runner import BatchProcessor
from sara_audio.server_delivery import main as server_delivery_main
from sara_audio.server_smoke import ServerSmokeVerifier

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def build_extract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sara-extract",
        description="Process one audio file or a directory using config/default.yaml.",
    )
    parser.add_argument("input", type=Path, help="Audio file or directory with audio files.")
    parser.add_argument(
        "output_directory",
        nargs="?",
        type=Path,
        help="Optional new output directory. Default: artifacts/<input-name>.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration path (default: config/default.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_extract_parser().parse_args(argv)
    try:
        settings = ApplicationSettings.from_yaml(args.config)
        result = BatchProcessor(settings).process(
            args.input,
            args.output_directory,
            show_progress=True,
        )
    except (OSError, SaraAudioError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Processed {len(result.records)} record(s): {result.directory}")
    print(f"Summary: {result.summary_path}")
    return 0


def validate_registry_main() -> int:
    try:
        registry = LinguisticFeatureRegistry.load_default()
    except SaraAudioError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Registry valid: {len(registry.definitions)} features")
    return 0


def inspect_audio_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sara-inspect-audio")
    parser.add_argument("input", type=Path, help="Normalized WAV file to inspect.")
    args = parser.parse_args(argv)
    try:
        inspection = AudioPreprocessor().inspect_wav(args.input)
    except (OSError, SaraAudioError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(inspection.__dict__, default=str, ensure_ascii=False, indent=2))
    return 0


def diagnose_asr_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sara-diagnose-asr")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load the configured Whisper model to verify its real execution device.",
    )
    args = parser.parse_args(argv)
    try:
        settings = ApplicationSettings.from_yaml(args.config)
        result = asr_runtime_diagnostics(settings.execution.asr, args.load_model)
    except (OSError, SaraAudioError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def server_smoke_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sara-server-smoke")
    parser.add_argument("input", type=Path, help="Stereo smoke-test WAV file or directory.")
    parser.add_argument("output_directory", type=Path, help="New directory for smoke-test artifacts.")
    parser.add_argument("--config", type=Path, default=Path("config/rtx4090_cuda12.yaml"))
    args = parser.parse_args(argv)
    try:
        settings = ApplicationSettings.from_yaml(args.config)
        result = ServerSmokeVerifier().run(settings, args.input, args.output_directory)
    except (OSError, SaraAudioError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def prepare_corpus_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sara-prepare-corpus")
    parser.add_argument("source", type=Path, help="Directory containing train/ and test.zip.")
    args = parser.parse_args(argv)
    try:
        prepared = CorpusPreparer().prepare(args.source)
    except (OSError, SaraAudioError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "input_root": str(prepared.input_root),
                "train_wav_count": prepared.train_wav_count,
                "test_wav_count": prepared.test_wav_count,
                "total_wav_count": prepared.train_wav_count + prepared.test_wav_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def package_server_run_main(argv: list[str] | None = None) -> int:
    try:
        return server_delivery_main(argv)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
