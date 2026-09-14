#!/usr/bin/env bash
set -euo pipefail

INPUT_PATH="${1:-${SARA_INPUT_PATH:-/data/input}}"
OUTPUT_PATH="${2:-${SARA_OUTPUT_PATH:-/data/output/server-$(date +%Y%m%d-%H%M%S)}}"
CONFIG_PATH="${SARA_CONFIG_PATH:-/app/config/rtx4090_cuda12.yaml}"
SMOKE_INPUT_PATH="${SARA_SMOKE_INPUT_PATH:-}"
SMOKE_OUTPUT_PATH="${OUTPUT_PATH}-smoke"

if [[ ! -e "$INPUT_PATH" ]]; then
  echo "Input path does not exist: $INPUT_PATH" >&2
  exit 2
fi
if [[ -e "$OUTPUT_PATH" ]]; then
  echo "Output path already exists: $OUTPUT_PATH" >&2
  exit 2
fi
if [[ -e "$SMOKE_OUTPUT_PATH" ]]; then
  echo "Smoke output path already exists: $SMOKE_OUTPUT_PATH" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

if [[ -f "$INPUT_PATH/test.zip" && -d "$INPUT_PATH/train" ]]; then
  echo "[1/4] Preparing train + test corpus..."
  sara-prepare-corpus "$INPUT_PATH"
else
  echo "[1/4] Corpus preparation skipped."
fi

if [[ "${SARA_SKIP_SMOKE:-0}" != "1" ]]; then
  if [[ -z "$SMOKE_INPUT_PATH" ]]; then
    mapfile -t audio_files < <(
      find "$INPUT_PATH" -type f \
        \( -iname '*.wav' -o -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.flac' -o -iname '*.ogg' \) \
        | sort
    )
    first_audio="${audio_files[0]:-}"
    if [[ -z "$first_audio" ]]; then
      echo "No supported audio file found under $INPUT_PATH" >&2
      exit 2
    fi
    smoke_dir="$(mktemp -d)"
    cp "$first_audio" "$smoke_dir/"
    SMOKE_INPUT_PATH="$smoke_dir"
  fi
  echo "[2/4] Running CUDA smoke test..."
  sara-server-smoke "$SMOKE_INPUT_PATH" "$SMOKE_OUTPUT_PATH" --config "$CONFIG_PATH"
else
  echo "[2/4] Smoke test skipped by SARA_SKIP_SMOKE=1."
fi

echo "[3/4] Processing full corpus..."
sara-extract "$INPUT_PATH" "$OUTPUT_PATH" --config "$CONFIG_PATH"

if [[ "${SARA_PACKAGE_DELIVERY:-0}" == "1" ]]; then
  echo "[4/4] Packaging delivery..."
  sara-package-server-run "$OUTPUT_PATH" "${OUTPUT_PATH}-delivery"
else
  echo "[4/4] Delivery packaging skipped. Set SARA_PACKAGE_DELIVERY=1 to enable it."
fi

echo "Finished. Results: $OUTPUT_PATH"
