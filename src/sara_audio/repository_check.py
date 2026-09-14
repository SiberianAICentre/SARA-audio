"""Static validation of files required for a reproducible SARA checkout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import unquote

from sara_audio.config import ApplicationSettings
from sara_audio.predictions import EXPERT_MODEL_NAMES, JsonLinearModel

REQUIRED_CONFIGS = (
    "default.yaml",
    "native_cpu.yaml",
    "rtx4090_cuda11_8.yaml",
    "rtx4090_cuda12.yaml",
    "external_delivery.yaml",
)
REQUIRED_DOCUMENTS = (
    "README.md",
    "CONTRIBUTING.md",
    "WINDOWS_SERVER_QUICKSTART.md",
    "docker/rtx4090/README.md",
    "docs/ARCHITECTURE.md",
    "docs/CONFIGURATION.md",
    "docs/CLI.md",
    "docs/METHODOLOGY.md",
    "docs/OUTPUTS.md",
    "docs/DEPLOYMENT.md",
    "docs/DEVELOPMENT.md",
    "docs/TROUBLESHOOTING.md",
    "SECURITY.md",
)
MODEL_NAMES = (
    "filter",
    *EXPERT_MODEL_NAMES,
    "agent_sex",
    "burnout",
)
DOCKERFILES = (
    "docker/rtx4090/Dockerfile",
    "docker/rtx4090/Dockerfile.cuda11_8",
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def verify_repository(root: Path) -> dict[str, Any]:
    """Validate repository contracts without loading any heavyweight model."""
    root = root.resolve()
    errors: list[str] = []
    for relative_path in (*REQUIRED_DOCUMENTS, "pyproject.toml"):
        if not (root / relative_path).is_file():
            errors.append(f"Required file is missing: {relative_path}")

    for relative_path in REQUIRED_DOCUMENTS:
        path = root / relative_path
        if not path.is_file():
            continue
        for raw_target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if Path(target).is_absolute() or PureWindowsPath(target).is_absolute():
                errors.append(
                    f"Non-portable absolute Markdown link in {relative_path}: {target}"
                )
                continue
            if not (path.parent / target).resolve().exists():
                errors.append(f"Broken Markdown link in {relative_path}: {target}")

    model_details: list[dict[str, object]] = []
    model_directory = root / "models" / "methodology"
    for name in MODEL_NAMES:
        path = model_directory / f"{name}.json"
        if not path.is_file():
            errors.append(f"Methodology model is missing: {path.relative_to(root)}")
            continue
        try:
            model = JsonLinearModel.load(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"Invalid methodology model {path.relative_to(root)}: {error}")
            continue
        model_details.append(
            {
                "name": name,
                "coefficient_count": len(model.coefficients),
                "sha256": model.model_sha256,
            }
        )

    config_details: list[dict[str, object]] = []
    for name in REQUIRED_CONFIGS:
        path = root / "config" / name
        if not path.is_file():
            errors.append(f"Configuration is missing: config/{name}")
            continue
        try:
            settings = ApplicationSettings.from_yaml(path)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"Invalid configuration config/{name}: {error}")
            continue
        prediction = settings.prediction
        for configured_path in (
            prediction.filter_model_path,
            prediction.methodology_model_directory,
        ):
            if configured_path is None:
                continue
            resolved = configured_path if configured_path.is_absolute() else root / configured_path
            if not resolved.exists():
                errors.append(
                    f"config/{name} references missing path: {configured_path}"
                )
        config_details.append(
            {
                "name": name,
                "asr_device": settings.execution.asr.device,
                "prediction_enabled": prediction.enabled,
            }
        )

    for relative_path in DOCKERFILES:
        path = root / relative_path
        if not path.is_file():
            errors.append(f"Dockerfile is missing: {relative_path}")
            continue
        if "COPY models ./models" not in path.read_text(encoding="utf-8"):
            errors.append(f"Dockerfile does not package methodology models: {relative_path}")

    if errors:
        raise RuntimeError("Repository validation failed:\n- " + "\n- ".join(errors))
    return {
        "status": "passed",
        "root": str(root),
        "configurations": config_details,
        "methodology_models": model_details,
        "documents": list(REQUIRED_DOCUMENTS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sara-validate-repository")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_repository(args.root)
    except RuntimeError as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
