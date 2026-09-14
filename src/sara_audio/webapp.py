"""Gradio service for batch audio inference on a server."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sara_audio.config import ApplicationSettings
from sara_audio.server_delivery import ServerDeliveryBuilder

SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".m4a", ".mp3", ".flac", ".ogg"})
PREVIEW_COLUMNS = (
    "file_id",
    "file_name",
    "agent_year_id",
    "burnout",
    "burnout_final",
    "burnout_label",
    "agent_sex_score",
    "agent_sex_label",
    "filter_score",
    "group_input_count",
    "group_informative_count",
)


@dataclass(frozen=True)
class WebServiceSettings:
    """Runtime settings that are intentionally separate from feature settings."""

    config_path: Path
    output_root: Path
    temp_root: Path | None = None


class SaraWebService:
    """Process uploaded audio batches through the regular extraction pipeline."""

    def __init__(self, settings: WebServiceSettings) -> None:
        self._settings = settings
        ApplicationSettings.from_yaml(settings.config_path)
        self._job_lock = threading.Lock()
        self._settings.output_root.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"sara_audio.web.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        log_path = self._settings.output_root / "sara-web.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self._logger.addHandler(handler)
        self._logger.info("Web service started with config=%s", settings.config_path.resolve())
        if self._settings.temp_root is not None:
            self._settings.temp_root.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        uploaded_files: list[Any] | tuple[Any, ...] | Any,
        make_delivery: bool = True,
    ) -> tuple[str, list[list[str]], str | None]:
        """Run one uploaded batch and return status, preview rows, and ZIP path."""
        paths = self._resolve_uploaded_paths(uploaded_files)
        if not paths:
            return "Загрузите хотя бы одну аудиозапись.", [], None

        job_id = uuid.uuid4().hex[:12]
        job_root = self._settings.output_root / f"job-{job_id}"
        result_root = job_root / "results"
        job_root.mkdir(parents=True, exist_ok=False)
        input_root = Path(
            tempfile.mkdtemp(prefix=f"sara-web-{job_id}-", dir=self._settings.temp_root)
        )
        try:
            for index, source in enumerate(paths, start=1):
                suffix = source.suffix.lower()
                if suffix not in SUPPORTED_AUDIO_SUFFIXES:
                    raise ValueError(
                        f"Неподдерживаемый формат: {source.name}. "
                        f"Разрешены: {', '.join(sorted(SUPPORTED_AUDIO_SUFFIXES))}."
                    )
                upload_directory = input_root / f"{index:04d}"
                upload_directory.mkdir()
                shutil.copy2(source, upload_directory / source.name)

            # GPU inference is serialized to avoid concurrent model runs exhausting VRAM.
            with self._job_lock:
                summary = self._run_worker(input_root, result_root, job_root)

            delivery_archive: Path | None = None
            if make_delivery:
                delivery_archive = ServerDeliveryBuilder(
                    result_root,
                    job_root / "delivery",
                ).build()

            archive = self._zip_job(job_root, job_root / "sara-batch.zip")
            preview = self._read_preview(result_root / "predictions.csv")
            delivery_note = "; delivery CSV/XLSX добавлен" if delivery_archive else ""
            prediction_note = (
                f"; итоговых прогнозов: {len(preview)}"
                if preview
                else "; все записи исключены фильтром, итоговый прогноз не рассчитан"
            )
            status = (
                f"Готово: обработано {summary['record_count']} файл(ов)"
                f"{prediction_note}{delivery_note}. "
                f"Результаты сохранены в `{result_root}`."
            )
            return status, preview, str(archive)
        except Exception as error:
            error_text = traceback.format_exc()
            self._logger.error(
                "Job %s failed; partial results kept at %s\n%s",
                job_id,
                job_root,
                error_text,
            )
            failure_path = job_root / "error.txt"
            failure_path.write_text(
                "SARA web processing failed\n"
                f"job_id: {job_id}\n"
                f"config: {self._settings.config_path.resolve()}\n"
                f"error: {error!r}\n\n"
                f"{error_text}",
                encoding="utf-8",
            )
            return (
                f"Ошибка обработки: {error}. Подробности сохранены в `{failure_path}`.",
                [],
                None,
            )
        finally:
            shutil.rmtree(input_root, ignore_errors=True)

    def _run_worker(self, input_root: Path, result_root: Path, job_root: Path) -> dict[str, Any]:
        """Run native extraction separately so a hard crash cannot kill Gradio."""
        log_path = job_root / "processing.log"
        project_root = self._settings.config_path.resolve().parent.parent
        command = [
            sys.executable,
            "-u",
            "-X",
            "faulthandler",
            "-m",
            "sara_audio.web_worker",
            "--config",
            str(self._settings.config_path.resolve()),
            "--input",
            str(input_root),
            "--output",
            str(result_root),
        ]
        with log_path.open("w", encoding="utf-8", newline="") as log_file:
            log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log_file.flush()
            completed = subprocess.run(
                command,
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                "Процесс обработки аварийно завершился "
                f"(код {completed.returncode}). Лог: {log_path}"
            )

        summary_path = result_root / "run_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(
                "Процесс завершился без итогового файла. "
                f"Лог: {log_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not isinstance(summary.get("record_count"), int):
            raise RuntimeError(f"Некорректный итоговый файл: {summary_path}")
        return summary

    @staticmethod
    def _resolve_uploaded_paths(uploaded_files: Any) -> list[Path]:
        if uploaded_files is None:
            return []
        values = uploaded_files if isinstance(uploaded_files, (list, tuple)) else [uploaded_files]
        paths: list[Path] = []
        for value in values:
            candidate = value
            if isinstance(value, dict):
                candidate = value.get("path") or value.get("name")
            elif hasattr(value, "path"):
                candidate = value.path
            elif hasattr(value, "name") and not isinstance(value, (str, Path)):
                candidate = value.name
            if not candidate:
                continue
            path = Path(str(candidate))
            if not path.is_file():
                raise FileNotFoundError(f"Загруженный файл не найден: {path}")
            paths.append(path)
        return paths

    @staticmethod
    def _read_preview(path: Path | None) -> list[list[str]]:
        if path is None or not path.is_file():
            return []
        with path.open(encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter=";")
            return [
                [str(row.get(column) or "") for column in PREVIEW_COLUMNS]
                for row in reader
            ]

    @staticmethod
    def _zip_job(job_root: Path, archive: Path) -> Path:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(job_root.rglob("*")):
                if not path.is_file() or path == archive:
                    continue
                bundle.write(path, path.relative_to(job_root).as_posix())
        return archive


def create_app(service: SaraWebService) -> Any:
    """Build the Gradio Blocks app without importing Gradio during CLI/tests import."""
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "The web service requires Gradio. Install the web extra: pip install -e .[web]."
        ) from error

    with gr.Blocks(title="SARA audio batch inference") as app:
        gr.Markdown("# SARA: пакетная обработка аудио")
        uploads = gr.File(
            label="Аудиозаписи",
            file_count="multiple",
            file_types=["audio"],
            type="filepath",
        )
        make_delivery = gr.Checkbox(
            label="Добавить компактный delivery-пакет CSV/XLSX",
            value=True,
        )
        run_button = gr.Button("Запустить обработку", variant="primary")
        status = gr.Markdown()
        preview = gr.Dataframe(
            headers=list(PREVIEW_COLUMNS),
            datatype=["str"] * len(PREVIEW_COLUMNS),
            label="Итоговый прогноз выгорания",
            interactive=False,
            wrap=True,
        )
        download = gr.File(label="Скачать ZIP результатов")
        run_button.click(
            fn=service.process,
            inputs=[uploads, make_delivery],
            outputs=[status, preview, download],
        )
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sara-web")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SARA_CONFIG_PATH", "/app/config/rtx4090_cuda12.yaml")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SARA_WEB_OUTPUT_ROOT", "/data/output/web-runs")),
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        default=(
            Path(os.environ["SARA_WEB_TEMP_ROOT"])
            if os.environ.get("SARA_WEB_TEMP_ROOT")
            else None
        ),
    )
    parser.add_argument("--host", default=os.environ.get("SARA_WEB_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SARA_WEB_PORT", "7860")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = SaraWebService(
        WebServiceSettings(
            config_path=args.config,
            output_root=args.output_root,
            temp_root=args.temp_root,
        )
    )
    app = create_app(service)
    app.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
        allowed_paths=[str(args.output_root.resolve())],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
