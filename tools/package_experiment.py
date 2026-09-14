"""Build a self-contained, auditable delivery package for an experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("results/experiment-04")
DEFAULT_OUTPUT = Path("deliveries/experiment-04-expert-package")
REQUIRED_RECORD_ARTIFACTS = (
    "features.csv",
    "frames.csv",
    "segments.json",
    "transcript.txt",
    "transcript.json",
    "report.json",
    "manifest.json",
)
TECHNICAL_FEATURES = {
    "TIME_first_pause_s": "Длительность первой паузы до первого речевого сегмента.",
    "TIME_mean_internal_pause_s": "Средняя длительность внутренних пауз; первая и финальная паузы исключены.",
    "TIME_replica_duration_s": "Длительность реплики: от начала первого до конца последнего речевого сегмента.",
    "TIME_audio_duration_s": "Полная длительность исходной аудиозаписи.",
    "TIME_active_speech_duration_s": "Суммарная длительность речевых сегментов после VAD; паузы исключены.",
    "RATE_words_per_second_active": "Число слов расшифровки в секунду активной речи.",
    "RATE_characters_per_second_active": "Число знаков расшифровки в секунду активной речи.",
    "RATE_syllables_per_second_active": "Оценка числа русских слогов в секунду активной речи.",
    "RATE_phonemes_proxy_per_second_active": "Прокси-оценка фонем по буквам русского текста в секунду активной речи.",
}
ACOUSTIC_DESCRIPTIONS = {
    "frame_energy": "Покадровая энергия сигнала.",
    "frame_intensity_loudness": "Покадровая интенсивность/громкость.",
    "mfcc": "Мел-частотные кепстральные коэффициенты.",
    "f0_shs": "Основная частота, рассчитанная методом Subharmonic Summation.",
    "f0_acf_cepstrum": "Основная частота методом автокорреляции/кепструма.",
    "jitter": "Вариации периода основного тона.",
    "shimmer": "Вариации амплитуды голосового сигнала.",
    "formant_frequency": "Частоты формант F1-F3.",
    "formant_bandwidth": "Полосы формант F1-F3.",
    "psychoacoustic_sharpness": "Психоакустическая резкость.",
    "spectral_harmonicity": "Спектральная гармоничность / отношение гармоник к шуму.",
    "f0_harmonics_ratios": "Отношения гармоник основного тона.",
}


@dataclass(frozen=True)
class DeliveryRecord:
    """Validated input and destination locations for one experiment record."""

    record_id: str
    input_path: str
    source_directory: Path
    destination_directory: Path


class ExperimentPackageBuilder:
    """Validate, describe, and archive a complete experiment result set."""

    def __init__(self, input_directory: Path, output_directory: Path) -> None:
        self._input_directory = input_directory
        self._output_directory = output_directory
        self._registry = self._load_registry()

    def build(self) -> Path:
        records = self._load_and_validate_records()
        if self._output_directory.exists():
            raise FileExistsError(
                f"Delivery directory already exists: {self._output_directory}. "
                "Choose a new output directory to protect the existing package."
            )

        self._output_directory.mkdir(parents=True)
        self._copy_record_artifacts(records)
        feature_rows = self._write_aggregate_features(records)
        self._write_records_index(records)
        self._write_feature_dictionary(feature_rows)
        self._write_acoustic_coverage(records)
        self._write_readme(records, feature_rows)
        self._write_manifest(records)
        return self._write_zip_archive()

    def _load_and_validate_records(self) -> list[DeliveryRecord]:
        summary_path = self._input_directory / "run_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Run summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records: list[DeliveryRecord] = []
        for entry in summary.get("records", []):
            record_id = Path(str(entry["directory"])).name
            source = self._input_directory / "records" / record_id
            if not source.is_dir():
                raise FileNotFoundError(f"Record directory is missing: {source}")
            missing = [
                name for name in REQUIRED_RECORD_ARTIFACTS if not (source / name).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{record_id}: missing artifacts: {', '.join(missing)}")
            self._validate_linguistic_feature_set(source / "features.csv", record_id)
            records.append(
                DeliveryRecord(
                    record_id=record_id,
                    input_path=str(entry["input_path"]),
                    source_directory=source,
                    destination_directory=self._output_directory / "records" / record_id,
                )
            )
        if not records:
            raise ValueError("The run summary contains no records.")
        if len(records) != int(summary["record_count"]):
            raise ValueError("Record count in run_summary.json does not match its entries.")
        return records

    def _validate_linguistic_feature_set(self, features_path: Path, record_id: str) -> None:
        with features_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        codes = {row["feature_code"] for row in rows}
        expected = set(self._registry)
        missing = expected - codes
        extra = {code for code in codes if code in expected and sum(1 for row in rows if row["feature_code"] == code) != 1}
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                parts.append(f"duplicate: {', '.join(sorted(extra))}")
            raise ValueError(f"{record_id}: invalid 92-feature registry coverage ({'; '.join(parts)})")

    def _copy_record_artifacts(self, records: list[DeliveryRecord]) -> None:
        for record in records:
            record.destination_directory.mkdir(parents=True)
            for artifact_name in REQUIRED_RECORD_ARTIFACTS:
                shutil.copy2(
                    record.source_directory / artifact_name,
                    record.destination_directory / artifact_name,
                )

    def _write_aggregate_features(self, records: list[DeliveryRecord]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for record in records:
            with (record.source_directory / "features.csv").open(
                encoding="utf-8", newline=""
            ) as source:
                rows.extend(csv.DictReader(source))
        if not rows:
            raise ValueError("No scalar features were found.")
        destination = self._output_directory / "tables" / "features_all.csv"
        self._write_csv(destination, rows, list(rows[0]))
        return rows

    def _write_records_index(self, records: list[DeliveryRecord]) -> None:
        rows = [
            {
                "record_id": record.record_id,
                "input_path": record.input_path,
                "record_directory": f"records/{record.record_id}",
                "features": f"records/{record.record_id}/features.csv",
                "frames": f"records/{record.record_id}/frames.csv",
                "transcript": f"records/{record.record_id}/transcript.txt",
                "segments": f"records/{record.record_id}/segments.json",
                "report": f"records/{record.record_id}/report.json",
            }
            for record in records
        ]
        self._write_csv(
            self._output_directory / "tables" / "records_index.csv",
            rows,
            list(rows[0]),
        )

    def _write_feature_dictionary(self, feature_rows: list[dict[str, str]]) -> None:
        codes = sorted({row["feature_code"] for row in feature_rows})
        rows = []
        for code in codes:
            description, group = self._feature_description(code)
            example = next(row for row in feature_rows if row["feature_code"] == code)
            rows.append(
                {
                    "feature_code": code,
                    "group": group,
                    "description": description,
                    "unit": example["unit"],
                    "source": example["source"],
                    "method_version": example["method_version"],
                }
            )
        self._write_csv(
            self._output_directory / "tables" / "feature_dictionary.csv",
            rows,
            list(rows[0]),
        )

    def _write_acoustic_coverage(self, records: list[DeliveryRecord]) -> None:
        coverage_by_category: dict[str, set[str]] = {
            category: set() for category in ACOUSTIC_DESCRIPTIONS
        }
        for record in records:
            manifest = json.loads(
                (record.source_directory / "manifest.json").read_text(encoding="utf-8")
            )
            coverage = manifest.get("opensmile_semantic_coverage", {})
            for category, columns in coverage.items():
                coverage_by_category.setdefault(category, set()).update(columns)
        rows = [
            {
                "requested_group": category,
                "description": ACOUSTIC_DESCRIPTIONS.get(category, "See openSMILE column name."),
                "status": "delivered" if columns else "not_delivered",
                "frame_columns": "; ".join(sorted(columns)),
                "artifact": "records/<record_id>/frames.csv",
            }
            for category, columns in coverage_by_category.items()
        ]
        self._write_csv(
            self._output_directory / "tables" / "acoustic_coverage.csv",
            rows,
            list(rows[0]),
        )

    def _write_readme(
        self,
        records: list[DeliveryRecord],
        feature_rows: list[dict[str, str]],
    ) -> None:
        codes = {row["feature_code"] for row in feature_rows}
        unavailable = sum(1 for row in feature_rows if not row["value"])
        text = f"""# Эксперимент 4: пакет результатов для экспертов

## Состав поставки

Пакет содержит результаты обработки {len(records)} аудиозаписей. Исходные аудиофайлы намеренно не включены: передаются результаты извлечения признаков и расшифровки.

- `tables/features_all.csv` — все скалярные признаки, {len(feature_rows)} строк: одна строка `признак × запись`.
- `tables/feature_dictionary.csv` — словарь {len(codes)} кодов признаков, единиц и методов расчета.
- `tables/records_index.csv` — индекс записей и относительные пути к их артефактам.
- `tables/acoustic_coverage.csv` — точное покрытие запрошенных групп acoustic-признаков колонками openSMILE.
- `records/<record_id>/features.csv` — скалярные признаки конкретной записи.
- `records/<record_id>/frames.csv` — покадровые low-level descriptors openSMILE из ComParE_2016 и eGeMAPSv02.
- `records/<record_id>/transcript.txt` и `transcript.json` — текст ASR, его источник и timestamps слов.
- `records/<record_id>/segments.json` — речевые сегменты и паузы, определенные VAD.
- `records/<record_id>/report.json` — сгруппированное резюме результатов записи.
- `records/<record_id>/manifest.json` — диагностическая информация и точное соответствие запрошенных акустических групп колонкам frame-level файла.
- `delivery_manifest.json` — состав пакета и SHA-256 каждого передаваемого файла.

## Как читать `features_all.csv`

`feature_code` — устойчивый код признака; его расшифровка дана в `feature_dictionary.csv`. `value` — числовое значение; пустое значение означает «не рассчитано», а причина находится в JSON-поле `evidence`. `source`, `confidence` и `method_version` фиксируют происхождение и воспроизводимость значения.

В набор включены 92 параметра из предоставленного реестра (`STR`, `LEX`, `MOR`, `ORT`, `PUN`, `SYN`, `EMO`), а также временные характеристики пауз и реплики и оценки скорости речи за вычетом пауз. Для пауз использован согласованный порог 300 ms. Всего в поставке {len(codes)} кодов скалярных признаков.

## Важные ограничения интерпретации

Расшифровка получена `faster-whisper` и не является вручную верифицированной. Поэтому орфографические признаки `ORT_*` и признаки, которым необходим проверенный текст, могут иметь пустое `value` и источник `not-computed`; это не потеря признака, а явное обозначение недопустимости расчета по ASR-тексту. Таких пустых значений в сводной таблице: {unavailable}.

Покадровый F0 методом SHS передан. Отдельный поток F0 методом ACF/cepstrum в этом запуске не был сформирован; это явно отмечено как `not_delivered` в `tables/acoustic_coverage.csv`. Не следует подменять его другими F0-колонками.

## Рекомендуемый порядок работы

1. Откройте `tables/records_index.csv` для навигации по записям.
2. Анализируйте скалярные показатели через `tables/features_all.csv` совместно с `tables/feature_dictionary.csv`.
3. Для временного анализа используйте соответствующий `frames.csv`; значения из двух наборов openSMILE разделены полем `frame_source` и не выравниваются принудительно.
4. Проверяйте текст и паузы через `transcript.txt`, `transcript.json` и `segments.json`.
5. Перед анализом сверяйте хэши с `delivery_manifest.json`.
"""
        (self._output_directory / "README.md").write_text(text, encoding="utf-8")

    def _write_manifest(self, records: list[DeliveryRecord]) -> None:
        files = sorted(
            path
            for path in self._output_directory.rglob("*")
            if path.is_file() and path.name != "delivery_manifest.json"
        )
        manifest = {
            "package": "experiment-04-expert-package",
            "record_count": len(records),
            "source_run": str(self._input_directory),
            "records": [record.record_id for record in records],
            "files": [
                {
                    "path": path.relative_to(self._output_directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": self._sha256(path),
                }
                for path in files
            ],
        }
        (self._output_directory / "delivery_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_zip_archive(self) -> Path:
        archive_path = self._output_directory.with_suffix(".zip")
        with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(self._output_directory.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(self._output_directory.parent))
        return archive_path

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _feature_description(self, code: str) -> tuple[str, str]:
        if code in self._registry:
            return self._registry[code]["description"], self._registry[code]["group_title"]
        if code in TECHNICAL_FEATURES:
            group = "Временные признаки" if code.startswith("TIME_") else "Скорость речи"
            return TECHNICAL_FEATURES[code], group
        return "See evidence and method_version in features_all.csv.", "Технический"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _load_registry() -> dict[str, dict[str, str]]:
        registry_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "sara_audio"
            / "data"
            / "linguistic_features.json"
        )
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        return {feature["code"]: feature for feature in payload["features"]}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an auditable expert delivery package for an experiment."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    archive = ExperimentPackageBuilder(args.input, args.output).build()
    print(f"Delivery directory: {args.output}")
    print(f"ZIP archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
