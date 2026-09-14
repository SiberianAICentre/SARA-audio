"""Package existing-rule corpus results with explicit numerical coverage."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from sara_audio.domain import Transcript
from sara_audio.external_delivery import (
    RATE_CODES,
    TIME_CODES,
    file_hash,
    read_json,
    write_json,
)
from sara_audio.feature_filter import filter_feature_rows, load_excluded_feature_codes
from sara_audio.linguistic import LinguisticFeatureExtractor
from sara_audio.registry import LinguisticFeatureRegistry

TECHNICAL = {
    "TIME_first_pause_s": "Пауза до первого VAD-сегмента в предоставленном WAV; не исходного звонка. Прежний метод возвращает null при отсутствии начальной паузы или речевых сегментов.",
    "TIME_mean_internal_pause_s": "Среднее внутренних пауз предоставленного WAV; первая и конечная исключены. При отсутствии внутренних пауз значение null.",
    "TIME_replica_duration_s": "От начала первого до конца последнего VAD-сегмента предоставленного WAV.",
    "TIME_audio_duration_s": "Полная длительность предоставленного WAV.",
    "TIME_active_speech_duration_s": "Суммарная длительность VAD-сегментов предоставленного WAV.",
    "RATE_words_per_second_active": "Слова в секунду активной речи; здесь не рассчитано из-за неподтвержденного соответствия текста и WAV.",
    "RATE_chars_per_second_active": "Буквенные знаки в секунду активной речи; здесь не рассчитано.",
    "RATE_syllables_per_second_active": "Оценка слогов по гласным в секунду активной речи; здесь не рассчитано.",
    "RATE_phonemes_per_second_active": "Фонемы в секунду активной речи; G2P не реализован.",
}


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"No schema for empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ExternalPackageWriter:
    """Verify records, derive tables from actual values, then publish a ZIP."""

    def __init__(self, root: Path, config: Path, inventory: Path, source: Path) -> None:
        self.root, self.config, self.inventory, self.source = (
            root,
            config,
            inventory,
            source,
        )
        self.registry = LinguisticFeatureRegistry.load_default()

    def build(self) -> None:
        summary = read_json(self.root / "run_summary.json")
        if not summary["complete"] or summary["errors"]:
            raise ValueError("Cannot publish incomplete delivery")
        tables = self.root / "tables"
        tables.mkdir()
        all_rows, numeric, wide, coverage, indexes, acoustics = [], [], [], [], [], []
        definitions = {d.code: d for d in self.registry.definitions}
        computed_codes = [
            f.code
            for f in LinguisticFeatureExtractor(self.registry).extract(Transcript(""))
            if f.value is not None
        ]
        excluded_feature_codes = load_excluded_feature_codes()
        published_computed_codes = [
            code for code in computed_codes if code not in excluded_feature_codes
        ]
        if len(computed_codes) != 92:
            raise ValueError("The heuristic profile must expose all 92 registry methods")
        expected = set(definitions) | set(TIME_CODES) | set(RATE_CODES)
        column_records = defaultdict(set)
        config = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        expect_frames = config["include_frames"] and config.get("opensmile", {}).get(
            "enabled", True
        )
        for record in summary["records"]:
            record_id = record["record_id"]
            folder = self.root / record["directory"]
            frames_available = (folder / "frames.csv").is_file()
            if expect_frames and record["audio_available"] and not frames_available:
                raise ValueError(f"Missing frame artifact: {record_id}")
            with (folder / "features.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 101 or {r["feature_code"] for r in rows} != expected:
                raise ValueError(f"Incomplete scalar feature set: {record_id}")
            if any(r["file_id"] != record_id for r in rows):
                raise ValueError(f"Wrong record identity: {record_id}")
            calculated = [
                r for r in rows if r["feature_code"] in definitions and r["value"] != ""
            ]
            text_meta = read_json(folder / "transcript.json")
            has_operator_text = bool(text_meta["text"].strip())
            expected_computed = set(computed_codes) if has_operator_text else set()
            if {r["feature_code"] for r in calculated} != expected_computed or any(
                r["value"] != "" for r in rows if r["feature_code"] in RATE_CODES
            ):
                raise ValueError(
                    f"Unexpected numerical coverage or unsafe rates: {record_id}"
                )
            if text_meta["is_verified"] or text_meta["tokens"]:
                raise ValueError(
                    f"Invalid provenance or invented alignment: {record_id}"
                )
            if (folder / "transcript.txt").read_text(
                encoding="utf-8"
            ).rstrip() != text_meta["text"].rstrip():
                raise ValueError(f"Transcript mismatch: {record_id}")
            manifest = read_json(folder / "manifest.json")
            if manifest["record_id"] != record_id:
                raise ValueError(f"Manifest identity mismatch: {record_id}")
            for variant, name in (
                ("processed", "dialogue_processed.txt"),
                ("before_roles", "dialogue_before_roles.txt"),
            ):
                expected_hash = manifest["transcript_sources"][variant]["sha256"]
                if file_hash(folder / name) != expected_hash:
                    raise ValueError(
                        f"Source transcript copy mismatch: {record_id}/{name}"
                    )
            all_rows.extend(rows)
            numeric.extend(calculated)
            wide.append(
                {
                    "record_id": record_id,
                    "split": record["split"],
                    "analysis_scope": "operator",
                    "analysis_text_status": "available"
                    if has_operator_text
                    else "no_operator_text",
                    **{
                        code: next(
                            (
                                r["value"]
                                for r in calculated
                                if r["feature_code"] == code
                            ),
                            "",
                        )
                        for code in published_computed_codes
                    },
                }
            )
            indexes.append(
                {
                    **record,
                    "analysis_text_status": "available"
                    if has_operator_text
                    else "no_operator_text",
                    "audio_source": manifest["audio_source"],
                    "audio_sha256": manifest["audio_sha256"],
                    "processed_transcript_source": manifest["transcript_sources"][
                        "processed"
                    ]["path"],
                    "expert_sheet_row": manifest["expert_sheet_row"],
                    "transcript": f"{record['directory']}/transcript.txt",
                    "features": f"{record['directory']}/features.csv",
                    "frames": f"{record['directory']}/frames.csv"
                    if frames_available
                    else "",
                    "report": f"{record['directory']}/report.json",
                }
            )
            for r in rows:
                coverage.append(
                    {
                        "record_id": record_id,
                        "feature_code": r["feature_code"],
                        "status": "computed" if r["value"] != "" else "not-computed",
                        "reason": json.loads(r["evidence"]).get("reason", ""),
                    }
                )
            if frames_available:
                with (folder / "frames.csv").open(
                    encoding="utf-8", newline=""
                ) as stream:
                    reader = csv.reader(stream)
                    header = next(reader)
                    frame_count = 0
                    source_index = header.index("frame_source")
                    file_index = header.index("file_id")
                    counts = Counter()
                    for frame in reader:
                        if len(frame) != len(header) or frame[file_index] != record_id:
                            raise ValueError(
                                f"Frame CSV schema/identity mismatch: {record_id}"
                            )
                        frame_count += 1
                        counts[frame[source_index]] += 1
                    if frame_count == 0:
                        raise ValueError(f"Empty frame artifact: {record_id}")
                indexes[-1]["frame_rows"] = frame_count
                indexes[-1]["frame_sources"] = json.dumps(counts)
                for column in header:
                    column_records[column].add(record_id)
            else:
                indexes[-1]["frame_rows"] = 0
                indexes[-1]["frame_sources"] = "{}"
            for group, columns in (
                manifest["opensmile_semantic_coverage"] or {}
            ).items():
                acoustics.append(
                    {
                        "record_id": record_id,
                        "requested_group": group,
                        "status": "columns-present" if columns else "not-delivered",
                        "columns": "; ".join(columns),
                    }
                )
        if len(indexes) != 390 or len({r["record_id"] for r in indexes}) != 390:
            raise ValueError("Expected 390 unique records")
        if sum(r["audio_available"] for r in indexes) != 385:
            raise ValueError("Expected 385 audio records and 5 text-only records")
        published_all_rows = filter_feature_rows(all_rows, excluded_feature_codes)
        published_numeric = filter_feature_rows(numeric, excluded_feature_codes)
        published_coverage = filter_feature_rows(coverage, excluded_feature_codes)
        published_expected = sorted(expected - excluded_feature_codes)
        write_csv(tables / "features_all.csv", published_all_rows)
        write_csv(tables / "linguistic_computed_92.csv", published_numeric)
        write_csv(tables / "linguistic_computed_92_wide.csv", wide)
        write_csv(tables / "records_index.csv", indexes)
        write_csv(
            tables / "no_operator_text_records.csv",
            [r for r in indexes if r["analysis_text_status"] == "no_operator_text"],
            list(indexes[0]),
        )
        write_csv(tables / "feature_status_by_record.csv", published_coverage)
        write_csv(
            tables / "acoustic_coverage.csv",
            acoustics,
            ["record_id", "requested_group", "status", "columns"],
        )
        write_csv(
            tables / "frame_columns.csv",
            [
                {"column": c, "records_present": len(ids)}
                for c, ids in column_records.items()
            ],
            ["column", "records_present"],
        )
        counts = defaultdict(Counter)
        reasons = defaultdict(set)
        for row in published_coverage:
            counts[row["feature_code"]][row["status"]] += 1
            if row["reason"]:
                reasons[row["feature_code"]].add(row["reason"])
        dictionary = []
        for code in published_expected:
            sample = next(r for r in published_all_rows if r["feature_code"] == code)
            dictionary.append(
                {
                    "feature_code": code,
                    "description": definitions[code].description
                    if code in definitions
                    else TECHNICAL[code],
                    "unit": sample["unit"],
                    "method_version": sample["method_version"],
                    "computed_records": counts[code]["computed"],
                    "not_computed_records": counts[code]["not-computed"],
                    "reasons": "; ".join(sorted(reasons[code])),
                }
            )
        write_csv(tables / "feature_dictionary.csv", dictionary)
        write_csv(
            tables / "text_only_records.csv",
            [r for r in indexes if not r["audio_available"]],
        )
        references = self.root / "reference"
        references.mkdir()
        shutil.copy2(self.config, references / "config.yaml")
        shutil.copy2(self.inventory, references / "input_inventory.json")
        from importlib.resources import files

        (references / "linguistic_registry.json").write_bytes(
            files("sara_audio.data").joinpath("linguistic_features.json").read_bytes()
        )
        external_files = [
            "2_file/df_full_standard.csv",
            "2_file/df_train_standard.csv",
            "2_file/df_test_standard.csv",
            "Анализ_транскрипций_НМ.xlsx",
        ]
        external_files.extend(
            p.relative_to(self.source).as_posix()
            for p in (self.source / "Обработанные_записи (1)/_Проверка").iterdir()
            if p.is_file()
        )
        for relative in external_files:
            target = references / "external" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.source / relative, target)
        versions = {
            name: importlib.metadata.version(name)
            for name in ("sara-audio", "opensmile", "pandas", "numpy", "PyYAML")
        }
        code_source = Path(__file__).parent
        code_root = references / "code_snapshot"
        code_root.mkdir()
        for file in sorted(code_source.rglob("*.py")):
            target = code_root / file.relative_to(code_source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
        shutil.copy2(
            references / "linguistic_registry.json",
            code_root / "data/linguistic_features.json",
        )
        validation = {
            "records": len(indexes),
            "audio_records": sum(r["audio_available"] for r in indexes),
            "text_only_records": sum(not r["audio_available"] for r in indexes),
            "scalar_rows": len(published_all_rows),
            "calculated_linguistic_rows": len(published_numeric),
            "linguistic_records": sum(r["linguistic_computed"] == 92 for r in indexes),
            "no_operator_text_records": sum(
                r["linguistic_computed"] == 0 for r in indexes
            ),
            "registered_linguistic_codes": 92,
            "calculated_linguistic_codes": 92,
            "unimplemented_linguistic_codes": 0,
            "frame_rows": sum(r["frame_rows"] for r in indexes),
            "frame_records": sum(bool(r["frames"]) for r in indexes),
            "rates_calculated": 0,
            "excluded_features": sorted(excluded_feature_codes),
            "versions": versions,
            "source_asr_executed": False,
        }
        write_json(self.root / "validation.json", validation)
        self.write_readme(validation)
        manifest = {
            "package": self.root.name,
            "validation": validation,
            "files": [
                {
                    "path": p.relative_to(self.root).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in sorted(self.root.rglob("*"))
                if p.is_file()
            ],
        }
        write_json(self.root / "delivery_manifest.json", manifest)
        archive_path = self.root.with_suffix(".zip")
        print("Tables validated. Building ZIP...", flush=True)
        with zipfile.ZipFile(
            archive_path, "x", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(self.root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        f"{self.root.name}/{path.relative_to(self.root).as_posix()}",
                    )
        print("Checking ZIP CRC...", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed")
        archive_path.with_suffix(".zip.sha256").write_text(
            f"{file_hash(archive_path)}  {archive_path.name}\n", encoding="ascii"
        )

    def write_readme(self, validation: dict) -> None:
        text = f"""# Реальные данные: 92 эвристических текстовых признака

## Что передается

390 записей: 385 с аудио и 5 без аудио. Для {validation["linguistic_records"]} записей рассчитаны все 92 эвристических текстовых признака. В {validation["no_operator_text_records"]} расшифровках нет реплик оператора: все 92 текстовых значения оставлены пустыми с причиной; клиентский текст не подставлялся. Итог: {validation["calculated_linguistic_rows"]} численных лингвистических значений и {validation["scalar_rows"]} строк полного скалярного реестра.

Новый Whisper не запускался. Источник текста: `Обработанные_записи (1)`, последняя документированная версия после назначения ролей и замены отдельных артефактов. Размер и версия исходной модели Whisper неизвестны. Это не вручную проверенная расшифровка.

## С чего начать

1. `tables/linguistic_computed_92_wide.csv`: основная удобная таблица, одна запись на строку, 92 признака по колонкам, ID и train/test рядом.
2. `tables/feature_dictionary.csv`: точные определения, единицы, количество рассчитанных и недоступных значений. Определения перенесены программно из исходного реестра.
3. `tables/records_index.csv`: список всех записей и пути к подробным результатам.
4. `records/<ID>/transcript.txt`: точный текст для расчета; пустой, если реплик оператора нет. Полные диалоги сохранены отдельно в любом случае.

Сводные CSV записаны в UTF-8 с BOM, разделитель — запятая, десятичный знак — точка. Если Excel открывает всю строку в одной колонке, используйте импорт «Из текста/CSV» с этими настройками. Поле `evidence` содержит JSON с диагностикой; для первичного просмотра удобнее широкая таблица без этого поля.

## Как подготовлен текст

Выбраны только строки с ролью «Оператор». Метки `[mm:ss]` и названия ролей исключены из счетчиков. Маркеры «Тишина» исключены только в строках, для которых есть зарегистрированная замена в журнале. Для частичной замены сохраняется оставшаяся часть реплики. Остальные слова, повторы, ошибки и пунктуация сохранены; строки соединены переводом строки. Все 92 признака рассчитаны переносимыми эвристиками с provenance и confidence.

Назначение ролей выполнено внешним обработчиком по содержанию текста. Смешанные и спорные реплики не исправлялись автоматически. Их перечень находится в `reference/external/.../_Проверка`. Эти ограничения действуют и на полученные счетчики.

## Файлы каждой записи

| Файл | Содержание |
|---|---|
| `features.csv`, `features.json` | 101 скалярный код: 92 лингвистических, 5 временных, 4 скорости; недоступные значения не заменены нулями |
| `frames.csv` | Покадровые ComParE_2016 и eGeMAPSv02 для предоставленного WAV; у пяти text-only записей файла нет |
| `transcript.txt`, `transcript.json` | Выбранный операторский текст, источник и `is_verified=false`; пословные timestamps не придумывались |
| `dialogue_processed.txt` | Полный диалог последней обработанной версии, побайтовая копия |
| `dialogue_before_roles.txt` | Полный диалог до документированного назначения ролей, побайтовая копия |
| `transcript_turns.json` | Все строки с исходной секундной отметкой, ролью и флагом включения в анализ |
| `segments.json` | VAD-сегменты и паузы в координатах предоставленного WAV; не координаты текстового диалога |
| `report.json` | Значения, определения, покрытие, происхождение и ограничения |
| `manifest.json` | ID, split, SHA-256 источников, наличие аудио, происхождение текста |

## Как читать значения

`value=0` означает нулевой результат реализованного счетчика. Пустая ячейка / JSON `null` означает отсутствие численного значения. Для нереализованных признаков, неподтвержденной скорости и отсутствующего аудио причина находится в `evidence.reason`. `feature_code` связан со словарем, `source` указывает обработчик, `method_version` его версию. Значения `confidence` унаследованы от experiment-04 и не являются измеренной вероятностью правильности.

Отдельная особенность прежнего VAD: `TIME_first_pause_s=null`, если начальная пауза отсутствует или речевые сегменты не найдены; `TIME_mean_internal_pause_s=null`, если внутренних пауз нет. В этих случаях причина определяется по `segments.json` и числу сегментов/пауз в evidence. Это поведение оставлено совместимым с experiment-04. В сводке статусов `not-computed` означает отсутствие численного значения, а не обязательно отсутствие запущенного обработчика.

Все 92 признака рассчитаны переносимыми эвристиками. Семантические, синтаксические и орфографические оценки имеют пониженную confidence и требуют калибровки по экспертной разметке. Пунктуационные результаты отражают знаки выбранной расшифровки.

`not-computed` сохраняется только для записей без операторского текста или для показателей, чей вход отсутствует. Все 92 кода реестра имеют численную эвристическую реализацию.

## Аудио, паузы и скорости

Все поставленные WAV имеют 16 kHz, mono, PCM 16 bit. По аудиту они короче временной шкалы соответствующих полных диалогов. Вероятно, это подготовленные фрагменты; карта обрезки и полный исходный звонок отсутствуют. Состав говорящих в WAV не верифицирован.

Паузы вычислены energy-VAD с порогом 300 ms по имеющимся WAV. `TIME_first_pause_s` — до первого сегмента этого файла, `TIME_mean_internal_pause_s` — среднее между сегментами без начальной и конечной пауз, `TIME_replica_duration_s` — от первого до последнего сегмента. Эти величины нельзя трактовать как паузы полного исходного звонка или проверенные паузы только оператора.

Порог 300 ms применяется к внутренним промежуткам: более короткие промежутки объединяются с речевыми сегментами и входят в их суммарную длительность. Начальная и конечная паузы могут быть короче порога.

Скорости слов, знаков и слогов в этом пакете оставлены `not-computed`: операторский текст полного разговора и доступный WAV не имеют подтвержденного соответствия. Подсчет фонем также не реализован. Для пяти записей без аудио все TIME/RATE недоступны; текстовые счетчики рассчитаны только при наличии операторского текста. Наличие аудио и наличие операторского текста — независимые признаки доступности данных.

openSMILE передан без смены конфигураций относительно experiment-04. Отдельного F0 через ACF/cepstrum нет. `tables/acoustic_coverage.csv` показывает результат прежнего сопоставления имен колонок, а не валидацию физиологического смысла каждого измерения. В частности, HNR, спектральная гармоничность и отношения гармоник не следует считать взаимозаменяемыми. Наличие колонки не гарантирует пригодность каждого кадра. Тишина, стыки фрагментов и очень короткие файлы требуют проверки при анализе.

`frames.csv`: `start/end` — интервалы кадра вида `0 days 00:00:...`, `frame_source` — используемый набор. Потоки ComParE и eGeMAPS записаны отдельными строками; пустые поля другого потока ожидаемы и не означают потерю колонки. `file` — имя WAV, связь с оригинальным путем дана в manifest.

## Сводки и внешние материалы

- `features_all.csv`: все 101 кода × 390 записей.
- `linguistic_computed_92.csv`: все 92 рассчитанных текстовых кода, длинная таблица с evidence.
- `feature_status_by_record.csv`: статус каждого кода каждой записи.
- `text_only_records.csv`: пять записей без аудио.
- `no_operator_text_records.csv`: записи без текста оператора; они также включены в общие таблицы с пустыми текстовыми значениями.
- `acoustic_coverage.csv`, `frame_columns.csv`: колонки и покрытие акустики.
- `reference/external`: исходные df-таблицы, анализ НМ и журналы обработки. Их показатели НЕ пересчитаны и не подмешаны к нашим значениям. `df_full.text` отличается по происхождению от выбранного здесь текста. Результаты НМ связаны с ним, а не автоматически с нашим analysis_text.
- `reference/config.yaml`, `reference/code_snapshot`, `reference/input_inventory.json`, `reference/linguistic_registry.json`: настройки, код, карта источников и все 92 определения.

Train/test сохранены из исходной поставки: 326/59, операторы между ними не пересекаются. Пять text-only записей отмечены отдельно. Исходные WAV, кэш и окружение Python в архив не включены.

Полные расшифровки и внешние материалы не обезличивались. Перед передачей убедитесь, что выбранным экспертам можно предоставлять содержащиеся в них сведения.

## Проверка и воспроизведение

`validation.json` содержит фактические количества и версии библиотек. `delivery_manifest.json` содержит SHA-256 каждого файла пакета, кроме самого манифеста. Рядом с ZIP находится SHA-256 архива. Проверены 390 уникальных ID, все 92 численных текстовых значения при наличии операторского текста и ни одного без него, отсутствие ошибочно вычисленных скоростей и структурная корректность всех строк frames.csv. Эти проверки не заменяют экспертную оценку качества признаков.

Из корня проекта: `.venv\\Scripts\\python.exe -m sara_audio.external_delivery data/new_raw_data <новая_папка_delivery>`. Все настройки находятся в `config/external_delivery.yaml`; дополнительные технические аргументы не требуются. Повторный запуск требует нового выходного каталога.
"""
        (self.root / "README.md").write_text(text, encoding="utf-8")
        (self.root / "README.txt").write_text(text, encoding="utf-8-sig")
