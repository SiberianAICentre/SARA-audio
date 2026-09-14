"""Generate the root feature extraction plan from the source DOCX.

The linguistic feature list is intentionally transferred from the DOCX by code.
If the expected 92-feature structure changes, generation fails instead of
silently producing an incomplete plan.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
EXPECTED_COUNTS = {
    "STR": 16,
    "LEX": 20,
    "MOR": 14,
    "ORT": 8,
    "PUN": 7,
    "SYN": 16,
    "EMO": 11,
}
FEATURE_CODE_PATTERN = re.compile(r"^(STR|LEX|MOR|ORT|PUN|SYN|EMO)_[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class LinguisticFeature:
    group_title: str
    code: str
    description: str

    @property
    def prefix(self) -> str:
        return self.code.split("_", maxsplit=1)[0]


class DocxLinguisticFeatureParser:
    """Extract the 92 linguistic features from Word tables."""

    def parse(self, docx_path: Path) -> list[LinguisticFeature]:
        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX not found: {docx_path}")

        with zipfile.ZipFile(docx_path) as archive:
            document_xml = archive.read("word/document.xml")

        root = ET.fromstring(document_xml)
        tables = root.findall(".//w:tbl", WORD_NS)
        if len(tables) != len(EXPECTED_COUNTS):
            raise ValueError(
                f"Expected {len(EXPECTED_COUNTS)} tables, found {len(tables)}"
            )

        features: list[LinguisticFeature] = []
        for table in tables:
            rows = table.findall("./w:tr", WORD_NS)
            if not rows:
                raise ValueError("Found an empty table")

            group_title = self._row_text(rows[0])
            if not group_title:
                raise ValueError("Found a table without a group title")

            for row in rows[1:]:
                cells = [self._cell_text(cell) for cell in row.findall("./w:tc", WORD_NS)]
                if len(cells) != 2:
                    raise ValueError(
                        f"Expected 2 cells in feature row, got {len(cells)}: {cells}"
                    )

                code, description = cells
                if not FEATURE_CODE_PATTERN.match(code):
                    raise ValueError(f"Unexpected feature code: {code!r}")
                if not description:
                    raise ValueError(f"Feature {code} has an empty description")

                features.append(
                    LinguisticFeature(
                        group_title=group_title,
                        code=code,
                        description=description,
                    )
                )

        self._validate_counts(features)
        return features

    def _validate_counts(self, features: list[LinguisticFeature]) -> None:
        seen_codes: set[str] = set()
        actual_counts = dict.fromkeys(EXPECTED_COUNTS, 0)

        for feature in features:
            if feature.code in seen_codes:
                raise ValueError(f"Duplicate feature code: {feature.code}")
            seen_codes.add(feature.code)
            actual_counts[feature.prefix] += 1

        total_expected = sum(EXPECTED_COUNTS.values())
        if len(features) != total_expected:
            raise ValueError(f"Expected {total_expected} features, got {len(features)}")

        if actual_counts != EXPECTED_COUNTS:
            raise ValueError(
                f"Expected counts {EXPECTED_COUNTS}, got {actual_counts}"
            )

    def _row_text(self, row: ET.Element) -> str:
        return " ".join(
            cell_text
            for cell_text in (self._cell_text(cell) for cell in row.findall("./w:tc", WORD_NS))
            if cell_text
        ).strip()

    def _cell_text(self, cell: ET.Element) -> str:
        texts = [node.text or "" for node in cell.findall(".//w:t", WORD_NS)]
        return "".join(texts).strip()


class FeaturePlanMarkdownRenderer:
    """Render implementation plan and the generated feature registry."""

    def render(self, features: list[LinguisticFeature], source_docx: Path) -> str:
        groups: dict[str, list[LinguisticFeature]] = {}
        for feature in features:
            groups.setdefault(feature.group_title, []).append(feature)

        feature_tables = "\n\n".join(
            self._render_feature_group(title, group_features)
            for title, group_features in groups.items()
        )

        counts = ", ".join(
            f"{prefix}={count}" for prefix, count in EXPECTED_COUNTS.items()
        )

        plan_body = textwrap.dedent(
            f"""\
            # План реализации извлечения аудиопризнаков и 92 лингвистических параметров

            Источник списка лингвистических параметров: `{source_docx}`.

            Этот файл сгенерирован скриптом `tools/generate_feature_plan.py`.
            Параметры из документа не переписывались вручную. Генерация падает,
            если счетчик признаков не совпадает с ожидаемым: {counts}, всего 92.

            ## Принятые решения

            - Порог паузы по умолчанию: `300 ms`.
            - Первая пауза: интервал от начала аудиофайла до первого речевого сегмента.
            - Средняя пауза в реплике: среднее по внутренним паузам между речевыми сегментами, без первой паузы и без хвостовой тишины.
            - Длительность реплики: интервал от начала первого речевого сегмента до конца последнего речевого сегмента.
            - Скорость речи считается по активной речи: из длительности вычитаются внутренние паузы и первая пауза.
            - Орфографические признаки `ORT_*` и часть пунктуационных признаков `PUN_*` достоверны только при наличии ручной или внешне подтвержденной текстовой расшифровки. Для чистого ASR они должны иметь отдельный флаг низкой надежности.

            ## Целевой результат

            Нужно построить промышленный пайплайн, который для набора аудиофайлов извлекает:

            - openSMILE-признаки: frame energy, frame intensity/loudness, MFCC, F0 ACF/Cepstrum, F0 SHS, jitter, shimmer, formants and bandwidths, psychoacoustic sharpness, spectral harmonicity, F0 harmonics ratios.
            - Временные признаки: первая пауза, средняя внутренняя пауза, длительность реплики, полная длительность аудио.
            - Оценку скорости речи без пауз: слова/сек, символы/сек, слоги/сек, опционально фонемы/сек.
            - 92 лингвистических признака из документа, с сохранением кода, описания, метода расчета и уверенности.

            ## Архитектура

            Проект организуем как Python-пакет `sara_audio` внутри `src/`.
            Ключевой принцип: каждый тип признаков реализован отдельным классом
            за общим интерфейсом, а общий пайплайн только координирует шаги.

            Базовые компоненты:

            - `AudioPreprocessor`: конвертация `.m4a` в mono WAV, нормализация sample rate, проверка длительности, тишины и клиппинга.
            - `FeatureExtractor`: общий интерфейс `extract(input, context) -> FeatureSet`.
            - `OpenSmileExtractor`: акустические признаки openSMILE.
            - `VadPauseExtractor`: речевые сегменты, первая пауза, внутренние паузы, длительность реплики.
            - `AsrTranscriber`: транскрипция с timestamps.
            - `SpeechRateExtractor`: скорость речи по словам, символам, слогам и, при наличии G2P, фонемам.
            - `LinguisticFeatureRegistry`: реестр 92 признаков, с кодами из документа и метаданными.
            - `LinguisticFeatureExtractor`: расчет признаков по тексту и NLP-аннотациям.
            - `FeaturePipeline`: последовательный запуск шагов, сбор результата и запись артефактов.

            ## Формат данных

            Выходные данные стоит разделить на три уровня:

            - `features.parquet` или `features.csv`: одна строка на аудиофайл, агрегированные признаки.
            - `frames.parquet`: frame-level openSMILE-признаки с временной осью.
            - `segments.jsonl`: речевые сегменты, паузы, ASR-токены, confidence и диагностическая информация.

            Для каждого признака сохраняем:

            - `file_id`
            - `feature_code`
            - `value`
            - `unit`
            - `source`
            - `confidence`
            - `method_version`

            ## Этапы реализации

            1. Создать структуру проекта: `src/sara_audio`, `tests`, `config`, `artifacts`.
            2. Добавить `pyproject.toml` с зависимостями, линтерами, форматированием и тестовым запуском.
            3. Реализовать доменные модели: `AudioFile`, `SpeechSegment`, `PauseSegment`, `FeatureValue`, `ExtractionResult`.
            4. Реализовать `AudioPreprocessor` и стабильную конвертацию `.m4a` в WAV через `ffmpeg`/`pydub`.
            5. Реализовать `OpenSmileExtractor`:
               - стандартные feature sets через Python API `opensmile`;
               - custom-конфиги или отдельные вызовы `SMILExtract` там, где стандартные наборы не дают нужный LLD;
               - отдельная запись frame-level и агрегированных признаков.
            6. Реализовать `VadPauseExtractor`:
               - VAD с настраиваемым threshold;
               - слияние коротких речевых фрагментов;
               - отбрасывание шумовых микропауз;
               - расчет `first_pause_s`, `mean_internal_pause_s`, `replica_duration_s`, `audio_duration_s`.
            7. Реализовать `AsrTranscriber`:
               - транскрипция русского аудио;
               - word timestamps;
               - хранение текста, токенов и confidence.
            8. Реализовать `SpeechRateExtractor`:
               - `words_per_second_active`;
               - `chars_per_second_active`;
               - `syllables_per_second_active` через подсчет русских гласных;
               - `phonemes_per_second_active` как опциональный G2P-модуль.
            9. Реализовать `LinguisticFeatureRegistry`:
               - загрузка 92 признаков из машинно проверяемого реестра;
               - проверка уникальности кодов;
               - проверка общего количества `92`;
               - проверка счетчиков по группам.
            10. Реализовать `LinguisticFeatureExtractor` по слоям:
                - rule-based признаки для лексики, союзов, частиц, пунктуации и простых счетчиков;
                - морфология и синтаксис через NLP-пайплайн для русского;
                - классификационные признаки для связности, аргументации, эмоциональности и семантических ошибок;
                - обязательные `confidence` и `evidence` для признаков, которые нельзя надежно вывести простыми правилами.
            11. Добавить CLI:
                - `sara-extract data/raw --out artifacts/features.parquet --config config/default.yaml`;
                - `sara-validate-registry`;
                - `sara-inspect-audio path/to/file.m4a`.
            12. Добавить тесты:
                - unit-тесты для пауз и длительностей;
                - golden-тесты на синтетическом аудио с известными паузами;
                - snapshot-тест списка openSMILE-колонок;
                - тест, что все 92 признака присутствуют;
                - интеграционный тест на одном файле из `data/raw`.
            13. Добавить документацию:
                - установка зависимостей;
                - запуск извлечения;
                - описание выходных колонок;
                - ограничения ASR для `ORT_*` и `PUN_*`.

            ## Реестр 92 лингвистических параметров

            Ниже список, перенесенный из `.docx` скриптом. Этот раздел нельзя
            править вручную без повторной генерации и проверки счетчиков.
            """
        ).strip()

        return f"{plan_body}\n\n{feature_tables}\n"

    def _render_feature_group(
        self,
        title: str,
        features: list[LinguisticFeature],
    ) -> str:
        lines = [
            f"### {title}",
            "",
            "| Код | Описание |",
            "| --- | --- |",
        ]
        for feature in features:
            lines.append(f"| `{feature.code}` | {feature.description} |")
        return "\n".join(lines)


class LinguisticFeatureRegistryRenderer:
    """Render a machine-readable registry without manually copying feature rows."""

    def render(self, features: list[LinguisticFeature], source_docx: Path) -> str:
        payload = {
            "schema_version": 1,
            "source_docx": str(source_docx),
            "expected_counts": EXPECTED_COUNTS,
            "features": [
                {
                    "group_title": feature.group_title,
                    "code": feature.code,
                    "description": feature.description,
                }
                for feature in features
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-docx",
        type=Path,
        default=Path(r"C:/Users/ru-lo/Downloads/Telegram Desktop/92 параметра.docx"),
        help="Path to the source DOCX with linguistic parameters.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("FEATURE_EXTRACTION_PLAN.md"),
        help="Output Markdown path.",
    )
    parser.add_argument(
        "--registry-out",
        type=Path,
        default=Path("src/sara_audio/data/linguistic_features.json"),
        help="Output path for the machine-readable linguistic feature registry.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    features = DocxLinguisticFeatureParser().parse(args.source_docx)
    markdown = FeaturePlanMarkdownRenderer().render(features, args.source_docx)
    registry = LinguisticFeatureRegistryRenderer().render(features, args.source_docx)
    args.out.write_text(markdown, encoding="utf-8")
    args.registry_out.parent.mkdir(parents=True, exist_ok=True)
    args.registry_out.write_text(registry, encoding="utf-8")
    print(
        f"Wrote {args.out} and {args.registry_out} "
        f"with {len(features)} linguistic features"
    )


if __name__ == "__main__":
    main()
