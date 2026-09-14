# Результаты и форматы

## Каталог запуска

Для запуска `artifacts/run-001` формируется структура:

```text
run-001/
  input_manifest.json
  run_summary.json
  features_wide.csv
  predictions.csv
  predictions.json
  interpretation_input.csv
  interpretation_input.json
  filter_predictions.csv
  filter_predictions.json
  batches/
    batch-0001.json
  records/
    <record-id>/
      features.csv
      features.parquet          # если включено
      frames.parquet            # если включено
      segments.json             # если включено
      transcript.txt
      transcript.json
      operator_speech_no_pauses.wav
      manifest.json
      report.json
  .cache/
```

Набор файлов зависит от `feature_formats`, `frame_formats`, `write_frames`,
`write_segments`, режима transcript и prediction profile.

## Общие таблицы

### `features_wide.csv`

Одна строка на входную запись. Первые поля:

- `file_id`: безопасный внутренний идентификатор;
- `input_path`: путь к исходному файлу;
- `analysis_audio_path`: speech-only WAV, по которому считались признаки.

Остальные колонки являются скалярными признаками. CSV использует `;` и UTF-8.
Колонки из `exclude_features.txt` удаляются из wide table, но остаются в
детальном наборе записи.

### `predictions.csv/json`

Компактный итог для UI и потребителей прогноза. Одна строка на принятую фильтром
запись. Если все записи исключены, файл существует, но не содержит результатов.

### `interpretation_input.csv/json`

Полная таблица принятой записи после реконструкции expert features и добавления
групповых sex/burnout полей. Предназначена для отдельного интерпретатора и
аудита того, на каких значениях получен итог.

### `filter_predictions.csv/json`

Одна строка на каждый входной файл, включая исключенные:

- имя файла и внутренний id;
- `filter_score`;
- признак прохождения порога;
- сведения о пропущенных коэффициентах, если они есть.

## Артефакты записи

### `features.*`

Long-form таблица, где каждая строка описывает один feature: код, значение,
группа, описание и evidence. Ее используют для отладки и объяснения wide table.

### `frames.*`

Покадровые openSMILE значения. Они велики; в CPU профиле запись frames отключена.

### `segments.json`

Содержит VAD speech segments, pauses, transcript и diagnostics. Временные границы
относятся к нормализованному аудио до удаления пауз.

### `transcript.txt` и `transcript.json`

TXT предназначен для чтения. JSON содержит источник, признак верификации и
ASR-токены с временными метками.

### `operator_speech_no_pauses.wav`

Mono PCM 16 kHz с конкатенированными речевыми интервалами. Именно этот файл
получает ASR. Временная шкала этой записи отличается от исходной после удаления
пауз.

### `manifest.json` и `report.json`

Manifest фиксирует вход и созданные файлы. Report содержит диагностические
сводки, включая coverage и fallback выбора канала.

## Manifests

`input_manifest.json` фиксирует упорядоченный список входов, их operator/year
группу и batch. `batches/batch-*.json` ограничивает логические партии значением
`execution.batch_size`. `run_summary.json` является признаком штатно завершенного
запуска и перечисляет основные результаты.

## Web job

Web создает:

```text
<output-root>/job-<12 hex>/
  processing.log
  error.txt                 # только при web-level ошибке
  results/                  # обычный каталог запуска
  delivery/                 # если отмечен checkbox
  delivery.zip              # delivery archive
  sara-batch.zip            # полный скачиваемый job
```

`processing.log` начинается с полной worker-команды и содержит stage markers.
При native crash последняя строка `BEGIN ...` показывает стадию, которая не
вернула управление.

## Server delivery

`sara-package-server-run SOURCE [DESTINATION]` не пересчитывает аудио. Он
упаковывает завершенный result в:

```text
delivery/
  README.md
  delivery_manifest.json
  tables/
    features_wide.csv
    transcripts.csv
    records_index.csv
    record_diagnostics.csv
    feature_dictionary.csv
    predictions.csv
    interpretation_input.csv
    filter_predictions.csv
    delivery.xlsx
  provenance/
```

Delivery ZIP получает соседний `.sha256`. `delivery_manifest.json` содержит
размер и SHA-256 каждого файла внутри. Все CSV используют `;`; XLSX собирается
без зависимости от Excel.

## Совместимость потребителей

- Не полагайтесь на физический порядок feature columns: используйте имена.
- Не заменяйте пустое значение нулем без методологического решения.
- Не определяйте успешность только по exit code web-сервера: проверяйте
  `run_summary.json` конкретного job.
- Храните вместе YAML config, commit id, model JSON checksums и output manifest.
- Путь `input_path` может раскрывать структуру сервера; удаляйте его при передаче
  данных за пределы доверенной среды.
