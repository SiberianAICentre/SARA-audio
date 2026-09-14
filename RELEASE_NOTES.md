# Release Notes

## Unreleased

### 2026-09-15 - External documentation and reproducibility checks

Added a complete maintainer documentation set covering architecture,
configuration, CLI, MIC model semantics, output contracts, Windows/Docker
deployment, development, security, and native-crash troubleshooting. The main
README and Windows quick start now identify the pinned native CUDA 12 path as
the primary server procedure and explicitly separate CUDA 11.8 dependencies.

Added `sara-validate-repository`, which validates documentation links, YAML
profiles and model paths, all 11 methodology JSON files and their SHA-256
checksums, and the Docker model-copy contract without loading heavyweight ML
models. Docker images now include `models/methodology`, and redundant
faster-whisper VAD is disabled in the base CPU/default profiles.

### 2026-09-14 - MIC additions and delivery exclusions

Ported MIC notebook additions into the package pipeline: ASR can use the
domain prompt and VAD parameters from YAML, common Whisper hallucination
segments are suppressed, and the pipeline now emits MIC scalar features such as
`word_count`, interword pauses, speech-rate aggregates, `duration_sec`, and
`rms_energy`.

Text embeddings are enabled in the shipped YAML profiles by default. Server
installation now uses `.[asr,nlp,parquet,embeddings]`, and the smoke test
requires `bert_1` so a server run cannot pass without the embedding block.

`exclude_features.txt` is now applied to final wide/delivery exports while
preserving full per-record scalar artifacts for audit. Server delivery packaging
also filters old completed `features_wide.csv` inputs before writing CSV/XLSX.

Added an RTX 4090 Docker fallback: the image installs the full server
dependency set, runs corpus preparation, CUDA smoke-test, full extraction, and
optionally delivery packaging from one container entrypoint.

### 2026-09-11 - Шаг 0: контракт поставки

Добавлен `IMPLEMENTATION_PLAN.md` с целевой архитектурой: выделение речи
оператора со второго канала, группировка `operator-year`, GPU-профиль RTX 4090,
93 предметных признака, embeddings Whisper и текста, интеграция VegaQ1 и
демонстрационный выход.

Зафиксированы критерии готовности для каждого шага, обязательный provenance
результатов и правило: отсутствующие/ненадёжные значения не маскируются нулями.
Также отдельно отмечены внешние контракты для будущего подключения
математической модели и text embeddings.

### 2026-09-11 - Шаг 1: канал оператора и речь без пауз

Добавлен `audio.operator_channel_index`. В поставляемом
`config/default.yaml` он равен `1`: из стереозаписи извлекается второй канал
оператора без downmix с клиентом. WAV с недостаточным числом каналов теперь
даёт диагностируемую ошибку.

После VAD создаётся `operator_speech_no_pauses.wav` и сохраняется в каталоге
записи. ASR и openSMILE запускаются на этом файле; временные признаки остаются
привязаны к выделенному каналу с исходными паузами. В manifest и segments
сохраняется путь к фактически проанализированному аудио.

Добавлены проверки выделения второго канала, удаления пауз, отказа на mono при
явно запрошенном втором канале и регрессии пакетного вывода. Проверка: `22
passed`, `ruff check src tests`.

### 2026-09-11 - Шаг 2: входной manifest и пакеты operator-year

Добавлен `InputManifestBuilder`. Каждый запуск теперь создаёт
`input_manifest.json` с `record_id`, абсолютным путём, SHA-256, `operator_id`,
годом и статусом полноты метаданных. Метаданные извлекаются из типовых имён
вроде `20230201144625_Agent_09.wav`; отсутствие года или оператора явно
маркируется `incomplete-metadata` и направляется в группу `unknown`.

Добавлены `batches/<operator>-<year>-<index>.json`. Записи внутри каждой
группы сортируются по `record_id` и разбиваются на части размера
`execution.batch_size` (по умолчанию 15). Пути manifest включены в
`run_summary.json` и объект результата пакетного запуска.

Добавлены тесты детерминированной группировки и разбиения 16 записей на `15 +
1`, хэширования и регрессия обычного пакетного вывода. Проверка: `23 passed`,
`ruff check src tests`.

### 2026-09-11 - Шаг 3: подготовка RTX 4090 / CUDA 12

Добавлены [config/rtx4090_cuda12.yaml](config/rtx4090_cuda12.yaml),
`requirements-cuda12.txt`, Dockerfile на CUDA 12.4 + cuDNN 9 и
`RTX4090_CUDA12_SETUP.md`. Серверный профиль использует `large-v3`, `cuda`,
`float16`, второй канал оператора и размер batch 15.

В CLI добавлен `--config`, поэтому запуск использует серверный профиль без
изменения исходников. Добавлена команда `sara-diagnose-asr --config ...
--load-model`: она показывает `nvidia-smi`, число CUDA-устройств CTranslate2 и
подтверждает загрузку Whisper на заданном device. Полный тестовый набор и
линтер проходят: `25 passed`, `ruff check src tests`.

Smoke-run на реальном RTX 4090 не выполнялся из этой рабочей среды и остаётся
явным критерием завершения шага 3.

### 2026-09-11 - Шаг 4: 92 лингвистических признака

`LinguisticFeatureExtractor` заменён на переносимый локальный backend
`russian-heuristics-v2`. Все 92 кода существующего реестра теперь возвращают
числовое значение `heuristic-text`; отсутствуют строки `not-computed`, вызванные
нехваткой реализации. Для семантических, орфографических и синтаксических
эвристик сохраняются пониженная `confidence` и provenance, поэтому они не
выдаются за экспертную оценку.

Обновлены архивный delivery-путь, пакетные проверки и документация: записи с
текстом оператора имеют покрытие 92/92, записи без такого текста остаются
явно недоступными. Проверка полного набора ожидается после обновления legacy
пакетного контракта.

### 2026-09-11 - Решение по Whisper embeddings

Audio embeddings Whisper исключены из обязательной поставки до inference
14.09.2026. Encoder Whisper не является специализированным представлением
спикера, а его безопасное подключение потребовало бы отдельного model contract,
pooling, кэша и валидации. Основной inference использует ASR `large-v3`,
текстовые признаки, temporal и openSMILE без этого блока.

### 2026-09-11 - Уточнение реестра признаков

Нормативный реестр содержит 92, а не 93 лингвистических признака. Упоминание
93-го признака удалено из плана: это число было перенесено из ранней постановки,
но не подтверждалось исходным DOCX или машинным реестром.

### 2026-09-11 - Шаг 5a: агрегаты openSMILE

Покадровые LLD openSMILE теперь дополняются scalar-признаками
`ACOUSTIC_*_mean`, `std`, `min`, `max`. Агрегируются только колонки,
попадающие в заданные семантические группы: energy, loudness, MFCC, F0,
jitter/shimmer, форманты, sharpness, harmonicity и отношения гармоник.

Агрегаты считаются по `operator_speech_no_pauses.wav`; исходный покадровый
`frames.csv/parquet` сохранён без изменений для проверки. Проверка: `25
passed`, `ruff check src tests`.

### 2026-09-11 - Шаг 5b: ASR-диагностика и wide-матрица

Добавлены пять ASR-признаков из metadata токенов: число токенов, средняя и
минимальная confidence, средняя длительность токена и доля токенов с
timestamps. Если расшифровка не содержит ASR-токенов, эти поля явно получают
`not-computed`, а не подменяются нулями.

Каждый пакетный запуск теперь создаёт `features_wide.csv` в корне результата:
одна строка на файл, а колонками служат все scalar-признаки, включая 92
лингвистических, TIME/RATE, `ACOUSTIC_*` и `ASR_*`. Long-form `features.csv`
и покадровый output не изменены. Проверка: `26 passed`, `ruff check src tests`.

### 2026-09-11 - Серверный smoke-test

Добавлена команда `sara-server-smoke`. Она запускается только в серверном
CUDA-контейнере и проверяет: видимость NVIDIA GPU для CTranslate2, загрузку
Whisper на `cuda`, ASR-токены, `operator_speech_no_pauses.wav`, presence
`ACOUSTIC_*` и обязательных wide-колонок, совпадение числа строк с файлами и
пакеты размером не более 15. Успех обозначается только JSON-полем
`"status": "passed"`.

Инструкция запуска добавлена в `RTX4090_CUDA12_SETUP.md`. Unit-проверки
покрывают корректный и отказной сценарии верификатора. Проверка: `28 passed`,
`ruff check src tests`.

### 2026-09-11 - Windows Server one-command deployment

Добавлен `scripts/windows/Invoke-SaraServerPipeline.ps1` и
`WINDOWS_SERVER_QUICKSTART.md`. Сценарий не использует Git: после копирования
проекта, `cuda-libs` и входных файлов он создаёт `.venv`, устанавливает Python
зависимости, настраивает DLL через `SARA_CUDA_LIB_DIR`, выполняет smoke-test и
запускает полный расчёт. Код ASR больше не требует строго cuDNN 8 на Windows и
принимает runtime cuDNN 8 или 9.

### 2026-09-11 - Упрощённый запуск Windows Server

`Invoke-SaraServerPipeline.ps1` теперь запускается без параметров: берёт
`data\raw`, автоматически выбирает одну запись для smoke-test и пишет результат
в timestamped каталог `results\server-...`. CUDA берётся из уже настроенной
серверной среды или автоматически найденного CUDA Toolkit; `cuda-libs` и
`-CudaLibraryPath` оставлены только как аварийный override.

### 2026-09-11 - Автоматическая подготовка train + test

Добавлена команда `sara-prepare-corpus` и безопасный `CorpusPreparer`. Для
поставки `data\new_raw_data\2_file` он извлекает только WAV из `test.zip` в
`test\`, не допускает path traversal внутри ZIP и проверяет наличие аудио в
обоих split. Windows Server entrypoint выполняет этот шаг автоматически перед
полным расчётом, поэтому train и test не расходятся по разным запускам.

Добавлены тесты обычной распаковки и отказа на небезопасном ZIP. Проверка:
`30 passed`, `ruff check src tests`.

### 2026-09-11 - Windows installation timeout handling

Windows entrypoint no longer attempts an unnecessary `pip` upgrade before
installation. It disables version checks and uses a 120-second timeout for the
actual dependency installation, preventing slow PyPI access from blocking the
server before CUDA validation or corpus processing begins.

### 2026-09-11 - Isolated Windows server environment

The launcher now creates and uses `.server-venv`, so a copied local `.venv`
cannot affect package resolution or CUDA startup on the server. It also checks
each external setup, smoke-test, and extraction command's exit code.

### 2026-09-11 - Visible calculation progress

The Windows launcher now prints five deployment stages. `sara-extract` and
`sara-server-smoke` print a completed/total counter and source file name for
every processed recording.

### 2026-09-11 - Offline Windows dependency bundle

Added `New-SaraWheelhouse.ps1`, which builds a wheel for this project and
builds wheels for all its Windows Python 3.11 dependencies, including packages
published only as source archives. The server launcher can install the bundled
project and all runtime extras using `-Offline` without PyPI access.

### 2026-09-11 - CUDA 11.8 server profile

The Windows Server launcher now uses `rtx4090_cuda11_8.yaml` with
`faster-whisper 0.10.1` and CTranslate2 3.24.0, which are compatible with
CUDA 11.8 and cuDNN 8. The Whisper model remains `large-v3` in `float16`;
only the inference engine was changed for runtime compatibility.

The offline launcher explicitly installs the verified PyAV 18 decoder because
the legacy faster-whisper package metadata requires PyAV 10, which has no
Windows wheel for Python 3.11.

The optional `-WheelhousePath` parameter permits separate CUDA-specific
offline dependency bundles to coexist in the copied project.

### 2026-09-11 - Mono operator-channel fallback

When a file is mono but the server profile requests operator channel 2, the
pipeline now uses the sole available channel and records that choice in the
per-record diagnostics. Stereo inputs continue to use channel 2. Set
`audio.allow_mono_operator_fallback: false` to retain strict rejection.

### 2026-09-11 - AppleDouble input filtering

Directory processing now ignores `._*.wav` AppleDouble metadata sidecars before
batch manifests are built. These files are not audio despite their `.wav`
suffix and no longer interrupt a valid corpus run.

### 2026-09-12 - Preserve completed server results

The Windows launcher now treats `features_wide.csv` as the completion marker.
If the extractor reports a nonzero process code after all artifacts were
written, the launcher preserves the completed result and emits a warning
instead of reporting a failed full run.

### 2026-09-12 - Server result delivery

Added `sara-package-server-run`. It packages a completed `results/server-*`
directory into a ZIP with the model-ready wide matrix, ASR transcripts,
record diagnostics, feature dictionary, provenance, and SHA-256 manifest,
without rerunning ASR, VAD, or openSMILE.

### 2026-09-12 - Excel-only server delivery

Server delivery now puts every tabular result into one dependency-free
`tables/delivery.xlsx` workbook: `Features`, `Transcripts`, `Records`,
`Diagnostics`, and `Dictionary`. It no longer emits CSV tables, so commas in
transcript text cannot change field boundaries. Packaging still copies only
already computed artifacts and never reruns ASR, VAD, or openSMILE.

### 2026-09-12 - Semicolon CSV and one-command Excel delivery

New pipeline CSV artifacts use `;` as their delimiter. Existing comma-delimited
results remain readable for packaging. `Invoke-SaraExcelDelivery.ps1` installs
the bundled update offline and creates the Excel delivery from a completed
result in one command, without ASR, VAD, openSMILE, CUDA, or model execution.

### 2026-09-12 - Complete delivery tables restored

The Excel delivery is now additional rather than a replacement. Every delivery
again includes `features_wide.csv`, `transcripts.csv`, `records_index.csv`,
`record_diagnostics.csv`, and `feature_dictionary.csv`; all use UTF-8 BOM and
a semicolon delimiter. `delivery.xlsx` contains the same five tables as sheets.
