# Руководство пользователя

## Запуск

У команды только два позиционных аргумента:

```powershell
sara-extract <аудиофайл-или-директория> [выходной-каталог]
```

Первый аргумент обязателен. Второй необязателен и задает новый каталог
результатов. Никакие параметры VAD, openSMILE, ASR, форматы или расшифровки
через аргументы командной строки не передаются: они задаются только в
`config/default.yaml`.

Примеры:

```powershell
# Автоматически создаст artifacts/raw
sara-extract data\raw

# Создаст ровно этот новый каталог
sara-extract data\raw results\experiment-01

# То же через Python-модуль
.venv\Scripts\python.exe -m sara_audio data\raw results\experiment-01
```

Если каталог с таким именем уже существует и был передан явно, запуск
останавливается без перезаписи результатов. Для автоматического имени при
повторном запуске создается суффикс `-2`, `-3` и так далее.

## Результаты

Для каждого входного файла создается собственная папка:

```text
artifacts/raw/
  run_summary.json
  records/
    A/
      features.csv
      features.json
      features.parquet
      frames.csv
      frames.parquet
      segments.json
      transcript.txt
      transcript.json
      manifest.json
      report.json
    B/
      features.parquet
      frames.parquet
      segments.jsonl
      manifest.json
  .cache/
```

`run_summary.json` связывает исходный аудиофайл с его папкой.

Если в конфигурации включён блок `prediction`, batch дополнительно создаёт:

- `predictions.csv` и `predictions.json` — итоговый прогноз выгорания и пола по информативным записям;
- `interpretation_input.csv` и `interpretation_input.json` — полная таблица признаков, включая восемь реконструированных экспертных показателей;
- `filter_predictions.csv` и `filter_predictions.json` — решение фильтра для каждого входного файла.

Сначала записи со значением `filter_score >= filter_threshold` исключаются из
агрегации. Для оставшихся записей восстанавливаются экспертные показатели,
затем по средним группы «оператор-год» рассчитываются пол (`m/f`) и выгорание.
`burnout_final=1` означает итоговый прогноз наличия выгорания.

Для запуска веб-сервиса без Docker и без CUDA используйте `config/native_cpu.yaml`:

```powershell
py -3.11 -m venv .native-venv
.native-venv\Scripts\python.exe -m pip install -e .[asr,nlp,web]
.native-venv\Scripts\sara-web.exe --config config/native_cpu.yaml --output-root D:\SARA-results --host 0.0.0.0 --port 7860
```

Этот профиль предназначен для проверки сервиса на CPU. BERT-эмбеддинги в нём отключены, поэтому его результаты не заменяют production-прогон с полной MIC-конфигурацией.

- `report.json`: основной файл для передачи человеку. В нем сгруппированы
  временные признаки, скорость, статус расшифровки, лингвистические признаки
  с описаниями и покрытие openSMILE.
- `features.csv`: все скалярные признаки в таблице, одна строка на признак.
- `features.json`: те же скалярные признаки как JSON-массив.
- `features.parquet`: тот же набор для аналитических инструментов.
- `frames.csv` и `frames.parquet`: покадровые LLD openSMILE. Эти файлы обычно
  большие; для обычного чтения человеку удобнее `report.json` и `features.csv`.
- `segments.json`: речевые сегменты, паузы и, при наличии, расшифровка.
- `transcript.txt`: расшифровка в читаемом виде. Создается, когда включена
  ручная расшифровка или ASR.
- `transcript.json`: тот же текст, его источник, статус проверки и timestamps
  слов от ASR. Создается вместе с `transcript.txt`.
- `manifest.json`: диагностика и точные колонки openSMILE.

## Настройка

Менять следует только [config/default.yaml](C:/Users/ru-lo/PycharmProjects/SARA-audio/config/default.yaml).

### Формат и артефакты

```yaml
execution:
  output_root: artifacts
  feature_formats: [csv, json, parquet]
  frame_formats: [csv, parquet]
  write_frames: true
  write_segments: true
```

Для внешней передачи по умолчанию уже создаются CSV и JSON. Можно, например,
оставить только `feature_formats: [csv, json]`. Для frame-level данных
доступны только `csv` и `parquet`.

`exclude_features.txt` в корне проекта применяется к итоговым wide/delivery
таблицам: перечисленные там коды не попадают в `features_wide.csv`,
`delivery.xlsx` и сводные delivery CSV. Подробные per-record `features.*`
остаются полными, чтобы можно было проверить, что именно было рассчитано.

### MIC и embeddings

```yaml
mic:
  enabled: true
  utterance_merge_gap_s: 1.2
text_embeddings:
  enabled: true
  model_name: ai-forever/ru-en-RoSBERTa
  device: auto
  max_length: 512
  feature_prefix: bert_
```

MIC-блок добавляет скалярные признаки из notebook-пайплайна: `word_count`,
паузы между timestamped ASR-токенами, среднюю/накопленную скорость речи,
`duration_sec` и `rms_energy`. Dense text embeddings включены в поставляемых
YAML-профилях по умолчанию и требуют зависимости `.[embeddings]`; серверные
скрипты и Docker-образ устанавливают их автоматически.

### Расшифровка

Без расшифровки (`transcript_mode: none`) запускаются VAD и openSMILE. Для
скорости речи и 92 лингвистических параметров выберите один из двух режимов.

Проверенные ручные тексты:

```yaml
execution:
  transcript_mode: manual_directory
  transcript_directory: data/transcripts
  transcript_suffix: .txt
```

Для входной директории `data/raw` текст должен повторять ее структуру. Так,
для `data/raw/session/A.m4a` используется `data/transcripts/session/A.txt`.
Если хотя бы один текст отсутствует, запуск останавливается с точным путем
недостающего файла, чтобы не создать неполный набор признаков.

ASR через faster-whisper. Для актуальных драйверов CUDA 12 достаточно общего
набора `.[asr]`. Для установленной на этой машине CUDA 11.8 и GTX 1650 нужна
совместимая связка из [CUDA_11_8_SETUP.md](C:/Users/ru-lo/PycharmProjects/SARA-audio/CUDA_11_8_SETUP.md).
После ее установки включите ASR так:

```yaml
execution:
  transcript_mode: asr
asr:
  model: small
  language: ru
  device: cuda
  compute_type: int8_float16
```

ASR-текст не считается проверенной расшифровкой. Поэтому орфографические
параметры и часть пунктуационных признаков получают пониженную уверенность.

### Акустические параметры и паузы

```yaml
audio:
  target_sample_rate_hz: 16000
  mono: true
vad:
  frame_duration_ms: 30
  pause_threshold_ms: 300
  min_speech_duration_ms: 90
  absolute_energy_floor_dbfs: -45.0
  activation_margin_db: 10.0
opensmile:
  enabled: true
  feature_set: ComParE_2016
```

`pause_threshold_ms: 300` -- согласованный порог: интервалы короче 300 ms
склеиваются и не попадают в среднюю внутреннюю паузу.

## Проверка реестра

После изменения кода реестра или генератора выполните:

```powershell
sara-validate-registry
```

Команда должна сообщить `Registry valid: 92 features`.

## Delivery для нового корпуса

Из корня проекта запустите:

```powershell
.venv\Scripts\python.exe -m sara_audio.external_delivery data/new_raw_data
```

Результаты появятся в `deliveries/real-data-existing19`, рядом будут ZIP и
его контрольная сумма SHA-256. Для другого каталога передайте второй аргумент:

```powershell
.venv\Scripts\python.exe -m sara_audio.external_delivery data/new_raw_data deliveries/real-data-run-02
```

Выходной каталог должен быть новым: готовые результаты не перезаписываются.
Это специальный профиль для уже проаудированного корпуса из 390 записей,
а не универсальный импорт произвольных TXT. Настройки находятся только в
`config/external_delivery.yaml`; `workers: 4` задает число параллельных
процессов. Для снижения нагрузки можно установить `workers: 1`.

Используется существующая обработанная расшифровка, только реплики оператора.
Повторный Whisper не запускается. Рассчитываются 92 эвристических текстовых
признака с provenance и confidence.
Для 385 доступных WAV дополнительно сохраняются акустика и локальные временные
параметры; пять записей остаются текстовыми. Скорости речи недоступны, поскольку
соответствие полного текста и укороченного WAV не подтверждено.

В трех расшифровках нет реплик оператора. Такие записи не исключаются:
все текстовые значения остаются пустыми с причиной, полный исходный диалог
сохраняется рядом. Поэтому численные текстовые результаты есть у 387 записей.
Список исключений находится в `tables/no_operator_text_records.csv`.

Начните с `README.md` внутри delivery и
`tables/linguistic_computed_92_wide.csv`: в этой таблице одна запись на строку.
Все определения перенесены скриптом в `tables/feature_dictionary.csv`.
Подробные результаты и использованный текст находятся в `records/<ID>`.

## Docker-сервис с интерфейсом

Для Linux-сервера с NVIDIA Container Toolkit можно собрать и запустить образ:

```bash
docker build -f docker/rtx4090/Dockerfile -t sara-audio:rtx4090 .
docker run --rm --gpus all -p 7860:7860 \
  -v /server/results:/data/output \
  -v /server/hf-cache:/models/huggingface \
  sara-audio:rtx4090
```

Откройте `http://SERVER_IP:7860`, загрузите несколько аудиофайлов и нажмите
кнопку запуска. UI возвращает ZIP с полными результатами, а также может
добавить компактные CSV/XLSX delivery-таблицы. Подробности: [docker/rtx4090/README.md](C:/Users/ru-lo/PycharmProjects/SARA-audio/docker/rtx4090/README.md).
