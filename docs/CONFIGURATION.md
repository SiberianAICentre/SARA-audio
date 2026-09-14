# Конфигурация

Все команды принимают YAML через `--config`. Относительные пути внутри YAML
разрешаются относительно текущего рабочего каталога, поэтому серверные команды
следует запускать из корня репозитория.

## Готовые профили

| Файл | Назначение |
| --- | --- |
| `config/native_cpu.yaml` | дешевый CPU smoke-test, Whisper `small`, без embeddings |
| `config/rtx4090_cuda12.yaml` | основной RTX 4090 профиль, CUDA 12, полный MIC |
| `config/rtx4090_cuda11_8.yaml` | резервный CUDA 11.8 профиль с отдельным стеком |
| `config/default.yaml` | общий шаблон editable-установки |
| `config/external_delivery.yaml` | импорт исторического корпуса без ASR |

Не используйте CUDA 11 requirements с CUDA 12 конфигурацией и наоборот.

## `audio`

| Ключ | Тип | Смысл |
| --- | --- | --- |
| `target_sample_rate_hz` | int | целевая частота, production использует 16000 |
| `mono` | bool | преобразовывать в mono |
| `operator_channel_index` | int/null | zero-based канал оператора; `1` означает второй канал |
| `allow_mono_operator_fallback` | bool | принять единственный mono-канал вместо ошибки |

При mono fallback в диагностике записи появляется предупреждение. Для файлов,
где расположение оператора неизвестно, канал нужно проверить до массового
запуска.

## `vad`

| Ключ | Тип | Смысл |
| --- | --- | --- |
| `frame_duration_ms` | int | размер энергетического кадра |
| `pause_threshold_ms` | int | минимальная внутренняя пауза |
| `min_speech_duration_ms` | int | минимальный речевой интервал |
| `absolute_energy_floor_dbfs` | float | абсолютный нижний порог энергии |
| `activation_margin_db` | float | отступ от оцененного noise floor |

Это собственный VAD pipeline. `asr.vad_filter` в поставляемых профилях выключен,
поскольку Whisper уже получает speech-only WAV. Включение второго VAD меняет
состав речи, добавляет ONNX Runtime и на Windows может вызвать native crash.

## `opensmile`

| Ключ | Тип | Смысл |
| --- | --- | --- |
| `enabled` | bool | включить акустические признаки |
| `feature_set` | string | набор openSMILE, сейчас `ComParE_2016` |
| `frame_step_seconds` | float | шаг кадра |
| `aggregate_statistics` | list | агрегаты кадра: mean/std/min/max |

## `mic`

`enabled` включает скалярные MIC-признаки. `utterance_merge_gap_s` объединяет
соседние ASR-фрагменты при расчете utterance-level статистик.

## `text_embeddings`

| Ключ | Тип | Смысл |
| --- | --- | --- |
| `enabled` | bool | включить RoSBERTa |
| `model_name` | string | Hugging Face model id или локальный путь |
| `device` | string | `cpu`, `cuda` или `auto` |
| `max_length` | int | максимальная длина токенизированного текста |
| `feature_prefix` | string | префикс колонок, по контракту `bert_` |

В профиле RTX 4090 embeddings считаются на CPU. Это не ошибка: CTranslate2
использует GPU для ASR, а раздельные runtimes уменьшают риск DLL-конфликтов на
Windows.

## `execution`

| Ключ | Допустимые значения | Смысл |
| --- | --- | --- |
| `output_root` | path | базовый каталог по умолчанию |
| `feature_formats` | csv/json/jsonl/parquet | форматы скалярных признаков записи |
| `frame_formats` | csv/parquet | форматы openSMILE frames |
| `write_frames` | bool | сохранять покадровую таблицу |
| `write_segments` | bool | сохранять VAD segments и pauses |
| `transcript_mode` | `asr`, `manual_directory`, `none` | источник текста |
| `transcript_directory` | path/null | каталог ручных transcript-файлов |
| `transcript_suffix` | string | обычно `.txt` |
| `batch_size` | positive int | размер manifest-партии, production 15 |

`batch_size` не запускает параллельный GPU inference. Он задает логические
партии и provenance manifests.

## `asr`

| Ключ | Смысл |
| --- | --- |
| `model` | faster-whisper model id, например `small` или `large-v3` |
| `language` | код языка, production `ru` |
| `device` | `cpu`, `cuda` или `auto` |
| `compute_type` | CTranslate2 compute type, например `int8` или `float16` |
| `initial_prompt` | доменная подсказка Whisper |
| `vad_filter` | дополнительный faster-whisper VAD; должен оставаться `false` |
| `no_speech_threshold` | фильтр no-speech сегментов |
| `compression_ratio_threshold` | фильтр повторов/галлюцинаций |
| `temperature` | одно значение или fallback-последовательность |
| `suppress_common_hallucinations` | post-filter известных пустых фраз |

Поля `vad_min_speech_duration_ms`, `vad_max_speech_duration_s` и
`vad_speech_pad_ms` используются только при `vad_filter: true`.

## `prediction`

| Ключ | Смысл |
| --- | --- |
| `enabled` | запускать прогноз после извлечения |
| `filter_model_path` | JSON простого фильтра |
| `methodology_model_directory` | каталог полного набора из 11 JSON |
| `filter_threshold` | порог фильтра, обычно 0.5 |
| `formats` | `csv`, `json` или оба |

Если `methodology_model_directory` не задан, выполняется только filter model.
Если задан, отсутствие любого обязательного файла является ошибкой запуска.

## Переменные окружения web

| Переменная | По умолчанию в Docker | Смысл |
| --- | --- | --- |
| `SARA_CONFIG_PATH` | `/app/config/rtx4090_cuda12.yaml` | YAML профиля |
| `SARA_WEB_OUTPUT_ROOT` | `/data/output/web-runs` | постоянные job-каталоги |
| `SARA_WEB_TEMP_ROOT` | системный temp | временные копии uploads |
| `SARA_WEB_HOST` | `0.0.0.0` | bind address |
| `SARA_WEB_PORT` | `7860` | bind port |
| `HF_HOME` | `/models/huggingface` | кэш загружаемых моделей |

Аргументы CLI имеют приоритет над web-переменными окружения.

## Создание нового профиля

1. Скопируйте ближайший рабочий YAML.
2. Меняйте один технический параметр за раз.
3. Не переименовывайте feature columns без новой версии моделей.
4. Запустите `sara-diagnose-asr --load-model`.
5. Запустите `Trace-SaraPipeline.ps1` или `sara-server-smoke` на реальном файле.
6. Сохраните YAML, версии requirements и результат smoke-test вместе с релизом.
