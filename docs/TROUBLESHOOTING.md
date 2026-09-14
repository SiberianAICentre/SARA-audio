# Диагностика

## Сначала определите слой сбоя

1. `nvidia-smi`: виден ли GPU из той же учетной записи.
2. `sara-diagnose-asr --load-model`: загружается ли Whisper на указанном device.
3. Прямой faster-whisper test: работает ли только ASR без остальных стадий.
4. `Trace-SaraPipeline.ps1`: на каком `BEGIN` закончился полный pipeline.
5. `run_summary.json`: завершилась ли файловая транзакция.
6. `predictions.csv`: дошел ли результат до методологии.

Не переустанавливайте зависимости до определения слоя: успешная загрузка модели
не доказывает совместимость CTranslate2, PyTorch, ONNX Runtime и openSMILE в одном
процессе.

## Где искать логи

| Сценарий | Лог |
| --- | --- |
| Web startup и traceback orchestration | `<output-root>/sara-web.log` |
| Конкретное web-задание | `<output-root>/job-*/processing.log` |
| Ошибка, пойманная web | `<output-root>/job-*/error.txt` |
| Ручной trace | `logs/pipeline-trace-*.log` |
| Частичные результаты | соответствующий `job-*/results` или trace output |

Последний `BEGIN Class.method` без соответствующего `END` указывает на стадию,
внутри которой процесс завершился на native уровне.

## Windows exit code 3221225477

`3221225477` и signed `-1073741819` означают `0xC0000005`, access violation.
Это не обычное Python exception, поэтому traceback может отсутствовать. Для
данного проекта основные причины:

- несовместимые DLL CTranslate2/cuBLAS/cuDNN;
- смешение CUDA PyTorch и CUDA CTranslate2 разных поколений;
- ONNX Runtime VAD внутри процесса с остальным ML stack;
- авария при уничтожении native objects после уже завершенной работы.

Рабочая конфигурация проекта снижает риски так:

- pinned native CUDA 12 installer;
- CTranslate2 ASR на GPU, CPU-only PyTorch embeddings;
- `asr.vad_filter: false`;
- web job в отдельном `web_worker`;
- `os._exit(0)` после flush успешного worker.

Не считайте падение после строки `SUCCESS` неуспешной обработкой старой CLI:
сначала проверьте `run_summary.json`. В актуальном worker штатный код должен быть
`0`.

## `cublas64_12.dll` или `cudnn*.dll` не найден

Запущено окружение, чей CTranslate2 ожидает другой runtime, либо каталоги wheel
DLL не добавлены в `PATH`. Не копируйте одну DLL вручную и не смешивайте cu11 и
cu12 packages. Создайте чистое окружение соответствующим installer/Dockerfile.

Проверка активного интерпретатора:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -c "import sys; print(sys.executable)"
.\.cuda12-venv-final\Scripts\python.exe -m pip show ctranslate2 faster-whisper
```

## `c10.dll` / WinError 1114

Это ошибка инициализации PyTorch или его зависимостей. На Windows не ставьте
CUDA PyTorch поверх CTranslate2 CUDA environment. Основной installer намеренно
ставит CPU-only `torch==2.5.1` без dependency resolution, затем фиксирует sympy.

После установки проверьте:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -c "from transformers.models.roberta.modeling_roberta import RobertaModel; print('OK')"
```

## `pkg_resources` отсутствует

Старые версии CTranslate2 импортируют `pkg_resources`. Установите совместимый
setuptools в конкретный venv, а не глобально:

```powershell
<venv>\Scripts\python.exe -m pip install "setuptools<81"
```

Предупреждение о deprecated `pkg_resources` не является причиной остановки,
если импорт и model load завершились.

## Конфликт `tokenizers`, `transformers`, `huggingface-hub`

Это признак смешанных профилей или свободного `pip install --upgrade`. Не
исправляйте отдельный конфликт еще одним upgrade. Удалите/переименуйте venv и
воспроизведите выбранный профиль с нуля.

Основные сочетания репозитория:

- CUDA 12 native: `requirements-cuda12-native.txt` + installer;
- CUDA 12 Docker: `requirements-cuda12.txt` + CUDA 12 Dockerfile;
- CUDA 11.8: `requirements-cuda11.txt` + CUDA 11.8 Dockerfile/script.

## ASR test работает, а полный pipeline падает

Значит модель и базовые CUDA DLL доступны, а сбой возникает при совместной
работе стадий. Запускайте `Trace-SaraPipeline.ps1`, который использует тот же
изолированный worker, что web. Типовые маркеры:

- `BEGIN OpenSmileExtractor.extract_frames`: openSMILE/native audio;
- `BEGIN FasterWhisperTranscriber.transcribe`: ASR/CUDA;
- `BEGIN TextEmbeddingExtractor.extract`: PyTorch/Transformers;
- `BEGIN JsonLinearModel.load`: JSON path/schema;
- `BEGIN BatchProcessor._write_record`: файловый вывод.

Если trace успешен, но web падает, сравните Python executable в первой строке
`COMMAND`, config path, рабочий каталог и права output/temp.

## Web сбросился без сообщения

В актуальной архитектуре native crash должен убить worker, а не основной Gradio
процесс. Проверьте, что web запущен через Python из обновленного venv и импортирует
код текущего checkout:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -c "import sara_audio; print(sara_audio.__file__)"
```

Если путь указывает на другой checkout, переустановите editable package в
нужное окружение либо создайте чистое окружение installer.

## Python/venv path не найден

`VenvPath` в PowerShell разрешается относительно `ProjectRoot`, а не домашнего
каталога. Находясь в корне проекта, используйте:

```powershell
-VenvPath .\.cuda12-venv-final
```

Проверьте реальный файл:

```powershell
Test-Path .\.cuda12-venv-final\Scripts\python.exe
```

Не запускайте `sara-extract.exe`, скопированный вместе с venv из другого пути:
Windows launcher хранит абсолютный путь к исходному Python. Используйте
`python.exe -m sara_audio...` или создайте venv заново на целевой машине.

## Docker pipe не найден на Windows

Сообщение про `dockerDesktopLinuxEngine` означает, что клиент Docker установлен,
но Linux engine Docker Desktop не запущен или выбран другой context. Это не
ошибка SARA. `docker version` должен показать и Client, и Server до build.

Большой build context также замедляет сборку. `.dockerignore` исключает corpus,
results, archives, venv и model cache; не храните временные аудиофайлы вне этих
исключенных каталогов.

## Прогнозы отсутствуют

Проверьте по порядку:

1. `prediction.enabled: true`;
2. существуют `models/methodology/*.json`;
3. в output есть `filter_predictions.csv`;
4. есть хотя бы одна строка с `accepted=true`;
5. `interpretation_input` содержит нужные model features;
6. имя файла корректно определяет operator/year group.

Если все записи исключены, пустой `predictions.csv` является корректным
техническим результатом, а не аварией.

## JSON model error

- Нет `Intercept`: файл не соответствует контракту.
- Нечисловой коэффициент: поврежден или неверно экспортирован JSON.
- Нет обязательного файла: передан неполный methodology bundle.
- Много missing features: extractor/config не соответствует версии модели.

Сравните канонический `models/methodology` с исходным `meth_patch` и не меняйте
имя файла простым копированием без regression test.

## Медленная первая обработка

Whisper и RoSBERTa загружаются с Hugging Face при первом использовании. Задайте
постоянный `HF_HOME`; проверьте сеть, свободное место и права. Web запускает новый
worker на каждый job, поэтому Python model objects не сохраняются между jobs,
но скачанные файлы остаются в cache.

## Минимальный отчет об ошибке

Сохраните:

- commit id (`git rev-parse HEAD`);
- выбранный YAML;
- `python --version`, `pip freeze`, `nvidia-smi`;
- вывод `sara-diagnose-asr --load-model`;
- полный `processing.log` или trace log;
- `error.txt` и частичный `run_summary.json`, если есть;
- тип/канальность тестового файла без передачи чувствительного содержимого.

Не отправляйте реальное клиентское аудио в публичный issue.
