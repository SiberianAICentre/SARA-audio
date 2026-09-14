# SARA-audio

SARA-audio представляет собой воспроизводимый сервис пакетной обработки записей
контакт-центра. Он извлекает акустические, временные, ASR, лингвистические и
текстовые признаки, применяет каскад моделей MIC и возвращает результаты
фильтрации, реконструированные экспертные признаки, пол оператора и прогноз
выгорания.

В репозитории находятся Python-пакет, CLI-команды, web-интерфейс Gradio,
Windows-скрипты развертывания и Docker-профили для CUDA 11.8 и CUDA 12.

## Что делает сервис

Для каждой записи pipeline:

1. выбирает настроенный канал оператора и преобразует аудио в mono PCM 16 кГц;
2. определяет речь и паузы встроенным энергетическим VAD;
3. сохраняет WAV только с речью оператора;
4. распознает речь через faster-whisper;
5. извлекает openSMILE, временные, ASR, MIC, лингвистические и RoSBERTa признаки;
6. применяет MIC-фильтр и восстанавливает восемь экспертных признаков;
7. объединяет прошедшие фильтр записи по оператору и году;
8. рассчитывает пол оператора и итоговый прогноз выгорания;
9. сохраняет детальные артефакты, общие таблицы и при необходимости delivery ZIP.

Web-интерфейс показывает итоговый прогноз и позволяет скачать полный пакет
результатов.

## Структура репозитория

```text
src/sara_audio/           Python-пакет и точки входа CLI
src/sara_audio/data/      Версионированный реестр лингвистических признаков
config/                   CPU и CUDA профили
models/methodology/       Канонические JSON-коэффициенты MIC
meth_patch/               Исходная передача файлов от методологов
scripts/windows/          Установка, запуск, трассировка и установка патча
docker/rtx4090/           Docker-профили CUDA 11.8 и CUDA 12
tests/                    Автоматические тесты
docs/                     Архитектура и документация разработчика
```

## Быстрый запуск на Windows Server

Основной проверенный профиль использует Windows, Python 3.11, RTX 4090 и CUDA 12.
В PowerShell из корня репозитория:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Install-SaraCuda12Native.ps1
```

Установщик самостоятельно создает новое окружение `.cuda12-venv-final`, ставит
зафиксированный набор зависимостей, проверяет RoBERTa и загружает Whisper на GPU.
Если целевой каталог уже существует, скрипт останавливается, чтобы не смешивать
несовместимые зависимости.

Перед production-запуском проверьте полный pipeline на реальном аудиофайле:

```powershell
.\scripts\windows\Trace-SaraPipeline.ps1 `
  -InputPath D:\SARA-input\smoke.wav `
  -VenvPath .\.cuda12-venv-final `
  -ConfigPath .\config\rtx4090_cuda12.yaml
```

Успешная проверка завершается кодом `0`, строкой `SUCCESS` в логе и созданием
`run_summary.json`.

Запуск web-интерфейса:

```powershell
$env:HF_HOME = "D:\SARA-model-cache"
.\.cuda12-venv-final\Scripts\python.exe -m sara_audio.webapp `
  --config .\config\rtx4090_cuda12.yaml `
  --output-root D:\SARA-results `
  --temp-root D:\SARA-temp `
  --host 0.0.0.0 `
  --port 7860
```

Интерфейс будет доступен по адресу `http://SERVER_IP:7860`. Не открывайте Gradio
напрямую в интернет. Для production необходимы reverse proxy, TLS,
аутентификация и сетевые ограничения.

Подробная инструкция находится в
[документе по развертыванию](docs/DEPLOYMENT.md).

## Локальная разработка

Reference version интерпретатора: Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,nlp,web]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\sara-validate-repository.exe
```

Extras `asr`, `embeddings` и `parquet` устанавливаются только при необходимости.
На Windows CUDA используйте готовый pinned installer, а не произвольную
комбинацию extras.

Обработка одного файла или каталога в CPU-профиле:

```powershell
.\.venv\Scripts\sara-extract.exe .\input .\artifacts\run-001 `
  --config .\config\native_cpu.yaml
```

Поддерживаются WAV, M4A, MP3, FLAC и OGG. `imageio-ffmpeg` содержит собственный
ffmpeg; системный `ffmpeg` из `PATH` имеет приоритет.

## Профили выполнения

| Профиль | Назначение | ASR | Embeddings |
| --- | --- | --- | --- |
| `config/native_cpu.yaml` | CPU smoke-test и отладка | `small`, CPU int8 | выключены |
| `config/rtx4090_cuda12.yaml` | основной Windows/Docker production | `large-v3`, CUDA float16 | CPU |
| `config/rtx4090_cuda11_8.yaml` | резервный legacy CUDA 11.8 | `large-v3`, CUDA float16 | CPU |
| `config/default.yaml` | общий шаблон editable-установки | настраивается | CPU |

Зависимости CUDA 11 и CUDA 12 разделены намеренно. Не устанавливайте оба профиля
в одно виртуальное окружение.

## Модели

Одиннадцать небольших JSON-файлов MIC находятся в `models/methodology/` и входят
в репозиторий. Whisper `large-v3` и `ai-forever/ru-en-RoSBERTa` загружаются с
Hugging Face при первом запуске. На сервере задайте постоянный `HF_HOME`.

Коэффициенты JSON применяются непосредственно к значениям одноименных признаков:

```text
score = Intercept + sum(feature_value * coefficient)
```

Текущие JSON не содержат mean/scale или других параметров стандартизации. Поэтому
сервис не стандартизирует признаки перед прогнозом. При замене моделей этот
контракт нужно отдельно подтвердить с методологами.

Полное описание находится в
[документе по методологии MIC](docs/METHODOLOGY.md).

## Документация

- [Архитектура](docs/ARCHITECTURE.md)
- [Конфигурация](docs/CONFIGURATION.md)
- [CLI и Windows-скрипты](docs/CLI.md)
- [Модели и методология MIC](docs/METHODOLOGY.md)
- [Форматы результатов](docs/OUTPUTS.md)
- [Развертывание и эксплуатация](docs/DEPLOYMENT.md)
- [Разработка](docs/DEVELOPMENT.md)
- [Диагностика](docs/TROUBLESHOOTING.md)
- [Краткий запуск Windows Server](WINDOWS_SERVER_QUICKSTART.md)

## Проверка комплекта

После установки выполните:

```powershell
.\.venv\Scripts\sara-validate-repository.exe
```

Команда проверяет документацию и ссылки, YAML-профили, наличие model paths,
структуру и SHA-256 всех 11 MIC JSON, а также включение моделей в Docker images.
Тяжелые ML-модели при этом не загружаются.

## Ограничения

- Прогноз является результатом методологической модели, а не медицинским диагнозом.
- Группа оператор-год определяется по имени файла. Неструктурированные имена
  попадают в общую группу `uploaded_batch`.
- Для stereo-файла production-профиль ожидает оператора во втором канале с
  zero-based индексом `1`. Mono принимается с диагностическим предупреждением.
- Web-сервис не содержит встроенных TLS, аутентификации, авторизации и политики
  хранения результатов.
- В репозитории пока нет открытой лицензии. До выбора лицензии публичное
  распространение и создание производных версий не разрешено.
