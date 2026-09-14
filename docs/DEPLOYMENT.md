# Развертывание и эксплуатация

## Матрица поставки

| Вариант | Состояние | Когда применять |
| --- | --- | --- |
| Windows native, CUDA 12 | основной проверенный путь | текущий Windows Server и RTX 4090 |
| Docker, CUDA 12 | переносимая Linux-поставка | когда NVIDIA Container Toolkit работает |
| Docker/native, CUDA 11.8 | резервный legacy-профиль | только для хоста с подтвержденным CUDA 11 стеком |
| CPU | диагностика логики | не для производительной полной обработки |

Версия драйвера NVIDIA и CUDA runtime внутри окружения связаны, но не равны.
Наличие нового драйвера не требует устанавливать самый новый пакет CUDA в venv.
Используйте зафиксированный набор конкретного профиля.

## Windows native CUDA 12

### Требования к машине

- Windows Server/Windows x64;
- NVIDIA GPU с рабочим `nvidia-smi`;
- Python 3.11 x64 и launcher `py`;
- свободное место для двух крупных Python CUDA wheels, Whisper, RoSBERTa,
  результатов и временных WAV;
- доступ к PyPI и Hugging Face на первой установке либо заранее подготовленный
  локальный кэш/пакет.

Системный CUDA Toolkit не является обязательным для основного native installer:
cuBLAS, cuDNN и NVRTC устанавливаются Python wheels внутрь venv.

### Чистая установка

Распакуйте или клонируйте репозиторий в постоянный каталог. Не переносите venv с
другой машины и не создавайте целевой venv вручную.

```powershell
cd D:\Services\SARA-audio
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Install-SaraCuda12Native.ps1
```

Скрипт создает `.cuda12-venv-final` и ставит:

- версии из `requirements-cuda12-native.txt`;
- CTranslate2 `4.4.0` и faster-whisper `1.2.1`;
- CUDA 12.4 cuBLAS/NVRTC и cuDNN 8.9 wheels;
- CPU-only PyTorch `2.5.1` для embeddings;
- `sympy 1.13.1` как совместимую зависимость PyTorch.

После установки он импортирует RoBERTa и загружает Whisper `large-v3` на CUDA.
Диагностика должна содержать:

```json
{
  "configured_device": "cuda",
  "cuda_device_count": 1,
  "model_loaded": true,
  "model_device": "cuda"
}
```

Если каталог `.cuda12-venv-final` уже есть, выберите новый `-VenvPath` или
сначала вручную переименуйте старый для отката. Не доустанавливайте новый профиль
поверх окружения с CUDA 11.

### End-to-end приемка

Используйте реальную запись с тем же количеством каналов, что production:

```powershell
$wav = "D:\SARA-input\smoke.wav"
.\scripts\windows\Trace-SaraPipeline.ps1 `
  -InputPath $wav `
  -VenvPath .\.cuda12-venv-final `
  -ConfigPath .\config\rtx4090_cuda12.yaml
```

Успех означает одновременно:

- `Pipeline exit code: 0`;
- в логе есть `SUCCESS`;
- создан `run_summary.json`;
- существуют `features_wide.csv`, `filter_predictions.csv`,
  `predictions.csv` и `interpretation_input.csv`;
- `processing.log`/trace не заканчивается на незакрытом `BEGIN`.

### Запуск web

```powershell
$env:HF_HOME = "D:\SARA-model-cache"
.\.cuda12-venv-final\Scripts\python.exe -m sara_audio.webapp `
  --config .\config\rtx4090_cuda12.yaml `
  --output-root D:\SARA-results `
  --temp-root D:\SARA-temp `
  --host 127.0.0.1 `
  --port 7860
```

Для доступа из сети либо укажите `0.0.0.0`, либо оставьте localhost и
проксируйте через IIS/nginx/Caddy. Второй вариант предпочтителен: reverse proxy
должен обеспечивать TLS, аутентификацию, ограничение размера upload и timeout,
достаточный для длинного аудио.

В репозитории нет Windows Service wrapper. Для автоматического запуска примените
принятый в организации service manager и направьте stdout/stderr в отдельный
операционный лог. Рабочие логи приложения находятся в output root.

## Docker CUDA 12

Требования: Linux Docker Engine, совместимый NVIDIA driver и NVIDIA Container
Toolkit. На Windows Docker Desktop Linux engine должен быть запущен до build.

```bash
docker build -f docker/rtx4090/Dockerfile \
  -t sara-audio:rtx4090-cuda12 .
```

Проверка модели:

```bash
docker run --rm --gpus all \
  -v sara-models:/models/huggingface \
  --entrypoint sara-diagnose-asr \
  sara-audio:rtx4090-cuda12 \
  --config config/rtx4090_cuda12.yaml --load-model
```

Запуск сервиса:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v /srv/sara/results:/data/output \
  -v sara-models:/models/huggingface \
  sara-audio:rtx4090-cuda12
```

Конфигурации и `models/methodology` встроены в образ. Результаты и Hugging Face
cache должны находиться в томах. Для production зафиксируйте image digest и не
используйте плавающий `latest`.

## CUDA 11.8

CUDA 11.8 использует отдельный Dockerfile и legacy ASR stack:

```bash
docker build -f docker/rtx4090/Dockerfile.cuda11_8 \
  -t sara-audio:rtx4090-cuda11_8 .
```

Не заменяйте один Dockerfile другим после сборки: CUDA runtime выбирается при
build. Профиль определяет одновременно base image, requirements и YAML.

## Обработка каталога без UI

В проверенном Windows CUDA окружении используйте изолированный worker, чтобы
native teardown не маскировал уже записанный результат:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -u -X faulthandler `
  -m sara_audio.web_worker `
  --config .\config\rtx4090_cuda12.yaml `
  --input D:\SARA-input `
  --output D:\SARA-results\run-001
```

Output directory не должен существовать. Worker завершает процесс кодом `0`
после записи файлов. Для обычного Linux/CPU процесса доступен `sara-extract`.

## Обновление server patch

Patch содержит код, requirements, конфигурации, скрипты и MIC models, но не
трогает существующие зависимости:

```powershell
.\scripts\windows\Install-SaraServerPatch.ps1 `
  -ProjectRoot D:\Services\SARA-audio `
  -PatchRoot D:\Incoming\sara-server-patch
```

Порядок безопасного обновления:

1. остановить web-процесс и дождаться текущего job;
2. сохранить commit id, YAML и копию рабочего каталога без results;
3. установить patch;
4. при изменении requirements создать новый venv, не модифицировать старый;
5. выполнить ASR diagnostic и один end-to-end trace;
6. запустить web и проверить upload, preview и ZIP;
7. только после этого переключить reverse proxy.

Rollback выполняется возвратом предыдущего каталога/commit и предыдущего venv.
Результаты пользователей не должны находиться внутри каталога кода.

## Offline

Для полностью воспроизводимой offline-поставки нужны три независимые части:

1. исходный код и `models/methodology`;
2. Python wheelhouse ровно для целевой ОС, Python и CUDA profile;
3. заполненный Hugging Face cache для Whisper и RoSBERTa.

Текущий `New-SaraWheelhouse.ps1` является вспомогательным legacy-скриптом и не
повторяет в точности специальную схему CPU PyTorch + CUDA 12 ASR native installer.
Перед offline production deployment его содержимое нужно сверить с
`requirements-cuda12-native.txt` и installer. Не считайте wheelhouse готовым,
пока smoke-test не выполнен на машине без сети.

## Мониторинг и хранение

- Проверяйте наличие свободного места в output/temp/HF cache.
- Следите за `sara-web.log` и каждым `job-*/processing.log`.
- Сигнал успеха job: корректный `results/run_summary.json`, не только HTTP 200.
- Web сериализует задания внутри одного процесса; очередь Gradio может расти.
- Настройте внешний срок хранения аудио, transcript, ZIP и логов.
- Не удаляйте активный job и не запускайте два сервиса с одним temp/output path.

## Production checklist

- [ ] Зафиксирован commit и checksums 11 MIC JSON.
- [ ] Записаны версии Python, driver, requirements и config.
- [ ] `nvidia-smi` работает от service account.
- [ ] Whisper и RoSBERTa доступны из постоянного cache.
- [ ] Реальный stereo smoke-test прошел полностью.
- [ ] Проверены preview, ZIP, delivery XLSX и SHA-256.
- [ ] TLS и authentication находятся перед Gradio.
- [ ] Output directory закрыт от постороннего чтения.
- [ ] Настроены backup/retention и наблюдение за диском.
- [ ] Задокументирован rollback на предыдущий venv/image.
