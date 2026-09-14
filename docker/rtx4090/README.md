# Docker-развертывание SARA

CUDA-профиль должен соответствовать runtime, выбранному для образа. Для CUDA
11.8 и CUDA 12 собираются разные images.

| Профиль | Базовый runtime | ASR stack |
| --- | --- | --- |
| `cuda11_8` | CUDA 11.8 + cuDNN 8 | faster-whisper 0.10.1, CTranslate2 3.24.0 |
| `cuda12` | CUDA 12.4 + cuDNN 9 | faster-whisper 1.2.1, CTranslate2 4.8.2 |

Runtime нельзя переключить внутри уже собранного image. Выбор выполняется на
этапе build.

## Сборка

Из корня репозитория:

```bash
docker build -f docker/rtx4090/Dockerfile \
  -t sara-audio:rtx4090-cuda12 .

docker build -f docker/rtx4090/Dockerfile.cuda11_8 \
  -t sara-audio:rtx4090-cuda11_8 .
```

PowerShell helper:

```powershell
.\docker\rtx4090\build.ps1 -CudaProfile cuda12
.\docker\rtx4090\build.ps1 -CudaProfile cuda11_8
```

Перед сборкой `docker version` должен показывать и Client, и Server. На Windows
необходимо запустить Docker Desktop Linux engine.

## Проверка GPU и Whisper

```bash
docker run --rm --gpus all \
  -v sara-models:/models/huggingface \
  --entrypoint sara-diagnose-asr \
  sara-audio:rtx4090-cuda12 \
  --config config/rtx4090_cuda12.yaml --load-model
```

Ожидаются `cuda_device_count >= 1`, `model_loaded: true` и
`model_device: cuda`. Это проверяет только ASR; перед production нужен полный
smoke-test на реальном аудио.

## Запуск web-сервиса

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v /srv/sara/results:/data/output \
  -v sara-models:/models/huggingface \
  sara-audio:rtx4090-cuda12
```

Откройте `http://SERVER_IP:7860`. Входные задания обрабатываются последовательно,
каждый pipeline запускается в отдельном worker process. Файлы Whisper и RoSBERTa
повторно используются из постоянного Hugging Face cache.

Каталог `models/methodology` и конфигурации входят в image. Results и model cache
должны быть подключены томами.

## Docker Compose

Измените host paths в `docker-compose.yml`, затем:

```bash
docker compose -f docker/rtx4090/docker-compose.yml up --build
```

Для CUDA 11.8:

```bash
SARA_DOCKERFILE=docker/rtx4090/Dockerfile.cuda11_8 \
SARA_IMAGE=sara-audio:rtx4090-cuda11_8 \
SARA_CONFIG_PATH=/app/config/rtx4090_cuda11_8.yaml \
docker compose -f docker/rtx4090/docker-compose.yml up --build
```

## Полный smoke-test

```bash
docker run --rm --gpus all \
  -v /srv/sara/smoke-input:/input:ro \
  -v /srv/sara/smoke-output:/output \
  -v sara-models:/models/huggingface \
  --entrypoint sara-server-smoke \
  sara-audio:rtx4090-cuda12 \
  /input /output --config config/rtx4090_cuda12.yaml
```

Production-развертывание допускается только после результата `status: passed`.
Подробные требования по безопасности, offline-поставке и эксплуатации находятся
в [общем документе развертывания](../../docs/DEPLOYMENT.md).
