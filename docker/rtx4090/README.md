# SARA Docker Service

The CUDA profile must match the runtime supported by the host driver. Build a
separate image for each profile:

| Profile | Image runtime | ASR stack |
| --- | --- | --- |
| `cuda11_8` | CUDA 11.8 + cuDNN 8 | `faster-whisper 0.10.1` + `ctranslate2 3.24.0` |
| `cuda12` | CUDA 12.4 + cuDNN 9 | `faster-whisper 1.2.1` + `ctranslate2 4.8.2` |

Docker cannot change the CUDA runtime inside an already-built image. Select the
profile at build time.

Build the image from the repository root:

```bash
docker build -f docker/rtx4090/Dockerfile.cuda11_8 -t sara-audio:rtx4090-cuda11_8 .
docker build -f docker/rtx4090/Dockerfile -t sara-audio:rtx4090-cuda12 .
```

On Windows PowerShell the helper is:

```powershell
.\docker\rtx4090\build.ps1 -CudaProfile cuda11_8
.\docker\rtx4090\build.ps1 -CudaProfile cuda12
```

Or use the included `docker-compose.yml` after changing the host volume paths:

```bash
# CUDA 12 profile (default)
docker compose -f docker/rtx4090/docker-compose.yml up --build

# CUDA 11.8 profile
SARA_DOCKERFILE=docker/rtx4090/Dockerfile.cuda11_8 \
SARA_IMAGE=sara-audio:rtx4090-cuda11_8 \
SARA_CONFIG_PATH=/app/config/rtx4090_cuda11_8.yaml \
docker compose -f docker/rtx4090/docker-compose.yml up --build
```

Run the Gradio service with NVIDIA GPU access:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v /server/results:/data/output \
  -v /server/hf-cache:/models/huggingface \
  sara-audio:rtx4090-cuda12
```

Open `http://SERVER_IP:7860`, choose several audio files, and click
`Запустить обработку`. The service:

1. processes the whole selected batch with the regular `sara-extract` pipeline;
2. keeps ASR and embedding models warm for the lifetime of the service;
3. shows a small prediction preview;
4. returns a ZIP with all record artifacts and `features_wide.csv`;
5. optionally adds the compact delivery CSV/XLSX package.

Useful overrides:

```bash
# Keep the service in the foreground on another host port
docker run --rm --gpus all \
  -p 8080:7860 \
  -v /server/results:/data/output \
  -v /server/hf-cache:/models/huggingface \
  sara-audio:rtx4090-cuda12

# Override the configuration or output directory
docker run --rm --gpus all \
  -p 7860:7860 \
  -e SARA_CONFIG_PATH=/app/config/rtx4090_cuda12.yaml \
  -e SARA_WEB_OUTPUT_ROOT=/data/output/web-runs \
  -v /server/results:/data/output \
  -v /server/hf-cache:/models/huggingface \
  sara-audio:rtx4090-cuda12
```

The Hugging Face cache mount is strongly recommended because the ASR and
RoSBERTa models are large and should survive container restarts. Each request
is processed sequentially to avoid concurrent GPU memory spikes.

The old one-shot CLI entrypoint remains available for scripted runs:

```bash
docker run --rm --gpus all \
  -v /server/input:/data/input \
  -v /server/results:/data/output \
  -v /server/hf-cache:/models/huggingface \
  --entrypoint sara-server-pipeline \
  sara-audio:rtx4090-cuda12
```
