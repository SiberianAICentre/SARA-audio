# RTX 4090 / CUDA 12 deployment

The server profile is [config/rtx4090_cuda12.yaml](config/rtx4090_cuda12.yaml):
`large-v3`, `cuda`, `float16`, the second source channel, and batches of 15.

The supplied Docker image is based on CUDA 12.4 with cuDNN 9. Current
faster-whisper/CTranslate2 GPU builds require CUDA 12 cuBLAS and cuDNN 9; the
legacy CUDA 11.8 instructions in `CUDA_11_8_SETUP.md` are not for this server.

Build from the repository root:

```bash
docker build -f docker/rtx4090/Dockerfile -t sara-audio:rtx4090 .
```

First verify that the container sees the GPU and can load Whisper. The first
run downloads `large-v3` and RoSBERTa; mount `/models/huggingface` so they are
cached:

```bash
docker run --rm --gpus all -v sara-models:/models/huggingface \
  --entrypoint sara-diagnose-asr sara-audio:rtx4090 \
  --config config/rtx4090_cuda12.yaml --load-model
```

The diagnostic must report a non-empty `nvidia_smi`, `cuda_device_count >= 1`,
`model_loaded: true`, and `model_device: cuda`. The normal service is then
available at `http://SERVER_IP:7860`:

```bash
docker run --rm --gpus all -p 7860:7860 \
  -v /srv/sara/output:/data/output \
  -v sara-models:/models/huggingface \
  sara-audio:rtx4090
```

Upload a batch of audio files in the Gradio page. It calculates the regular
pipeline, keeps models cached for the lifetime of the service, and returns a
ZIP with full artifacts plus the filtered `features_wide.csv`.

For a separate preflight smoke test on a real stereo recording with operator
speech on channel 2, use:

```bash
docker run --rm --gpus all \
  -v /srv/sara/smoke-input:/input:ro -v /srv/sara/smoke-output:/output \
  -v sara-models:/models/huggingface \
  --entrypoint sara-server-smoke sara-audio:rtx4090 \
  /input /output --config config/rtx4090_cuda12.yaml
```

Only a JSON result with `"status": "passed"` authorizes production use.
The old one-shot corpus runner remains available for scripted directory runs:

```bash
docker run --rm --gpus all \
  -v /srv/sara/input:/data/input \
  -v /srv/sara/output:/data/output \
  -v sara-models:/models/huggingface \
  --entrypoint sara-server-pipeline \
  sara-audio:rtx4090
```

The optional checkbox in the UI adds CSV/XLSX delivery tables to the download.
