# SARA-audio

SARA-audio is a reproducible batch service for extracting acoustic,
temporal, ASR, linguistic, and text-embedding features from contact-center
audio. The complete profile applies the MIC methodology cascade and returns
filter, reconstructed expert attributes, operator sex, and burnout predictions.

The repository contains a Python package, CLI tools, a Gradio web interface,
Windows deployment scripts, and Docker profiles for CUDA 11.8 and CUDA 12.

## What the service does

For each uploaded recording the pipeline:

1. selects the configured operator channel and converts audio to mono 16 kHz PCM;
2. detects speech and pauses with the built-in energy VAD;
3. writes an operator speech-only WAV;
4. transcribes speech with faster-whisper;
5. extracts openSMILE, temporal, ASR, MIC, linguistic, and RoSBERTa features;
6. applies the MIC filter and reconstructs eight expert attributes;
7. aggregates accepted recordings by operator and year;
8. predicts operator sex and burnout;
9. writes auditable per-record artifacts, batch tables, and an optional delivery ZIP.

The web page shows the final prediction immediately and still lets the user
download all detailed results.

## Repository map

```text
src/sara_audio/           Python package and CLI entry points
src/sara_audio/data/      Versioned linguistic feature registry
config/                   CPU and CUDA runtime profiles
models/methodology/       Canonical MIC JSON coefficient files
meth_patch/               Original methodology handoff files (provenance)
scripts/windows/          Native Windows install, launch, trace, and patch scripts
docker/rtx4090/           CUDA 11.8 and CUDA 12 container definitions
tests/                    Unit and integration-style tests with mocked heavy models
docs/                     Maintainer and deployment documentation
```

## Recommended production path

The currently validated native Windows path is Python 3.11 with the CUDA 12
profile. From PowerShell in the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Install-SaraCuda12Native.ps1
```

The installer creates a new `.cuda12-venv-final` environment, installs the
pinned native stack, verifies imports, and loads Whisper on the GPU. It stops
if that environment already exists. Start the web service:

```powershell
.\.cuda12-venv-final\Scripts\python.exe -m sara_audio.webapp `
  --config .\config\rtx4090_cuda12.yaml `
  --output-root D:\SARA-results `
  --host 0.0.0.0 `
  --port 7860
```

Open `http://SERVER_IP:7860`. Do not expose this development server directly
to the public internet; put authentication and TLS in a reverse proxy.

Before a production run, trace one real stereo file end to end:

```powershell
.\scripts\windows\Trace-SaraPipeline.ps1 `
  -InputPath D:\SARA-input\smoke.wav `
  -VenvPath .\.cuda12-venv-final `
  -ConfigPath .\config\rtx4090_cuda12.yaml
```

The trace log contains `BEGIN` and `END` markers for every native and model
stage. A successful worker exits with code `0` and writes `run_summary.json`.

## Local development

Python 3.11 is the reference interpreter. A lightweight development install:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,nlp,web]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\sara-validate-repository.exe
```

Install `asr`, `embeddings`, and `parquet` extras only when those stages are
needed. For native Windows CUDA, use the pinned installer instead of combining
extras manually.

Process one file or a directory:

```powershell
.\.venv\Scripts\sara-extract.exe .\input .\artifacts\run-001 `
  --config .\config\native_cpu.yaml
```

Supported inputs are WAV, M4A, MP3, FLAC, and OGG. `imageio-ffmpeg` supplies a
bundled ffmpeg binary; a system `ffmpeg` in `PATH` takes precedence.

## Runtime profiles

| Profile | Intended use | ASR | Embeddings |
| --- | --- | --- | --- |
| `config/native_cpu.yaml` | CPU smoke/debug | `small`, CPU int8 | disabled |
| `config/rtx4090_cuda12.yaml` | validated Windows/Docker production | `large-v3`, CUDA float16 | CPU |
| `config/rtx4090_cuda11_8.yaml` | legacy/reserve CUDA 11.8 | `large-v3`, CUDA float16 | CPU |
| `config/default.yaml` | generic editable-install baseline | configurable CUDA | CPU |

CUDA 11 and CUDA 12 dependencies are intentionally isolated. Do not install
both requirement profiles into one virtual environment.

## Models and first-run downloads

The 11 small MIC coefficient files are versioned in `models/methodology/`.
Whisper `large-v3` and `ai-forever/ru-en-RoSBERTa` are downloaded from Hugging
Face on first use. Set `HF_HOME` to a persistent directory on servers and copy
that cache separately for an offline deployment.

Model coefficients are applied directly to the feature values named by each
JSON key. The current JSON format does not contain scaler parameters, so the
service does not standardize features before prediction. This contract must be
confirmed with methodology owners whenever models are replaced.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [CLI and scripts](docs/CLI.md)
- [MIC methodology and model contract](docs/METHODOLOGY.md)
- [Output and delivery formats](docs/OUTPUTS.md)
- [Deployment and operations](docs/DEPLOYMENT.md)
- [Development and contribution workflow](docs/DEVELOPMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [User guide](USER_GUIDE.md)
- [Detailed feature pipeline](PIPELINE_DESCRIPTION.md)
- [Windows quick start](WINDOWS_SERVER_QUICKSTART.md)

## Important limitations

- Predictions are methodological outputs, not medical diagnoses.
- The grouping key is derived from the filename. Unstructured names fall into
  `uploaded_batch`, which can combine unrelated operators.
- A stereo source is expected to contain the operator on zero-based channel
  index `1`; mono input is accepted with a diagnostic by default.
- The Gradio server has no built-in authentication, authorization, retention
  policy, or TLS termination.
- No open-source license is currently declared. External redistribution is not
  permitted until the repository owner adds an explicit license.
