# Whisper on CUDA 11.8

## Verified local configuration

- GPU: NVIDIA GeForce GTX 1650, 4 GB VRAM.
- Driver: 522.06, reporting CUDA 11.8.
- Toolkit: CUDA 11.8 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`.
- Pipeline configuration: `small`, `cuda`, `int8_float16` in `config/default.yaml`.

The current `faster-whisper` release needs CUDA 12. CUDA 11.8 instead uses
`faster-whisper 0.10.1` and `ctranslate2 3.24.0`, which need cuDNN 8.

## Python packages

The compatible Python packages have already been installed in this project's
`.venv`. To recreate the environment, run these commands from the repository
root:

```powershell
.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  "faster-whisper==0.10.1" "ctranslate2==3.24.0"
.venv\Scripts\python.exe -m pip install `
  "setuptools<81" "requests>=2.31" "tokenizers>=0.13,<0.16"
```

`PyAV 10` has no usable Windows wheel for this Python 3.11 environment.
The installed modern `av` package is used only to decode audio and was verified
with the project input.

## Required cuDNN 8 runtime

CUDA Toolkit does not include cuDNN. Install a Windows x86-64 cuDNN 8 package
for CUDA 11, then make `cudnn64_8.dll` available. The practical option is the
NVIDIA wheel:

```powershell
.venv\Scripts\python.exe -m pip install "nvidia-cudnn-cu11==8.9.5.29"
```

It is a large download (about 712 MB). If the local network blocks this wheel,
download an official cuDNN 8 Windows package for CUDA 11 from NVIDIA and place
its `bin` directory on `PATH` before starting the terminal or IDE.

`WindowsCudaLibraryPaths` in `src/sara_audio/asr.py` automatically adds the
CUDA Toolkit and NVIDIA wheel `bin` directories during ASR startup, so a cuDNN
wheel installed into `.venv` needs no manual `PATH` change.

## Verification

After cuDNN is present, run:

```powershell
.venv\Scripts\python.exe -c "from sara_audio.asr import WindowsCudaLibraryPaths; WindowsCudaLibraryPaths.configure(); from faster_whisper import WhisperModel; model = WhisperModel('small', device='cuda', compute_type='int8_float16'); segments, _ = model.transcribe('data\\raw\\А.m4a', language='ru'); print(next(iter(segments)).text)"
```

Then use the regular pipeline command; no ASR arguments are required:

```powershell
sara-extract data\raw results\gpu-run
```
