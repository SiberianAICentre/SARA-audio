# Windows Server: copy, smoke-test, calculate

No Git checkout is needed. Copy the complete project directory to the server,
including `config`, `src`, `scripts`, `pyproject.toml` and this document. If the
supplied corpus is copied unchanged, leave it in `data\new_raw_data\2_file`:
the script finds `train\` and safely extracts `test.zip` automatically. If the
corpus is supplied as a ready directory of audio files instead, place it in
`data\raw`.

Before the first run the server needs only two machine-level prerequisites:

1. NVIDIA driver visible through `nvidia-smi`.
2. Python 3.11 with the Windows `py` launcher.
The script first uses the CUDA environment already configured for other server
software. If `CUDA_PATH` is absent, it automatically detects the latest toolkit
under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`. A project-local
`cuda-libs` directory or `-CudaLibraryPath` is only an override for a broken or
nonstandard server environment.

From PowerShell in the project directory run one command:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\windows\Invoke-SaraServerPipeline.ps1
```

The script creates `.server-venv`, installs the project and its dependencies, loads
Whisper on CUDA, prepares the supplied `train + test.zip` corpus when present,
selects the first input recording for `sara-server-smoke`, and only then
calculates the complete corpus. It writes a timestamped result in
`results\server-...` and a sibling smoke result. The first run downloads
`large-v3` into the model cache.

The script does not upgrade `pip`, uses a 600-second download timeout, and
does not reuse a copied local `.venv`. If the server cannot reach PyPI, it stops
before any corpus calculation; use `-Offline` only after copying a prepared
`wheelhouse` directory.

For a server without internet, build the package set once on a computer with
internet access:

```powershell
.\scripts\windows\New-SaraWheelhouse.ps1
```

Copy the resulting `wheelhouse` directory alongside the project and append
`-Offline`. The model cache must also be copied or made available to the server
before the first smoke run.
