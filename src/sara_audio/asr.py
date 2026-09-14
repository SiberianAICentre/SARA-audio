"""Optional Russian ASR backend using faster-whisper timestamps."""

from __future__ import annotations

import os
import re
import site
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sara_audio.domain import Transcript, TranscriptToken
from sara_audio.errors import DependencyUnavailableError
from sara_audio.interfaces import AsrTranscriber


class WindowsCudaLibraryPaths:
    """Make CUDA runtime DLLs installed in a virtual environment discoverable."""

    _directory_handles: list[object] = []
    _configured = False

    @classmethod
    def configure(cls) -> None:
        """Add CUDA and NVIDIA wheel directories before loading CTranslate2."""
        if os.name != "nt" or cls._configured:
            return

        candidates = cls._candidate_directories()
        existing_paths = os.environ.get("PATH", "").split(os.pathsep)
        normalized_paths = {path.casefold() for path in existing_paths}
        for directory in candidates:
            directory_text = str(directory)
            if directory_text.casefold() not in normalized_paths:
                os.environ["PATH"] = directory_text + os.pathsep + os.environ.get(
                    "PATH", ""
                )
                normalized_paths.add(directory_text.casefold())
            cls._directory_handles.append(os.add_dll_directory(directory_text))
        cls._configured = True

    @classmethod
    def has_cudnn_runtime(cls) -> bool:
        """Return whether a supported CUDA 11/12 cuDNN runtime is visible."""
        search_directories = list(cls._candidate_directories())
        search_directories.extend(
            Path(entry)
            for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry
        )
        return any(
            (directory / library).is_file()
            for directory in search_directories
            for library in ("cudnn64_8.dll", "cudnn64_9.dll", "cudnn_ops_infer64_9.dll")
        )

    @staticmethod
    def _candidate_directories() -> tuple[Path, ...]:
        candidates: list[Path] = []
        bundled_cuda_libraries = os.environ.get("SARA_CUDA_LIB_DIR")
        if bundled_cuda_libraries:
            candidates.append(Path(bundled_cuda_libraries))
        cuda_root = os.environ.get("CUDA_PATH")
        if cuda_root:
            candidates.append(Path(cuda_root) / "bin")
        for package_root in site.getsitepackages():
            nvidia_root = Path(package_root) / "nvidia"
            candidates.extend(
                (
                    nvidia_root / "cudnn" / "bin",
                    nvidia_root / "cublas" / "bin",
                    nvidia_root / "cuda_nvrtc" / "bin",
                )
            )
        return tuple(directory for directory in candidates if directory.is_dir())


@dataclass(frozen=True)
class FasterWhisperSettings:
    model: str = "large-v3"
    language: str = "ru"
    device: str = "auto"
    compute_type: str = "int8"
    initial_prompt: str | None = None
    vad_filter: bool = False
    vad_min_speech_duration_ms: int = 250
    vad_max_speech_duration_s: float = 15.0
    vad_speech_pad_ms: int = 600
    no_speech_threshold: float | None = None
    compression_ratio_threshold: float | None = None
    temperature: tuple[float, ...] | None = None
    suppress_common_hallucinations: bool = True


class FasterWhisperTranscriber(AsrTranscriber):
    """Transcribe audio and preserve the backend's word-level timestamps."""

    _HALLUCINATION_PATTERN = re.compile(
        r"(прис|продолжение\s+следует|субтитры\s+сделал|"
        r"спасибо\s+за\s+субтитры\s+алексею\s+дубровскому|"
        r"субтитры\s+создавал|dimatorzok|подписывайтесь\s+на\s+канал|"
        r"спасибо\s+за\s+просмотр)",
        re.IGNORECASE,
    )

    def __init__(self, settings: FasterWhisperSettings | None = None) -> None:
        self._settings = settings or FasterWhisperSettings()
        self._model: object | None = None

    def transcribe(self, input_path: Path) -> Transcript:
        model = self.load_model()
        segments, _info = model.transcribe(str(input_path), **self._transcribe_kwargs())
        texts: list[str] = []
        tokens: list[TranscriptToken] = []
        for segment in segments:
            cleaned = self._clean_segment_text(segment.text)
            if not cleaned:
                continue
            texts.append(cleaned)
            for word in segment.words or ():
                tokens.append(
                    TranscriptToken(
                        text=word.word,
                        start_s=word.start,
                        end_s=word.end,
                        confidence=word.probability,
                    )
                )
        return Transcript(
            text=" ".join(part for part in texts if part),
            tokens=tuple(tokens),
            source="faster-whisper",
            is_verified=False,
        )

    def _transcribe_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "language": self._settings.language,
            "word_timestamps": True,
        }
        if self._settings.initial_prompt:
            kwargs["initial_prompt"] = self._settings.initial_prompt
        if self._settings.vad_filter:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = {
                "min_speech_duration_ms": self._settings.vad_min_speech_duration_ms,
                "max_speech_duration_s": self._settings.vad_max_speech_duration_s,
                "speech_pad_ms": self._settings.vad_speech_pad_ms,
            }
        if self._settings.no_speech_threshold is not None:
            kwargs["no_speech_threshold"] = self._settings.no_speech_threshold
        if self._settings.compression_ratio_threshold is not None:
            kwargs["compression_ratio_threshold"] = (
                self._settings.compression_ratio_threshold
            )
        if self._settings.temperature is not None:
            kwargs["temperature"] = list(self._settings.temperature)
        return kwargs

    def _clean_segment_text(self, text: str) -> str:
        cleaned = text.strip()
        if self._settings.suppress_common_hallucinations:
            cleaned = self._HALLUCINATION_PATTERN.sub("", cleaned).strip()
        if re.fullmatch(r"[\s.,\-?!]+", cleaned):
            return ""
        return cleaned

    def load_model(self) -> object:
        """Load the configured model so deployment can verify the real device."""
        try:
            WindowsCudaLibraryPaths.configure()
            if (
                self._settings.device == "cuda"
                and os.name == "nt"
                and not WindowsCudaLibraryPaths.has_cudnn_runtime()
            ):
                raise DependencyUnavailableError(
                    "CUDA ASR requires a visible cuBLAS and cuDNN runtime. "
                    "Set SARA_CUDA_LIB_DIR to the project cuda-libs directory."
                )
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise DependencyUnavailableError(
                "ASR dependencies are unavailable: "
                f"{error}. See CUDA_11_8_SETUP.md."
            ) from error

        if self._model is None:
            self._model = WhisperModel(
                self._settings.model,
                device=self._settings.device,
                compute_type=self._settings.compute_type,
            )
        return self._model


def asr_runtime_diagnostics(
    settings: FasterWhisperSettings,
    load_model: bool = False,
) -> dict[str, object]:
    """Report CUDA visibility and optionally load the model on its configured device."""
    payload: dict[str, object] = {
        "configured_device": settings.device,
        "configured_compute_type": settings.compute_type,
        "configured_model": settings.model,
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["nvidia_smi"] = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        payload["nvidia_smi_error"] = str(error)

    try:
        import ctranslate2

        payload["ctranslate2_version"] = ctranslate2.__version__
        get_count = getattr(ctranslate2, "get_cuda_device_count", None)
        if get_count is not None:
            payload["cuda_device_count"] = get_count()
    except ImportError as error:
        payload["ctranslate2_error"] = str(error)

    if load_model:
        FasterWhisperTranscriber(settings).load_model()
        payload["model_loaded"] = True
        payload["model_device"] = settings.device
    return payload
