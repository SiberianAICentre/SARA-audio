"""Safe preparation of the supplied train directory and test ZIP corpus."""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class PreparedCorpus:
    source_root: Path
    input_root: Path
    train_wav_count: int
    test_wav_count: int


class CorpusPreparer:
    """Extract only test WAVs and reject archive path traversal."""

    def prepare(self, source_root: Path) -> PreparedCorpus:
        source_root = source_root.resolve()
        train_directory = source_root / "train"
        archive = source_root / "test.zip"
        test_directory = source_root / "test"
        if not train_directory.is_dir():
            raise FileNotFoundError(f"Train directory is missing: {train_directory}")
        if not archive.is_file():
            raise FileNotFoundError(f"Test archive is missing: {archive}")
        if not test_directory.exists():
            self._extract_test_wavs(archive, test_directory)
        train_count = self._wav_count(train_directory)
        test_count = self._wav_count(test_directory)
        if not train_count or not test_count:
            raise RuntimeError("Prepared corpus must contain both train and test WAV files")
        return PreparedCorpus(source_root, source_root, train_count, test_count)

    def _extract_test_wavs(self, archive: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.extracting")
        if temporary.exists():
            raise FileExistsError(f"Incomplete extraction directory exists: {temporary}")
        temporary.mkdir()
        try:
            with zipfile.ZipFile(archive) as source:
                members = [
                    member
                    for member in source.infolist()
                    if not member.is_dir() and member.filename.lower().endswith(".wav")
                ]
                if not members:
                    raise RuntimeError(f"No WAV files found in {archive}")
                for member in members:
                    relative = self._safe_relative_path(member.filename)
                    output_path = temporary / relative
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as input_file, output_path.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
            temporary.replace(destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    @staticmethod
    def _safe_relative_path(name: str) -> Path:
        parts = PurePosixPath(name).parts
        if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
            raise ValueError(f"Unsafe archive member path: {name}")
        if parts[0].casefold() == "test":
            parts = parts[1:]
        if not parts:
            raise ValueError(f"Archive member has no output filename: {name}")
        return Path(*parts)

    @staticmethod
    def _wav_count(directory: Path) -> int:
        return sum(
            path.is_file() and path.suffix.lower() == ".wav"
            for path in directory.rglob("*")
        )
