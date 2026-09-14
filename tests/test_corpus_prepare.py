import zipfile
from pathlib import Path

import pytest

from sara_audio.corpus_prepare import CorpusPreparer


def test_preparer_combines_train_and_extracted_test_wavs(tmp_path: Path) -> None:
    source = tmp_path / "2_file"
    train = source / "train"
    train.mkdir(parents=True)
    (train / "train.wav").write_bytes(b"train")
    with zipfile.ZipFile(source / "test.zip", "w") as archive:
        archive.writestr("test/test.wav", b"test")

    prepared = CorpusPreparer().prepare(source)

    assert prepared.train_wav_count == 1
    assert prepared.test_wav_count == 1
    assert (source / "test" / "test.wav").read_bytes() == b"test"


def test_preparer_rejects_unsafe_test_archive_paths(tmp_path: Path) -> None:
    source = tmp_path / "2_file"
    (source / "train").mkdir(parents=True)
    (source / "train" / "train.wav").write_bytes(b"train")
    with zipfile.ZipFile(source / "test.zip", "w") as archive:
        archive.writestr("../escape.wav", b"bad")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        CorpusPreparer().prepare(source)
