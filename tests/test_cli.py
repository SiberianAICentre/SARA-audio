from pathlib import Path

from sara_audio.cli import build_extract_parser
from sara_audio.config import ApplicationSettings


def test_extract_cli_accepts_an_explicit_server_config() -> None:
    args = build_extract_parser().parse_args(
        ["input", "output", "--config", "config/rtx4090_cuda12.yaml"]
    )
    assert args.config == Path("config/rtx4090_cuda12.yaml")


def test_rtx4090_profile_uses_cuda_large_v3_and_batches_of_fifteen() -> None:
    settings = ApplicationSettings.from_yaml(Path("config/rtx4090_cuda12.yaml"))
    assert settings.execution.asr.model == "large-v3"
    assert settings.execution.asr.device == "cuda"
    assert settings.execution.asr.compute_type == "float16"
    assert settings.execution.batch_size == 15
    assert settings.pipeline.text_embeddings.enabled
