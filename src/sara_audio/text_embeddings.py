"""Optional transformer text embeddings for operator transcripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sara_audio.domain import FeatureValue, Transcript
from sara_audio.errors import DependencyUnavailableError


@dataclass(frozen=True)
class TextEmbeddingSettings:
    """Configuration for dense text-vector extraction."""

    enabled: bool = False
    model_name: str = "ai-forever/ru-en-RoSBERTa"
    device: str = "auto"
    max_length: int = 512
    feature_prefix: str = "bert_"

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("text_embeddings.max_length must be positive")
        if not self.feature_prefix:
            raise ValueError("text_embeddings.feature_prefix cannot be empty")


class TextEmbeddingExtractor:
    """Mean-pool the last hidden state, matching the MIC notebook approach."""

    method_version = "transformer-mean-pooling-v1"

    def __init__(self, settings: TextEmbeddingSettings | None = None) -> None:
        self._settings = settings or TextEmbeddingSettings()
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._torch: object | None = None
        self._device: str | None = None
        self._embedding_size: int | None = None

    def extract(self, transcript: Transcript) -> tuple[FeatureValue, ...]:
        vector = self._vector(transcript.text)
        evidence = {
            "transcript_source": transcript.source,
            "model_name": self._settings.model_name,
            "pooling": "attention-mask mean over last hidden state",
            "max_length": self._settings.max_length,
            "device": self._device,
        }
        return tuple(
            FeatureValue(
                code=f"{self._settings.feature_prefix}{index}",
                value=float(value),
                unit="embedding",
                source="transformer-text-embedding",
                confidence=None,
                method_version=self.method_version,
                evidence=evidence,
            )
            for index, value in enumerate(vector, start=1)
        )

    def _vector(self, text: str) -> np.ndarray:
        tokenizer, model, torch = self._load()
        if not isinstance(text, str) or not text.strip():
            return np.zeros(self._embedding_size or 0, dtype=np.float32)
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._settings.max_length,
                padding=True,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return (summed / counts).squeeze(0).cpu().numpy()

    def _load(self) -> tuple[object, object, object]:
        if self._tokenizer is not None and self._model is not None and self._torch:
            return self._tokenizer, self._model, self._torch
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise DependencyUnavailableError(
                "Text embeddings require optional dependencies: install .[embeddings]."
            ) from error
        device = self._settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self._settings.model_name)
        model = AutoModel.from_pretrained(self._settings.model_name).to(device).eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._device = device
        self._embedding_size = int(model.config.hidden_size)
        return tokenizer, model, torch
