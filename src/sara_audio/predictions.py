"""JSON linear models and batch prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sara_audio.pipeline import PipelineRun

EXPERT_MODEL_NAMES = (
    "participation_modal_n",
    "greeting_flg",
    "farewell_flg",
    "etiquette_present_flg",
    "etiquette_absent_flg",
    "participation_dominant_flg",
    "distancing_dominant_flg",
    "clarifying_absent_flg",
)

FINAL_PREDICTION_COLUMNS = (
    "file_id",
    "file_name",
    "agent_year_id",
    "burnout",
    "burnout_final",
    "burnout_label",
    "agent_sex_score",
    "agent_sex",
    "agent_sex_label",
    "filter_score",
    "group_input_count",
    "group_informative_count",
)

FILTER_PREDICTION_COLUMNS = (
    "file_id",
    "file_name",
    "agent_year_id",
    "input_path",
    "analysis_audio_path",
    "filter_score",
    "filter_threshold",
    "filter_remove",
    "filter_keep",
    "filter_missing_feature_count",
    "filter_unavailable_feature_count",
    "filter_missing_features",
    "filter_unavailable_features",
)


@dataclass(frozen=True)
class PredictionResult:
    """One model score together with information needed for interpretation."""

    score: float
    threshold: float
    remove: bool
    missing_features: tuple[str, ...]
    unavailable_features: tuple[str, ...]

    @property
    def keep(self) -> bool:
        return not self.remove


class JsonLinearModel:
    """Apply a MIC-style linear model stored as ``feature -> coefficient`` JSON."""

    def __init__(self, coefficients: Mapping[str, float], source_path: Path) -> None:
        if "Intercept" not in coefficients:
            raise ValueError(f"Model {source_path} must contain an Intercept")
        self.coefficients = dict(coefficients)
        self.source_path = source_path
        self.model_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: Path) -> JsonLinearModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Model {path} must contain a JSON object")
        coefficients: dict[str, float] = {}
        for feature, coefficient in payload.items():
            if not isinstance(feature, str):
                raise ValueError(f"Model {path} contains a non-string feature name")
            try:
                value = float(coefficient)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Model {path} has a non-numeric coefficient for {feature}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"Model {path} has a non-finite coefficient for {feature}")
            coefficients[feature] = value
        return cls(coefficients, path)

    def predict(
        self,
        features: Mapping[str, Any],
        *,
        threshold: float = 0.5,
        round_score: bool = True,
    ) -> PredictionResult:
        score = self.coefficients["Intercept"]
        missing: list[str] = []
        unavailable: list[str] = []
        for feature, weight in self.coefficients.items():
            if feature == "Intercept":
                continue
            if feature not in features:
                missing.append(feature)
                continue
            value = features[feature]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                unavailable.append(feature)
                continue
            if not math.isfinite(numeric):
                unavailable.append(feature)
                continue
            score += numeric * weight
        threshold_score = round(score, 8)
        return PredictionResult(
            score=threshold_score if round_score else score,
            threshold=threshold,
            remove=threshold_score >= threshold,
            missing_features=tuple(sorted(missing)),
            unavailable_features=tuple(sorted(unavailable)),
        )


@dataclass(frozen=True)
class MethodologyModelBundle:
    """Filter, reconstructed expert labels, sex, and burnout models."""

    filter_model: JsonLinearModel
    expert_models: tuple[tuple[str, JsonLinearModel], ...]
    agent_sex_model: JsonLinearModel
    burnout_model: JsonLinearModel

    @classmethod
    def load(
        cls,
        filter_model_path: Path,
        model_directory: Path,
    ) -> MethodologyModelBundle:
        return cls(
            filter_model=JsonLinearModel.load(filter_model_path),
            expert_models=tuple(
                (name, JsonLinearModel.load(model_directory / f"{name}.json"))
                for name in EXPERT_MODEL_NAMES
            ),
            agent_sex_model=JsonLinearModel.load(model_directory / "agent_sex.json"),
            burnout_model=JsonLinearModel.load(model_directory / "burnout.json"),
        )


@dataclass(frozen=True)
class MethodologyPredictionRows:
    """Three output tables produced by the complete methodology cascade."""

    final: list[dict[str, object]]
    interpretation: list[dict[str, object]]
    filtering: list[dict[str, object]]


def run_prediction_rows(
    runs: list[PipelineRun],
    model: JsonLinearModel,
    *,
    threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return compact predictions and the full interpretation input table."""

    compact: list[dict[str, object]] = []
    full: list[dict[str, object]] = []
    for run in runs:
        features = {feature.code: feature.value for feature in run.result.features}
        result = model.predict(features, threshold=threshold)
        identity = {
            "file_id": run.result.file_id,
            "input_path": str(run.result.input_path),
            "analysis_audio_path": str(run.analysis_audio_path),
        }
        compact.append(
            {
                **identity,
                "filter_score": result.score,
                "filter_threshold": result.threshold,
                "filter_remove": int(result.remove),
                "filter_keep": int(result.keep),
            }
        )
        full_row: dict[str, object] = {
            **identity,
            "filter_score": result.score,
            "filter_threshold": result.threshold,
            "filter_remove": int(result.remove),
            "filter_keep": int(result.keep),
            "filter_missing_feature_count": len(result.missing_features),
            "filter_unavailable_feature_count": len(result.unavailable_features),
            "filter_missing_features": list(result.missing_features),
            "filter_unavailable_features": list(result.unavailable_features),
        }
        full_row.update(features)
        full.append(full_row)
    return compact, full


def run_methodology_prediction_rows(
    runs: list[PipelineRun],
    models: MethodologyModelBundle,
    *,
    threshold: float,
) -> MethodologyPredictionRows:
    """Apply the methodology cascade without changing feature extraction."""

    groups: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    filter_rows: list[dict[str, object]] = []

    for run in runs:
        features = {feature.code: feature.value for feature in run.result.features}
        file_name = run.result.input_path.name
        group_id = agent_year_id(file_name)
        identity: dict[str, object] = {
            "file_id": run.result.file_id,
            "file_name": file_name,
            "agent_year_id": group_id,
            "input_path": str(run.result.input_path),
            "analysis_audio_path": str(run.analysis_audio_path),
        }
        filter_result = models.filter_model.predict(
            features,
            threshold=threshold,
            round_score=False,
        )
        filter_rows.append(
            {
                **identity,
                "filter_score": filter_result.score,
                "filter_threshold": filter_result.threshold,
                "filter_remove": int(filter_result.remove),
                "filter_keep": int(filter_result.keep),
                "filter_missing_feature_count": len(filter_result.missing_features),
                "filter_unavailable_feature_count": len(
                    filter_result.unavailable_features
                ),
                "filter_missing_features": list(filter_result.missing_features),
                "filter_unavailable_features": list(
                    filter_result.unavailable_features
                ),
            }
        )
        if filter_result.remove:
            continue

        enriched = dict(features)
        expert_diagnostics: dict[str, object] = {}
        for expert_name, expert_model in models.expert_models:
            expert_result = expert_model.predict(features, round_score=False)
            enriched[expert_name] = expert_result.score
            expert_diagnostics[f"{expert_name}_missing_feature_count"] = len(
                expert_result.missing_features
            )
            expert_diagnostics[f"{expert_name}_unavailable_feature_count"] = len(
                expert_result.unavailable_features
            )
        groups.setdefault(group_id, []).append(
            (
                {
                    **identity,
                    "filter_score": filter_result.score,
                    **enriched,
                    **expert_diagnostics,
                },
                enriched,
            )
        )

    final_rows: list[dict[str, object]] = []
    interpretation_rows: list[dict[str, object]] = []
    group_input_counts: dict[str, int] = {}
    for row in filter_rows:
        group_id = str(row["agent_year_id"])
        group_input_counts[group_id] = group_input_counts.get(group_id, 0) + 1

    for group_id, records in groups.items():
        means = _numeric_means([features for _, features in records])
        sex_result = models.agent_sex_model.predict(means, round_score=False)
        sex = int(round(sex_result.score, 8) > 0.5)
        means["m"] = sex
        means["f"] = 1 - sex
        burnout_result = models.burnout_model.predict(means, round_score=False)
        burnout = burnout_result.score
        burnout_final = int(round(burnout, 8) > 0.5)
        group_input_count = group_input_counts[group_id]
        informative_count = len(records)

        for full_row, _ in records:
            prediction = {
                "burnout": burnout,
                "burnout_final": burnout_final,
                "burnout_label": "Да" if burnout_final else "Нет",
                "agent_sex_score": sex_result.score,
                "agent_sex": sex,
                "agent_sex_label": "M" if sex else "F",
                "m": sex,
                "f": 1 - sex,
                "group_input_count": group_input_count,
                "group_informative_count": informative_count,
                "agent_sex_missing_feature_count": len(
                    sex_result.missing_features
                ),
                "agent_sex_unavailable_feature_count": len(
                    sex_result.unavailable_features
                ),
                "burnout_missing_feature_count": len(
                    burnout_result.missing_features
                ),
                "burnout_unavailable_feature_count": len(
                    burnout_result.unavailable_features
                ),
            }
            interpretation_rows.append({**full_row, **prediction})
            final_rows.append(
                {column: ({**full_row, **prediction}).get(column, "")
                 for column in FINAL_PREDICTION_COLUMNS}
            )

    return MethodologyPredictionRows(
        final=final_rows,
        interpretation=interpretation_rows,
        filtering=filter_rows,
    )


def agent_year_id(file_name: str) -> str:
    """Return the methodologists' operator-year key from an original filename."""

    stem = Path(file_name).stem.removesuffix("_muted")
    timestamp, separator, agent = stem.partition("_")
    if not separator or len(timestamp) < 4 or not timestamp[:4].isdigit():
        return "uploaded_batch"
    for prefix in ("Оператор", "Operator", "Agent_", "Agent"):
        if agent.casefold().startswith(prefix.casefold()):
            agent = agent[len(prefix):]
            break
    agent = agent.replace(".", "").strip("_ ") or "unknown"
    return f"{agent.zfill(2)}_{timestamp[:4]}"


def _numeric_means(rows: list[dict[str, object]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for feature, value in row.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.setdefault(feature, []).append(numeric)
    return {
        feature: sum(feature_values) / len(feature_values)
        for feature, feature_values in values.items()
    }
