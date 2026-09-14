"""Machine-validated registry for the 92 DOCX-defined linguistic features."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from importlib.resources import files

from sara_audio.errors import RegistryValidationError

EXPECTED_COUNTS = {
    "STR": 16,
    "LEX": 20,
    "MOR": 14,
    "ORT": 8,
    "PUN": 7,
    "SYN": 16,
    "EMO": 11,
}


@dataclass(frozen=True)
class LinguisticFeatureDefinition:
    """A source-controlled definition transferred from the input DOCX."""

    group_title: str
    code: str
    description: str

    @property
    def prefix(self) -> str:
        return self.code.split("_", maxsplit=1)[0]


class LinguisticFeatureRegistry:
    """Loads and validates the generated 92-feature JSON registry."""

    def __init__(self, definitions: tuple[LinguisticFeatureDefinition, ...]) -> None:
        self._definitions = definitions
        self._by_code = {definition.code: definition for definition in definitions}
        self.validate()

    @classmethod
    def load_default(cls) -> LinguisticFeatureRegistry:
        registry_path = files("sara_audio.data").joinpath("linguistic_features.json")
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
        definitions = tuple(
            LinguisticFeatureDefinition(
                group_title=item["group_title"],
                code=item["code"],
                description=item["description"],
            )
            for item in payload["features"]
        )
        return cls(definitions)

    @property
    def definitions(self) -> tuple[LinguisticFeatureDefinition, ...]:
        return self._definitions

    def get(self, code: str) -> LinguisticFeatureDefinition:
        return self._by_code[code]

    def validate(self) -> None:
        codes = [definition.code for definition in self._definitions]
        duplicates = [code for code, count in Counter(codes).items() if count > 1]
        if duplicates:
            raise RegistryValidationError(f"Duplicate feature codes: {duplicates}")

        expected_total = sum(EXPECTED_COUNTS.values())
        if len(self._definitions) != expected_total:
            raise RegistryValidationError(
                f"Expected {expected_total} definitions, got {len(self._definitions)}"
            )

        counts = Counter(definition.prefix for definition in self._definitions)
        if dict(counts) != EXPECTED_COUNTS:
            raise RegistryValidationError(
                f"Expected prefix counts {EXPECTED_COUNTS}, got {dict(counts)}"
            )

        for definition in self._definitions:
            if not definition.group_title or not definition.description:
                raise RegistryValidationError(
                    f"Feature {definition.code} has missing source metadata"
                )
