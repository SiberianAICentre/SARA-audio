from __future__ import annotations

import unittest

from sara_audio.registry import EXPECTED_COUNTS, LinguisticFeatureRegistry


class RegistryTests(unittest.TestCase):
    def test_default_registry_preserves_all_92_source_features(self) -> None:
        registry = LinguisticFeatureRegistry.load_default()

        self.assertEqual(len(registry.definitions), 92)
        self.assertEqual(
            {prefix: sum(item.prefix == prefix for item in registry.definitions) for prefix in EXPECTED_COUNTS},
            EXPECTED_COUNTS,
        )
        self.assertEqual(len({item.code for item in registry.definitions}), 92)
