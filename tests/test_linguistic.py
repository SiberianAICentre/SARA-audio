from __future__ import annotations

import unittest

from sara_audio.domain import Transcript
from sara_audio.linguistic import LinguisticFeatureExtractor
from sara_audio.registry import LinguisticFeatureRegistry


class LinguisticFeatureExtractorTests(unittest.TestCase):
    def test_returns_a_record_for_each_docx_feature(self) -> None:
        extractor = LinguisticFeatureExtractor(LinguisticFeatureRegistry.load_default())
        values = extractor.extract(Transcript("Я очень рад, потому что это прекрасно!"))

        by_code = {value.code: value for value in values}
        self.assertEqual(len(values), 92)
        self.assertEqual(len(by_code), 92)
        self.assertEqual(by_code["LEX_EmotLex"].value, 2.0)
        self.assertEqual(by_code["MOR_IntensPart"].value, 1.0)
        self.assertEqual(by_code["SYN_ErrSogl"].value, 0.0)
        self.assertTrue(all(value.value is not None for value in values))
        self.assertEqual({value.source for value in values}, {"heuristic-text"})
