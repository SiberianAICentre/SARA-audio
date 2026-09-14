"""Portable heuristic extraction for the Russian linguistic registry."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from sara_audio.domain import FeatureValue, Transcript
from sara_audio.registry import LinguisticFeatureRegistry

WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+(?:-[A-Za-zА-Яа-яЁё]+)*")
SENTENCE = re.compile(r"[^.!?]+")
COORD = frozenset({"и", "а", "но", "или", "либо", "да", "зато"})
SUBORD = frozenset({"что", "чтобы", "если", "когда", "потому", "хотя", "который", "будто"})
EMOTIONAL = frozenset({"люблю", "ненавижу", "рад", "рада", "счастлив", "ужас", "прекрасно", "страшно", "восхитительно", "обидно"})
LOWERED = frozenset({"блин", "типа", "офигенно", "жесть", "фигня", "короче"})
LOGIC = frozenset({"поэтому", "следовательно", "итак", "однако", "значит", "во-первых", "во-вторых"})
INTENSIFIERS = frozenset({"очень", "крайне", "весьма", "совсем", "так", "совершенно", "невероятно"})
LIMITERS = frozenset({"только", "лишь", "всего", "едва", "почти"})


@dataclass(frozen=True)
class _Context:
    text: str
    words: tuple[str, ...]
    sentences: tuple[tuple[str, ...], ...]

    def count(self, vocabulary: frozenset[str]) -> float:
        return float(sum(word in vocabulary for word in self.words))


class LinguisticFeatureExtractor:
    """Compute every registry code with local, auditable Russian heuristics.

    The classifier-free implementation is portable to the server. Semantic,
    spelling and syntactic decisions remain explicitly low-confidence heuristics
    until calibrated against an expert-labelled corpus.
    """

    method_version = "russian-heuristics-v2"

    def __init__(self, registry: LinguisticFeatureRegistry) -> None:
        self._registry = registry
        self._rules = self._build_rules()

    def extract(self, transcript: Transcript) -> tuple[FeatureValue, ...]:
        context = self._context(transcript.text)
        return tuple(
            self._value(definition.code, self._rules[definition.code](context), transcript)
            for definition in self._registry.definitions
        )

    def _build_rules(self) -> dict[str, Callable[[_Context], float]]:
        vocabularies: dict[str, frozenset[str]] = {
            "STR_Digress": frozenset({"кстати", "вообще", "между", "отвлекусь"}),
            "STR_Argum": frozenset({"потому", "поэтому", "факт", "доказательство"}),
            "STR_EmClaim": EMOTIONAL,
            "STR_Chaotic": LOWERED | frozenset({"ну", "ээ"}),
            "LEX_EmotLex": EMOTIONAL,
            "LEX_Tropes": frozenset({"словно", "будто", "подобно", "невероятно"}),
            "LEX_CollocViol": frozenset({"ложить"}),
            "LEX_Underlex": frozenset({"это", "такое", "там"}),
            "LEX_Antonyms": frozenset({"хороший", "плохой", "большой", "маленький", "всегда", "никогда"}),
            "LEX_Comparisons": frozenset({"как", "словно", "будто", "подобно"}),
            "LEX_TempLex": frozenset({"сегодня", "вчера", "завтра", "сначала", "потом", "теперь"}),
            "LEX_IntelLex": frozenset({"думать", "понимать", "знать", "решить", "анализ", "логика", "идея"}),
            "LEX_Attitude": frozenset({"кажется", "думаю", "считаю", "уверен", "мнение", "вероятно"}),
            "LEX_WorkSyn": frozenset({"труд", "работа"}),
            "LEX_Antithesis": frozenset({"но", "а"}),
            "LEX_Oxymoron": frozenset({"лед", "тишина"}),
            "LEX_SemViol": frozenset({"ложить"}),
            "LEX_WordOmiss": frozenset({"..."}),
            "LEX_PleonTaut": frozenset({"суть", "сувенир", "планировать"}),
            "LEX_AntonymErr": frozenset({"уникальный", "оптимальный"}),
            "MOR_RelPronConj": frozenset({"который", "где", "когда"}),
            "MOR_StateWords": frozenset({"жарко", "холодно", "грустно", "весело", "нельзя", "можно", "стыдно"}),
            "MOR_AdvMPC": frozenset({"здесь", "там", "туда", "сюда", "почему", "зачем", "поэтому"}),
            "MOR_ModalFreq": frozenset({"можно", "нужно", "надо", "должен", "должна", "вероятно", "возможно"}),
            "EMO_Aesthetic": frozenset({"красиво", "прекрасно", "некрасиво", "гармонично"}),
            "EMO_Pragmatic": frozenset({"удобно", "полезно", "выгодно", "практично", "эффективно"}),
            "EMO_Rational": frozenset({"разумно", "логично", "правильно", "ответственно"}),
            "EMO_Ethical": frozenset({"честно", "стыдно", "совесть", "правильно", "вина", "долг"}),
            "EMO_Sentimental": frozenset({"скучаю", "помню", "родной", "милый", "сердце", "ностальгия"}),
            "EMO_ShameAppeal": frozenset({"стыдись", "стыдно", "совесть", "постыдно", "позор"}),
        }
        rules: dict[str, Callable[[_Context], float]] = {
            code: lambda context, words=words: context.count(words)
            for code, words in vocabularies.items()
        }
        rules.update(
            {
                "STR_Coher": lambda c: self._binary(bool(c.sentences) and self._logic_density(c) > 0),
                "STR_LogLink": lambda c: c.count(LOGIC),
                "STR_Concise": lambda c: self._binary(bool(c.words) and self._mean_sentence_size(c) <= 15),
                "STR_ConjLinks": lambda c: c.count(COORD | SUBORD),
                "STR_ClearLogic": lambda c: self._binary(self._logic_density(c) >= 0.1),
                "STR_NoLogic": lambda c: self._binary(len(c.sentences) > 1 and self._logic_density(c) == 0),
                "STR_Deductive": lambda c: self._binary(self._contains(c, {"в целом", "вообще"}) and self._contains(c, {"например"})),
                "STR_Inductive": lambda c: self._binary(self._contains(c, {"например"}) and self._contains(c, {"итак", "значит", "поэтому"})),
                "STR_LogErrors": self._contradictions,
                "STR_NoLogErrors": lambda c: self._binary(bool(c.sentences) and self._contradictions(c) == 0),
                "STR_LogErrorsSSC": lambda c: self._contradictions(c) * self._complex_sentences(c),
                "STR_SSC_Coherent": lambda c: self._binary(len(c.sentences) > 1 and self._logic_density(c) > 0),
                "LEX_Overlex": self._repetitions,
                "LEX_SynRows34": lambda c: self._enumerations(c, 3, 4),
                "LEX_MultiComp": lambda c: self._multi_comparisons(c),
                "LEX_SynChain": lambda c: self._enumerations(c, 4, 99),
                "MOR_PassiveFreq": lambda c: self._endings(c, ("ться", "нный", "тый")),
                "MOR_ActiveFreq": lambda c: self._endings(c, ("ет", "ют", "ит", "ают", "ала", "или")),
                "MOR_IntensPart": lambda c: c.count(INTENSIFIERS),
                "MOR_CoordDomin": lambda c: self._binary(c.count(COORD) > c.count(SUBORD)),
                "MOR_CoordConj": lambda c: c.count(COORD),
                "MOR_SubordDomin": lambda c: self._binary(c.count(SUBORD) > c.count(COORD)),
                "MOR_SubordConj": lambda c: c.count(SUBORD),
                "MOR_LimitPart": lambda c: c.count(LIMITERS),
                "MOR_ImperFreq": lambda c: self._endings(c, ("йте", "й", "и")),
                "MOR_GenderErr": self._gender_conflicts,
                "ORT_AltRootVow": lambda c: self._spelling(c, {"ростание", "лагается", "гареть"}),
                "ORT_AdvPrefixErr": lambda c: self._spelling(c, {"по русски"}),
                "ORT_CompoundNum": lambda c: self._spelling(c, {"двухста"}),
                "ORT_DoubleCons": lambda c: self._spelling(c, {"колектив", "группаа"}),
                "ORT_PrePriErr": lambda c: self._spelling(c, {"приувеличить"}),
                "ORT_TakZheErr": lambda c: self._spelling(c, {"так же", "то же"}),
                "ORT_NotErr": lambda c: self._spelling(c, {"не сделан"}),
                "ORT_CompNounErr": lambda c: self._spelling(c, {"юго запад"}),
                "PUN_Dash": lambda c: float(c.text.count("-") + c.text.count("—")),
                "PUN_Parenth": lambda c: float(c.text.count("(") + c.text.count(")")),
                "PUN_EmoMarks": lambda c: float(c.text.count("!") + c.text.count("?")),
                "PUN_Quotes": lambda c: float(c.text.count('"') + c.text.count("«") + c.text.count("»")),
                "PUN_CommaDetach": lambda c: float(len(re.findall(r",\s*(?:как|который|если|когда)\b", c.text.lower()))),
                "PUN_ComplexMarks": lambda c: float(c.text.count(";") + c.text.count(":")),
                "PUN_HomogErr": lambda c: float(len(re.findall(r"\bи\s+и\b|,\s*и\s*,", c.text.lower()))),
                "SYN_NominalPh": self._nominal_pairs,
                "SYN_AdjectPh": self._adjective_pairs,
                "SYN_VerbalPh": self._verbal_pairs,
                "SYN_Sogl": self._adjective_pairs,
                "SYN_SoglDomin": lambda c: self._binary(self._adjective_pairs(c) > self._verbal_pairs(c)),
                "SYN_Upravl": self._verbal_pairs,
                "SYN_UpravlDomin": lambda c: self._binary(self._verbal_pairs(c) > self._adjective_pairs(c)),
                "SYN_HighSpread": lambda c: float(sum(len(sentence) >= 12 for sentence in c.sentences)),
                "SYN_AsyndetFreq": self._asyndetic,
                "SYN_ElipSent": lambda c: float(sum(len(sentence) <= 3 for sentence in c.sentences)),
                "SYN_ChtobyPref": lambda c: c.count(frozenset({"чтобы"})),
                "SYN_ComplexSent": self._complex_sentences,
                "SYN_ConjFreq": lambda c: c.count(COORD | SUBORD),
                "SYN_MixConst": lambda c: c.count(frozenset({"если", "чтобы", "но"})) * self._binary(self._complex_sentences(c) > 0),
                "SYN_ErrUpravl": lambda c: self._spelling(c, {"согласно приказа", "благодаря дождя"}),
                "SYN_ErrSogl": lambda c: self._spelling(c, {"много людей пришли", "большинство были"}),
                "EMO_General": lambda c: c.count(EMOTIONAL | LOWERED | INTENSIFIERS),
                "EMO_Diminutive": lambda c: self._endings(c, ("очка", "ечка", "оньк", "еньк")),
                "EMO_ExpressLex": lambda c: c.count(EMOTIONAL | LOWERED),
                "EMO_LoweredLex": lambda c: c.count(LOWERED),
                "EMO_Hyperbole": lambda c: c.count(frozenset({"всегда", "никогда", "абсолютно", "невероятно"})),
            }
        )
        missing = {item.code for item in self._registry.definitions} - set(rules)
        if missing:
            raise RuntimeError(f"No linguistic heuristic for {sorted(missing)}")
        return rules

    def _value(self, code: str, value: float, transcript: Transcript) -> FeatureValue:
        sensitive = code.startswith(("STR_", "ORT_", "SYN_")) or code in {"LEX_CollocViol", "LEX_SemViol", "LEX_WordOmiss"}
        return FeatureValue(
            code=code,
            value=value,
            unit="count",
            source="heuristic-text",
            confidence=(0.35 if sensitive else 0.6) if not transcript.is_verified else (0.45 if sensitive else 0.75),
            method_version=self.method_version,
            evidence={"transcript_source": transcript.source, "implementation": "portable-heuristic"},
        )

    @staticmethod
    def _context(text: str) -> _Context:
        sentences = tuple(
            tuple(word.lower() for word in WORD.findall(part))
            for part in SENTENCE.findall(text)
            if WORD.search(part)
        )
        return _Context(text, tuple(word.lower() for word in WORD.findall(text)), sentences)

    @staticmethod
    def _binary(value: bool) -> float:
        return 1.0 if value else 0.0

    @staticmethod
    def _contains(context: _Context, values: set[str]) -> bool:
        return bool(set(context.words) & values) or any(value in context.text.lower() for value in values)

    @staticmethod
    def _mean_sentence_size(context: _Context) -> float:
        return sum(map(len, context.sentences)) / len(context.sentences) if context.sentences else 0.0

    @staticmethod
    def _logic_density(context: _Context) -> float:
        return context.count(LOGIC) / max(len(context.words), 1)

    @staticmethod
    def _repetitions(context: _Context) -> float:
        return float(sum(context.words.count(word) - 1 for word in set(context.words) if len(word) > 3))

    @staticmethod
    def _contradictions(context: _Context) -> float:
        return float(sum({"всегда", "никогда"} <= set(sentence) for sentence in context.sentences))

    @staticmethod
    def _enumerations(context: _Context, lower: int, upper: int) -> float:
        return float(sum(lower <= sentence.count("и") + sentence.count("или") + 1 <= upper for sentence in context.sentences))

    @staticmethod
    def _multi_comparisons(context: _Context) -> float:
        markers = {"как", "словно", "будто", "подобно"}
        return float(sum(sum(word in markers for word in sentence) >= 2 for sentence in context.sentences))

    @staticmethod
    def _endings(context: _Context, endings: tuple[str, ...]) -> float:
        return float(sum(word.endswith(endings) or any(end in word for end in endings if len(end) > 3) for word in context.words))

    @staticmethod
    def _gender_conflicts(context: _Context) -> float:
        return float(bool(set(context.words) & {"рада", "сделала", "должна"}) and bool(set(context.words) & {"рад", "сделал", "должен"}))

    @staticmethod
    def _spelling(context: _Context, patterns: set[str]) -> float:
        lowered = context.text.lower()
        return float(sum(pattern in lowered for pattern in patterns))

    @staticmethod
    def _adjective_pairs(context: _Context) -> float:
        return float(sum(first.endswith(("ый", "ий", "ая", "ое", "ые", "ого", "ому")) for first, _ in zip(context.words, context.words[1:], strict=False)))

    @staticmethod
    def _nominal_pairs(context: _Context) -> float:
        return float(sum(len(first) > 3 and len(second) > 3 for first, second in zip(context.words, context.words[1:], strict=False)))

    @staticmethod
    def _verbal_pairs(context: _Context) -> float:
        return float(sum(first.endswith(("ет", "ют", "ит", "ают", "ала", "или", "ать", "ять")) for first, _ in zip(context.words, context.words[1:], strict=False)))

    @staticmethod
    def _asyndetic(context: _Context) -> float:
        return float(sum(len(sentence) >= 6 and not set(sentence) & (COORD | SUBORD) for sentence in context.sentences))

    @staticmethod
    def _complex_sentences(context: _Context) -> float:
        return float(sum(bool(set(sentence) & SUBORD) for sentence in context.sentences))
