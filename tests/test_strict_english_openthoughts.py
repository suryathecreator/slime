from __future__ import annotations

from dataclasses import dataclass

from examples.qwen3_8b_opd_tillicum import _strict_english_filters as filters
from examples.qwen3_8b_opd_tillicum.prepare_strict_english_openthoughts3 import evaluate_row


@dataclass
class Language:
    name: str


@dataclass
class Span:
    language: Language
    start_index: int
    end_index: int
    word_count: int


class FakeDetector:
    def detect_language_of(self, text: str):
        if "bonjour" in text.lower() or "por lo tanto" in text.lower():
            return Language("SPANISH")
        return Language("ENGLISH")

    def detect_multiple_languages_of(self, text: str):
        lowered = text.lower()
        for needle, language in (("bonjour tout le monde", "FRENCH"), ("por lo tanto", "SPANISH")):
            if needle in lowered:
                start = lowered.index(needle)
                return [Span(Language(language), start, start + len(needle), len(needle.split()))]
        return [Span(Language("ENGLISH"), 0, len(text), len(text.split()))]


def row(assistant: str, prompt: str = "Solve the equation $x^2=4$."):
    return {
        "domain": "math",
        "source": "fixture",
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": assistant},
        ],
    }


def test_latex_and_math_symbols_are_masked_without_rejecting_english():
    text = r"Use $\alpha + \beta = 2$ and \[x^2=4\] to prove the result."
    assert filters.english_rejection_reasons(text, FakeDetector()) == []


def test_latin_script_foreign_and_multilingual_spans_are_rejected():
    assert filters.english_rejection_reasons("Por lo tanto this is the answer", FakeDetector())
    assert filters.english_rejection_reasons("The proof ends. Bonjour tout le monde.", FakeDetector())


def test_non_latin_prose_is_rejected_but_isolated_greek_math_is_allowed():
    assert filters.english_rejection_reasons("Let alpha denote α and solve the equation.", FakeDetector()) == []
    assert "non_latin_prose" in filters.english_rejection_reasons("The result is 因此这是答案", FakeDetector())


def test_exact_think_final_answer_and_math_domain_are_required():
    good = evaluate_row(
        row("<think>We solve x^2=4, so x is 2 or -2.</think>The final answer is $\\boxed{2,-2}$"),
        7,
        FakeDetector(),
    )
    assert good["accepted"]
    assert good["strict_math"]

    incomplete = evaluate_row(row("<think>Still solving x^2=4."), 8, FakeDetector())
    assert incomplete["incomplete_think"]
    assert not incomplete["accepted"]

    duplicate = evaluate_row(
        row("<think>a=b.\na=b.\na=b.\na=b.</think>The final answer is $\\boxed{b}$."),
        9,
        FakeDetector(),
    )
    assert duplicate["accepted"]
    assert not duplicate["strict_math"]
    assert "repeated_lines_15pct" in duplicate["strict_math_reasons"]


def test_repeated_ngram_self_loop_is_rejected():
    loop = "one two three four five six seven eight " * 5
    assert "repeated_8gram" in filters.repetition_rejection_reasons(loop)
