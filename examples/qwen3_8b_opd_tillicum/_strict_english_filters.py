"""Pure filtering helpers for the strict-English OpenThoughts experiment."""

from __future__ import annotations

import importlib.metadata
import re
import unicodedata
from typing import Any


MATH_MASKS = (
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    re.compile(r"\\\[.*?\\\]", re.S),
    re.compile(r"\\\(.*?\\\)", re.S),
    re.compile(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.S),
    re.compile(r"https?://\S+|www\.\S+", re.I),
    re.compile(r"\\[A-Za-z]+(?:\{[^{}]*\})?"),
    re.compile(r"(?<!\w)[\d.,:+*/=<>^_%|()[\]{}-]+(?!\w)"),
)
MATH_TERMS = re.compile(
    r"\b(theorem|proof|equation|function|integer|polynomial|matrix|integral|derivative|probability|"
    r"geometry|algebra|calculus|sequence|sum|product|factor|prime|divisible|angle|triangle|radius|"
    r"solution|answer|evaluate|compute|solve|show that|boxed)\b",
    re.I,
)
ANSWER_MARKERS = re.compile(r"\\boxed|final answer|answer is|therefore|hence", re.I)


def build_detector() -> Any:
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as exc:
        raise RuntimeError(
            "lingua-language-detector is missing; run 11_setup_strict_language_env.sbatch first"
        ) from exc
    detector = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()
    print(f"lingua-language-detector {importlib.metadata.version('lingua-language-detector')}", flush=True)
    return detector


def mask_math_and_noise(text: str) -> str:
    masked = text
    for pattern in MATH_MASKS:
        masked = pattern.sub(" ", masked)
    return re.sub(r"\s+", " ", masked).strip()


def _language_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).upper()


def _meaningful_letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _non_latin_prose_letters(text: str) -> int:
    count = 0
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "LATIN" not in name and "GREEK" not in name:
            count += 1
    return count


def english_rejection_reasons(text: str, detector: Any) -> list[str]:
    masked = mask_math_and_noise(text)
    letters = _meaningful_letters(masked)
    if letters == 0:
        return []
    reasons: list[str] = []
    non_latin = _non_latin_prose_letters(masked)
    if non_latin >= 4:
        reasons.append("non_latin_prose")
    language = detector.detect_language_of(masked)
    if letters >= 20 and language is not None and _language_name(language) != "ENGLISH":
        reasons.append(f"whole_{_language_name(language).lower()}")
    try:
        spans = detector.detect_multiple_languages_of(masked)
    except (AttributeError, ValueError):
        spans = []
    for span in spans:
        lang = _language_name(getattr(span, "language", "UNKNOWN"))
        if lang == "ENGLISH":
            continue
        start = int(getattr(span, "start_index", 0))
        end = int(getattr(span, "end_index", start))
        span_text = masked[start:end]
        word_count = int(getattr(span, "word_count", len(re.findall(r"\b\w+\b", span_text))))
        if _meaningful_letters(span_text) >= 12 or word_count >= 3:
            reasons.append(f"span_{lang.lower()}")
    return sorted(set(reasons))


def noise_rejection_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    if "\ufffd" in text or re.search(r"(?:Ã.|Â.|â€){2,}", text):
        reasons.append("mojibake")
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text):
        reasons.append("control_character")
    if len(re.findall(r"https?://|www\.", text, re.I)) >= 4:
        reasons.append("excessive_urls")
    if len(re.findall(r"</?(?:html|body|div|script|style|iframe|nav|footer)\b", text, re.I)) >= 3:
        reasons.append("web_boilerplate")
    if re.search(r"(.)\1{24,}", text, re.S):
        reasons.append("repeated_character")
    if re.search(r"\b\w{100,}\b", text):
        reasons.append("nonsensical_long_token")
    return reasons


def is_math_heavy(prompt: str, thinking: str, answer: str) -> bool:
    text = "\n".join((prompt, thinking, answer))
    signals = 0
    signals += min(4, len(re.findall(r"\\[A-Za-z]+|\$[^$]+\$|\\\[|\\\(", text)))
    signals += min(3, len(re.findall(r"\d+\s*[-+*/=<>^]\s*\d+|[A-Za-z]\s*=", text)))
    signals += min(3, len(MATH_TERMS.findall(text)))
    signals += 2 if ANSWER_MARKERS.search(answer) else 0
    return signals >= 3


def repetition_rejection_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    lines = [re.sub(r"\s+", " ", line.strip().lower()) for line in text.splitlines() if line.strip()]
    if lines:
        repeated_line_chars = sum(len(line) for line in lines if lines.count(line) > 1)
        if repeated_line_chars / max(1, sum(map(len, lines))) >= 0.15:
            reasons.append("repeated_lines_15pct")
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if any(a == b and len(a) >= 20 for a, b in zip(sentences, sentences[1:])):
        reasons.append("consecutive_duplicate_sentence")
    tokens = re.findall(r"\S+", text.lower())
    if len(tokens) >= 8:
        counts: dict[tuple[str, ...], int] = {}
        for i in range(len(tokens) - 7):
            gram = tuple(tokens[i : i + 8])
            counts[gram] = counts.get(gram, 0) + 1
        max_repeats = max(counts.values(), default=0)
        if max_repeats >= 4 and (max_repeats * 8) / len(tokens) >= 0.20:
            reasons.append("repeated_8gram")
        for cycle in range(4, min(64, len(tokens) // 3) + 1):
            if tokens[-cycle:] == tokens[-2 * cycle : -cycle] == tokens[-3 * cycle : -2 * cycle]:
                reasons.append("repeated_suffix_cycle")
                break
    return reasons
