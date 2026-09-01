"""Handwritten text disambiguation engine for Phase 2.

This module provides tools to resolve common handwritten character ambiguities:
- 0 vs O/o (zero vs letter)
- 1 vs l vs I vs | (one vs lowercase L vs uppercase I vs pipe)
- 5 vs S, 2 vs Z, 9 vs g vs q
- Indentation patterns (spaces vs tabs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# Standard confusion pairs in handwriting
STANDARD_CONFUSION_PAIRS = [
    ("0", "O"),  # zero vs letter O
    ("0", "o"),  # zero vs lowercase o
    ("1", "l"),  # one vs lowercase L
    ("1", "I"),  # one vs uppercase I
    ("1", "|"),  # one vs pipe
    ("5", "S"),  # five vs uppercase S
    ("2", "Z"),  # two vs uppercase Z
    ("9", "g"),  # nine vs lowercase g
    ("9", "q"),  # nine vs lowercase q
    ("8", "B"),  # eight vs uppercase B
]

# Python-specific context hints
PYTHON_KEYWORDS = {
    "if", "else", "elif", "for", "while", "def", "class", "return", "import",
    "from", "as", "try", "except", "finally", "with", "assert", "pass", "break",
    "continue", "lambda", "yield", "global", "nonlocal", "None", "True", "False",
    "and", "or", "not", "in", "is", "print", "len", "str", "int", "float", "list",
    "dict", "set", "tuple", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "sum", "min", "max", "all", "any", "open", "file", "input", "output",
}

# Common library patterns
COMMON_PATTERNS = {
    "numpy": re.compile(r"\bnp\.|numpy"),
    "pandas": re.compile(r"\bpd\.|pandas"),
    "matplotlib": re.compile(r"\bplt\.|matplotlib"),
    "requests": re.compile(r"\brequests\.|from requests"),
    "tensorflow": re.compile(r"\btf\.|tensorflow"),
    "torch": re.compile(r"\btorch\.|pytorch"),
}


@dataclass
class Correction:
    """Represents a single character correction."""

    position: int
    original: str
    corrected: str
    confidence: float
    reasoning: str


class HandwritingDisambiguator:
    """Engine for resolving handwritten text ambiguities."""

    def __init__(self, custom_confusion_pairs: Sequence[tuple[str, str]] | None = None):
        """Initialize the disambiguator.

        Args:
            custom_confusion_pairs: Optional custom confusion pairs to consider.
        """
        self.confusion_pairs = list(STANDARD_CONFUSION_PAIRS)
        if custom_confusion_pairs:
            self.confusion_pairs.extend(custom_confusion_pairs)

    def disambiguate(
        self,
        text: str,
        context_language: str = "python",
        context_clues: dict[str, str] | None = None,
    ) -> dict[str, any]:  # type: ignore
        """Disambiguate text based on context and confusion pairs.

        Args:
            text: Text to disambiguate.
            context_language: Expected programming language ('python', 'javascript', 'c++', 'sql', etc.).
            context_clues: Optional dict with keys like 'surrounding_text', 'line_indent', etc.

        Returns:
            Dictionary with:
            - disambiguated_text: Corrected text
            - corrections: List of Correction objects
            - confidence: Overall confidence (0-1)
            - unresolved: Characters that remain ambiguous
        """
        corrections: list[Correction] = []
        result_chars = list(text)
        confidence_scores: list[float] = []

        for i, char in enumerate(text):
            # Find confusion pairs involving this character
            candidates = self._find_candidates(char)

            if not candidates:
                continue

            # Evaluate each candidate
            scores = {}
            for candidate in candidates:
                score = self._score_candidate(
                    text=text,
                    position=i,
                    char=char,
                    candidate=candidate,
                    language=context_language,
                    context_clues=context_clues or {},
                )
                scores[candidate] = score

            # Pick best candidate if confidence is high enough
            best_candidate = max(scores, key=scores.get)
            best_score = scores[best_candidate]

            if best_candidate != char and best_score > 0.6:
                correction = Correction(
                    position=i,
                    original=char,
                    corrected=best_candidate,
                    confidence=best_score,
                    reasoning=self._explain_correction(char, best_candidate, best_score),
                )
                corrections.append(correction)
                result_chars[i] = best_candidate
                confidence_scores.append(best_score)

        disambiguated = "".join(result_chars)
        overall_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.9
        )

        return {
            "disambiguated_text": disambiguated,
            "corrections": corrections,
            "confidence": overall_confidence,
            "unresolved": self._find_unresolved(text, corrections),
        }

    def _find_candidates(self, char: str) -> list[str]:
        """Find all candidates for a given character."""
        candidates = set()
        for wrong, correct in self.confusion_pairs:
            if char == wrong:
                candidates.add(correct)
            elif char == correct:
                candidates.add(wrong)
        return list(candidates)

    def _score_candidate(
        self,
        text: str,
        position: int,
        char: str,
        candidate: str,
        language: str,
        context_clues: dict[str, str],
    ) -> float:
        """Score how likely a candidate is correct (0-1)."""
        score = 0.0

        # Check language-specific keywords
        if language == "python":
            score += self._score_python_context(text, position, candidate)
        elif language in {"javascript", "typescript"}:
            score += self._score_javascript_context(text, position, candidate)

        # Check common library patterns
        score += self._score_library_patterns(text, candidate)

        # Digit/letter distinction
        score += self._score_digit_letter_distinction(text, position, candidate)

        # Indentation hints
        if "line_indent" in context_clues:
            score += self._score_indentation(position, candidate, context_clues["line_indent"])

        # Normalize to 0-1 range
        return min(1.0, max(0.0, score))

    def _score_python_context(self, text: str, position: int, candidate: str) -> float:
        """Score candidate in Python context."""
        # Extract word around position
        word_match = self._extract_word_at_position(text, position)
        if not word_match:
            return 0.0

        word = word_match

        # Check if forms a valid Python keyword
        if word in PYTHON_KEYWORDS:
            # This helps disambiguate keywords like 'if', 'for', etc.
            if candidate in {"I", "l", "1", "O"}:  # Common confusions
                return 0.7  # Slightly prefer letters in keywords
            return 0.3

        # Check for common patterns
        if "(" in word or ")" in word:
            # Function calls typically use 0 (zero) not O (letter)
            if candidate == "0":
                return 0.6
            elif candidate == "O":
                return 0.2

        return 0.1

    def _score_javascript_context(self, text: str, position: int, candidate: str) -> float:
        """Score candidate in JavaScript context."""
        word = self._extract_word_at_position(text, position)
        if not word:
            return 0.0

        js_keywords = {
            "if", "else", "for", "while", "function", "const", "let", "var",
            "return", "import", "export", "class", "extends", "super", "this",
            "true", "false", "null", "undefined", "async", "await", "try",
            "catch", "finally", "new", "delete", "typeof", "instanceof",
        }

        if word in js_keywords:
            if candidate in {"I", "l", "1", "O"}:
                return 0.65
            return 0.25

        return 0.1

    def _score_library_patterns(self, text: str, candidate: str) -> float:
        """Score based on library import patterns."""
        score = 0.0
        for lib, pattern in COMMON_PATTERNS.items():
            if pattern.search(text):
                # Slight bias based on common patterns in that library
                if lib == "numpy" and candidate in {"0", "l"}:
                    score += 0.15
        return score

    def _score_digit_letter_distinction(
        self,
        text: str,
        position: int,
        candidate: str,
    ) -> float:
        """Score based on digit vs letter distinction."""
        # Look at surrounding characters
        prev_char = text[position - 1] if position > 0 else " "
        next_char = text[position + 1] if position < len(text) - 1 else " "

        # In numeric contexts (e.g., "2023", "0x1F"), prefer digits
        if (prev_char.isdigit() or next_char.isdigit()) and candidate.isdigit():
            return 0.4

        # In identifier contexts, prefer letters
        if (prev_char.isalnum() or next_char.isalnum()) and candidate.isalpha():
            return 0.35

        return 0.0

    def _score_indentation(
        self,
        position: int,
        candidate: str,
        line_indent: str,
    ) -> float:
        """Score based on indentation context."""
        if candidate == " " and line_indent:
            return 0.3
        return 0.0

    def _extract_word_at_position(self, text: str, position: int) -> str:
        """Extract word containing the character at position."""
        start = position
        end = position + 1

        while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
            start -= 1
        while end < len(text) and (text[end].isalnum() or text[end] == "_"):
            end += 1

        return text[start:end]

    def _explain_correction(self, original: str, corrected: str, confidence: float) -> str:
        """Generate explanation for a correction."""
        if confidence > 0.8:
            return f"High confidence: '{original}' → '{corrected}'"
        elif confidence > 0.6:
            return f"Moderate confidence: '{original}' → '{corrected}'"
        else:
            return f"Low confidence: '{original}' → '{corrected}'"

    def _find_unresolved(
        self,
        text: str,
        corrections: list[Correction],
    ) -> list[str]:
        """Find characters that remain ambiguous after corrections."""
        unresolved = []
        for i, char in enumerate(text):
            if any(c.position == i for c in corrections):
                continue  # Already corrected

            candidates = self._find_candidates(char)
            if candidates:
                unresolved.append(f"Position {i}: '{char}' (ambiguous)")

        return unresolved


class IndentationAnalyzer:
    """Analyze indentation patterns in handwritten code."""

    def detect_indentation_style(self, text: str) -> dict[str, any]:  # type: ignore
        """Detect indentation style (spaces vs tabs vs visual offsets).

        Args:
            text: Multi-line text to analyze.

        Returns:
            Dictionary with:
            - style: 'spaces', 'tabs', or 'mixed'
            - indent_size: Number of spaces/tabs per level
            - confidence: Confidence in detection
            - line_indents: List of indentation strings for each line
        """
        lines = text.split("\n")
        indents: list[str] = []

        for line in lines:
            if line.strip():  # Non-empty line
                indent = line[: len(line) - len(line.lstrip())]
                indents.append(indent)

        # Analyze indents
        space_count = sum(1 for i in indents if " " in i and "\t" not in i)
        tab_count = sum(1 for i in indents if "\t" in i)
        mixed = tab_count > 0 and space_count > 0

        if mixed:
            style = "mixed"
        elif tab_count > space_count:
            style = "tabs"
        else:
            style = "spaces"

        # Detect indent size
        indent_sizes = []
        for indent in indents:
            if indent.startswith("\t"):
                indent_sizes.append(1)
            elif indent.startswith(" "):
                # Count leading spaces
                size = len(indent) - len(indent.lstrip(" "))
                if size > 0:
                    indent_sizes.append(size)

        indent_size = max(indent_sizes) if indent_sizes else 4

        confidence = min(1.0, max(space_count, tab_count) / (len(indents) + 1))

        return {
            "style": style,
            "indent_size": indent_size,
            "confidence": confidence,
            "line_indents": indents,
        }
