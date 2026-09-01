"""Language and dependency detection module for Phase 2.

This module detects programming languages and frameworks in handwritten code,
extracting implicit dependencies like:
- numpy from 'np.array' → 'import numpy as np'
- pandas from 'pd.DataFrame' → 'import pandas as pd'
- matplotlib from 'plt.plot' → 'import matplotlib.pyplot as plt'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProgrammingLanguage(Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    JAVA = "java"
    CPP = "cpp"
    SQL = "sql"
    R = "r"
    JULIA = "julia"
    UNKNOWN = "unknown"


@dataclass
class Dependency:
    """Represents a detected dependency."""

    name: str
    import_statement: str
    confidence: float
    reason: str


class LanguageDetector:
    """Detect programming language from handwritten code."""

    def __init__(self):
        """Initialize language detector with patterns."""
        self.language_patterns = {
            ProgrammingLanguage.PYTHON: self._compile_python_patterns(),
            ProgrammingLanguage.JAVASCRIPT: self._compile_javascript_patterns(),
            ProgrammingLanguage.TYPESCRIPT: self._compile_typescript_patterns(),
            ProgrammingLanguage.CSHARP: self._compile_csharp_patterns(),
            ProgrammingLanguage.JAVA: self._compile_java_patterns(),
            ProgrammingLanguage.CPP: self._compile_cpp_patterns(),
            ProgrammingLanguage.SQL: self._compile_sql_patterns(),
            ProgrammingLanguage.R: self._compile_r_patterns(),
            ProgrammingLanguage.JULIA: self._compile_julia_patterns(),
        }

    @staticmethod
    def _compile_python_patterns() -> dict[str, re.Pattern[str]]:
        """Python language patterns."""
        return {
            "def": re.compile(r"\bdef\s+\w+\s*\("),
            "class": re.compile(r"\bclass\s+\w+\s*[\(:]"),
            "import": re.compile(r"\b(import|from)\s+\w+"),
            "indent": re.compile(r"^\s+"),
            "print": re.compile(r"\bprint\s*\("),
            "list_comp": re.compile(r"\[.*\bfor\b.*\bin\b.*\]"),
            "for_in": re.compile(r"\bfor\s+\w+\s+in\s+"),
            "lambda": re.compile(r"\blambda\s+"),
            "decorator": re.compile(r"@\w+"),
            "dict": re.compile(r"\{.*:\s*.*\}"),
        }

    @staticmethod
    def _compile_javascript_patterns() -> dict[str, re.Pattern[str]]:
        """JavaScript language patterns."""
        return {
            "function": re.compile(r"\bfunction\s+\w+\s*\("),
            "const": re.compile(r"\bconst\s+\w+\s*="),
            "let": re.compile(r"\blet\s+\w+\s*="),
            "var": re.compile(r"\bvar\s+\w+\s*="),
            "arrow": re.compile(r"=>"),
            "require": re.compile(r"\brequire\s*\("),
            "console": re.compile(r"\bconsole\s*\."),
            "async": re.compile(r"\basync\s+(function|\(|\w+)"),
            "template": re.compile(r"`[^`]*\$\{"),
        }

    @staticmethod
    def _compile_typescript_patterns() -> dict[str, re.Pattern[str]]:
        """TypeScript language patterns."""
        return {
            **LanguageDetector._compile_javascript_patterns(),
            "interface": re.compile(r"\binterface\s+\w+\s*\{"),
            "type": re.compile(r"\btype\s+\w+\s*="),
            "generic": re.compile(r"<\w+>"),
            "decorator": re.compile(r"@\w+"),
        }

    @staticmethod
    def _compile_csharp_patterns() -> dict[str, re.Pattern[str]]:
        """C# language patterns."""
        return {
            "class": re.compile(r"\bpublic\s+class\s+\w+"),
            "using": re.compile(r"\busing\s+\w+"),
            "async": re.compile(r"\basync\s+\w+"),
            "linq": re.compile(r"\.Where\s*\(|\.Select\s*\("),
            "var": re.compile(r"\bvar\s+\w+\s*="),
        }

    @staticmethod
    def _compile_java_patterns() -> dict[str, re.Pattern[str]]:
        """Java language patterns."""
        return {
            "class": re.compile(r"\bpublic\s+class\s+\w+"),
            "import": re.compile(r"\bimport\s+[\w\.]+"),
            "public": re.compile(r"\bpublic\s+(static\s+)?\w+"),
            "main": re.compile(r"public\s+static\s+void\s+main"),
            "new": re.compile(r"\bnew\s+\w+\s*\("),
        }

    @staticmethod
    def _compile_cpp_patterns() -> dict[str, re.Pattern[str]]:
        """C++ language patterns."""
        return {
            "include": re.compile(r"#include\s+[<\"]"),
            "using": re.compile(r"\busing\s+namespace"),
            "template": re.compile(r"\btemplate\s*<"),
            "ptr": re.compile(r"->|\*\w+"),
            "vector": re.compile(r"\bstd::\w+\s*<"),
        }

    @staticmethod
    def _compile_sql_patterns() -> dict[str, re.Pattern[str]]:
        """SQL language patterns."""
        return {
            "select": re.compile(r"\bSELECT\b", re.IGNORECASE),
            "from": re.compile(r"\bFROM\b", re.IGNORECASE),
            "where": re.compile(r"\bWHERE\b", re.IGNORECASE),
            "join": re.compile(r"\bJOIN\b", re.IGNORECASE),
            "group_by": re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
            "order_by": re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
        }

    @staticmethod
    def _compile_r_patterns() -> dict[str, re.Pattern[str]]:
        """R language patterns."""
        return {
            "assign": re.compile(r"<-|<<-"),
            "function": re.compile(r"\bfunction\s*\("),
            "package": re.compile(r"\blibrary\s*\(|require\s*\("),
            "data_frame": re.compile(r"\bdata\.frame\s*\("),
            "pipe": re.compile(r"%\w+%"),
        }

    @staticmethod
    def _compile_julia_patterns() -> dict[str, re.Pattern[str]]:
        """Julia language patterns."""
        return {
            "function": re.compile(r"\bfunction\s+\w+\s*\("),
            "module": re.compile(r"\bmodule\s+\w+"),
            "type": re.compile(r"\b(struct|mutable struct)\s+\w+"),
            "using": re.compile(r"\busing\s+\w+"),
            "broadcast": re.compile(r"\w+\."),
        }

    def detect(self, code: str) -> dict[str, Any]:
        """Detect the programming language of code.

        Args:
            code: Code snippet to analyze.

        Returns:
            Dictionary with:
            - language: Detected ProgrammingLanguage
            - confidence: Confidence score (0-1)
            - scores: Dict of language -> score for all languages
            - patterns_matched: Set of matched patterns
        """
        scores: dict[ProgrammingLanguage, float] = {}
        pattern_matches: dict[ProgrammingLanguage, set[str]] = {
            lang: set() for lang in self.language_patterns.keys()
        }

        # Score each language
        for language, patterns in self.language_patterns.items():
            score = 0.0
            matched_patterns = set()

            for pattern_name, pattern in patterns.items():
                if pattern.search(code):
                    score += 1.0
                    matched_patterns.add(pattern_name)

            # Normalize score by number of patterns
            if patterns:
                score = score / len(patterns)

            scores[language] = score
            pattern_matches[language] = matched_patterns

        # Find best match
        best_language = max(scores, key=scores.get)
        best_score = scores[best_language]

        if best_score == 0.0:
            return {
                "language": ProgrammingLanguage.UNKNOWN,
                "confidence": 0.0,
                "scores": scores,
                "patterns_matched": set(),
            }

        # Confidence based on how much better best match is
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = (best_score - sorted_scores[1]) / best_score
        else:
            confidence = best_score

        return {
            "language": best_language,
            "confidence": min(1.0, max(0.0, confidence)),
            "scores": scores,
            "patterns_matched": pattern_matches[best_language],
        }


class DependencyExtractor:
    """Extract implicit dependencies from code."""

    def __init__(self):
        """Initialize dependency patterns."""
        self.patterns = self._compile_dependency_patterns()

    @staticmethod
    def _compile_dependency_patterns() -> dict[str, tuple[re.Pattern[str], Dependency]]:
        """Compile patterns for detecting implicit dependencies."""
        return {
            "numpy": (
                re.compile(r"\bnp\.|numpy"),
                Dependency(
                    name="numpy",
                    import_statement="import numpy as np",
                    confidence=0.95,
                    reason="Found numpy/np usage",
                ),
            ),
            "pandas": (
                re.compile(r"\bpd\.|pandas"),
                Dependency(
                    name="pandas",
                    import_statement="import pandas as pd",
                    confidence=0.95,
                    reason="Found pandas/pd usage",
                ),
            ),
            "matplotlib": (
                re.compile(r"\bplt\.|matplotlib\.pyplot"),
                Dependency(
                    name="matplotlib",
                    import_statement="import matplotlib.pyplot as plt",
                    confidence=0.95,
                    reason="Found matplotlib.pyplot/plt usage",
                ),
            ),
            "seaborn": (
                re.compile(r"\bsns\.|seaborn"),
                Dependency(
                    name="seaborn",
                    import_statement="import seaborn as sns",
                    confidence=0.90,
                    reason="Found seaborn/sns usage",
                ),
            ),
            "scipy": (
                re.compile(r"\bscipy\."),
                Dependency(
                    name="scipy",
                    import_statement="from scipy import stats",
                    confidence=0.85,
                    reason="Found scipy usage",
                ),
            ),
            "sklearn": (
                re.compile(r"sklearn\."),
                Dependency(
                    name="scikit-learn",
                    import_statement="from sklearn import *",
                    confidence=0.90,
                    reason="Found sklearn usage",
                ),
            ),
            "tensorflow": (
                re.compile(r"\btf\.|tensorflow"),
                Dependency(
                    name="tensorflow",
                    import_statement="import tensorflow as tf",
                    confidence=0.95,
                    reason="Found tensorflow/tf usage",
                ),
            ),
            "torch": (
                re.compile(r"\btorch\."),
                Dependency(
                    name="torch",
                    import_statement="import torch",
                    confidence=0.95,
                    reason="Found torch usage",
                ),
            ),
            "requests": (
                re.compile(r"\brequests\."),
                Dependency(
                    name="requests",
                    import_statement="import requests",
                    confidence=0.95,
                    reason="Found requests usage",
                ),
            ),
            "os": (
                re.compile(r"\bos\."),
                Dependency(
                    name="os",
                    import_statement="import os",
                    confidence=0.95,
                    reason="Found os usage",
                ),
            ),
            "sys": (
                re.compile(r"\bsys\."),
                Dependency(
                    name="sys",
                    import_statement="import sys",
                    confidence=0.95,
                    reason="Found sys usage",
                ),
            ),
            "json": (
                re.compile(r"\bjson\."),
                Dependency(
                    name="json",
                    import_statement="import json",
                    confidence=0.95,
                    reason="Found json usage",
                ),
            ),
            "re": (
                re.compile(r"\bre\."),
                Dependency(
                    name="re",
                    import_statement="import re",
                    confidence=0.95,
                    reason="Found regex/re usage",
                ),
            ),
            "datetime": (
                re.compile(r"\bdatetime\."),
                Dependency(
                    name="datetime",
                    import_statement="from datetime import datetime",
                    confidence=0.90,
                    reason="Found datetime usage",
                ),
            ),
        }

    def extract(self, code: str, language: ProgrammingLanguage = ProgrammingLanguage.PYTHON) -> list[Dependency]:
        """Extract dependencies from code.

        Args:
            code: Code snippet to analyze.
            language: Programming language (for context-aware extraction).

        Returns:
            List of detected dependencies.
        """
        dependencies: dict[str, Dependency] = {}

        if language != ProgrammingLanguage.PYTHON:
            # Other languages have different dependency extraction logic
            return []

        # Match patterns
        for lib_name, (pattern, dep_template) in self.patterns.items():
            if pattern.search(code):
                dependencies[lib_name] = dep_template

        # Check for common class usages
        class_patterns = {
            "DataFrame": ("pandas", "import pandas as pd"),
            "array": ("numpy", "import numpy as np"),
            "Series": ("pandas", "import pandas as pd"),
            "Tensor": ("torch", "import torch"),
            "Model": ("tensorflow", "import tensorflow as tf"),
        }

        for class_name, (lib_name, import_stmt) in class_patterns.items():
            if re.search(rf"\b{class_name}\b", code) and lib_name not in dependencies:
                dependencies[lib_name] = Dependency(
                    name=lib_name,
                    import_statement=import_stmt,
                    confidence=0.75,
                    reason=f"Found {class_name} class usage",
                )

        return list(dependencies.values())
