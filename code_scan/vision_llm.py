"""Multimodal vision-language model abstraction layer for Phase 2.

This module provides a unified interface for interacting with different vision-LLM providers
(Google Gemini, OpenAI GPT-4o, Anthropic Claude) for handwritten code recognition and OCR.

Supported providers:
- google: Google's Gemini 1.5 Pro/Flash
- openai: OpenAI's GPT-4o
- anthropic: Anthropic's Claude 3.5 Sonnet
"""

from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Union

from PIL import Image

try:
    import google.generativeai as genai  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    genai = None

ImageLike = Union[str, Path, Image.Image]
ProviderType = Literal["google", "openai", "anthropic"]


class VisionLLMProvider(ABC):
    """Abstract base class for vision-language model providers."""

    def __init__(self, api_key: str | None = None):
        """Initialize the provider with optional API key.
        
        Args:
            api_key: API key for the provider. If None, will attempt to load from environment.
        """
        self.api_key = api_key

    @abstractmethod
    def recognize_text(
        self,
        image: ImageLike,
        context: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Recognize text from a handwritten image.

        Args:
            image: Image file path, PIL Image, or base64-encoded image.
            context: Optional contextual prompt (e.g., "This is Python code from a whiteboard").
            confidence_threshold: Minimum confidence score to include recognized text.

        Returns:
            Dictionary with keys:
            - text: Recognized text content
            - confidence: Overall confidence score (0-1)
            - regions: List of recognized regions with individual confidence scores
            - language: Detected language code (e.g., 'python', 'markdown')
            - ambiguities: List of potential ambiguous characters and their alternatives
        """

    @abstractmethod
    def disambiguate_text(
        self,
        image: ImageLike,
        text: str,
        confusion_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Resolve common handwritten ambiguities in recognized text.

        Args:
            image: Original image for reference.
            text: Recognized text that may contain ambiguities.
            confusion_pairs: List of (likely_incorrect, likely_correct) pairs to check.
                If None, uses standard handwriting confusion pairs.

        Returns:
            Dictionary with keys:
            - disambiguated_text: Corrected text
            - corrections: List of applied corrections with confidence scores
            - unresolved_ambiguities: Characters that remain ambiguous
        """

    @abstractmethod
    def extract_metadata(
        self,
        image: ImageLike,
    ) -> dict[str, Any]:
        """Extract metadata about the handwritten content.

        Args:
            image: Image to analyze.

        Returns:
            Dictionary with keys:
            - dominant_language: Detected programming language (if code)
            - contains_math: Boolean indicating presence of mathematical notation
            - contains_diagrams: Boolean indicating presence of diagrams/drawings
            - handwriting_style: Estimated style (e.g., 'printed', 'cursive', 'mixed')
            - estimated_quality: Quality score (0-1)
            - page_density: Text density on page (0-1)
        """


class GoogleGeminiProvider(VisionLLMProvider):
    """Google Gemini vision-language model provider."""

    def __init__(self, api_key: str | None = None):
        """Initialize Google Gemini provider.
        
        Args:
            api_key: Google API key. If None, loads from GOOGLE_API_KEY environment variable.
        """
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        self.model_name = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-3.6-flash")
        self.genai = None
        self._import_client()

    def _import_client(self) -> None:
        """Lazy import Google AI client."""
        try:
            if genai is not None:
                self.genai = genai
                self.genai.configure(api_key=self.api_key)
                return

            import google.generativeai as legacy_genai  # type: ignore

            self.genai = legacy_genai
            self.genai.configure(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(
                "google-generativeai is required for GoogleGeminiProvider. "
                "Install it with: pip install google-generativeai"
            ) from e

    def _image_to_base64(self, image: ImageLike) -> tuple[str, str]:
        """Convert image to base64 and return with media type.
        
        Returns:
            Tuple of (base64_string, media_type)
        """
        if isinstance(image, str):
            with open(image, "rb") as f:
                image_data = f.read()
        elif isinstance(image, Path):
            with open(image, "rb") as f:
                image_data = f.read()
        elif isinstance(image, Image.Image):
            import io
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_data = buffer.getvalue()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        base64_str = base64.b64encode(image_data).decode("utf-8")
        return base64_str, "image/png"

    def recognize_text(
        self,
        image: ImageLike,
        context: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Recognize handwritten text using Gemini 1.5 Pro."""
        base64_image, media_type = self._image_to_base64(image)

        prompt = """Analyze this handwritten document and extract all text content.

Provide your response in the following JSON format:
{
    "text": "Full recognized text exactly as written",
    "confidence": 0.95,
    "regions": [
        {"text": "line or block", "confidence": 0.92, "bbox": [x, y, w, h]},
    ],
    "language": "python|markdown|text|other",
    "ambiguities": [
        {"position": 10, "character": "0", "alternatives": ["O", "o"], "confidence": 0.6}
    ]
}

Be precise about:
- 0 (zero) vs O (letter)
- 1 (one) vs l (lowercase L) vs I (uppercase i) vs | (pipe)
- 5 vs S, 2 vs Z, 9 vs g vs q
- Indentation patterns (spaces vs tabs)
"""
        if context:
            prompt += f"\n\nContext: {context}"

        try:
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                [
                    {
                        "text": prompt,
                    },
                    {
                        "mime_type": media_type,
                        "data": base64_image,
                    },
                ]
            )

            response_text = response.text
            # Try to parse as JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Extract JSON from markdown code blocks if needed
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    result = {
                        "text": response_text,
                        "confidence": 0.5,
                        "regions": [],
                        "language": "text",
                        "ambiguities": [],
                    }

            # Filter by confidence threshold
            if "regions" in result:
                result["regions"] = [
                    r for r in result["regions"] if r.get("confidence", 1.0) >= confidence_threshold
                ]

            return result
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "regions": [],
                "language": "text",
                "ambiguities": [],
                "error": str(e),
            }

    def disambiguate_text(
        self,
        image: ImageLike,
        text: str,
        confusion_pairs: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Resolve handwritten ambiguities using Gemini vision analysis."""
        base64_image, media_type = self._image_to_base64(image)

        if confusion_pairs is None:
            confusion_pairs = [
                ("0", "O"),
                ("1", "l"),
                ("1", "I"),
                ("5", "S"),
                ("2", "Z"),
                ("9", "g"),
                ("9", "q"),
            ]

        pairs_json = json.dumps(confusion_pairs)
        prompt = f"""Looking at this handwritten document, help disambiguate the following text.

Original recognized text:
{text}

Common confusion pairs to check: {pairs_json}

Provide a JSON response:
{{
    "disambiguated_text": "corrected text",
    "corrections": [
        {{"position": 5, "original": "0", "corrected": "O", "confidence": 0.85}}
    ],
    "unresolved_ambiguities": ["character at position 12"]
}}

Focus on:
1. Context clues (is this code? math? prose?)
2. Handwriting style and consistency
3. Neighboring characters and word context
"""

        try:
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                [
                    {
                        "text": prompt,
                    },
                    {
                        "mime_type": media_type,
                        "data": base64_image,
                    },
                ]
            )

            response_text = response.text
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    result = {
                        "disambiguated_text": text,
                        "corrections": [],
                        "unresolved_ambiguities": [],
                    }

            return result
        except Exception as e:
            return {
                "disambiguated_text": text,
                "corrections": [],
                "unresolved_ambiguities": [],
                "error": str(e),
            }

    def extract_metadata(
        self,
        image: ImageLike,
    ) -> dict[str, Any]:
        """Extract metadata about handwritten content."""
        base64_image, media_type = self._image_to_base64(image)

        prompt = """Analyze this handwritten document and extract metadata.

Provide JSON:
{
    "dominant_language": "python|markdown|text|math|sql|other",
    "contains_math": true/false,
    "contains_diagrams": true/false,
    "handwriting_style": "printed|cursive|mixed",
    "estimated_quality": 0.85,
    "page_density": 0.6
}
"""

        try:
            model = self.genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                [
                    {
                        "text": prompt,
                    },
                    {
                        "mime_type": media_type,
                        "data": base64_image,
                    },
                ]
            )

            response_text = response.text
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    result = json.loads(json_str)
                else:
                    result = self._default_metadata()

            return result
        except Exception as e:
            metadata = self._default_metadata()
            metadata["error"] = str(e)
            return metadata

    @staticmethod
    def _default_metadata() -> dict[str, Any]:
        """Return default metadata structure."""
        return {
            "dominant_language": "text",
            "contains_math": False,
            "contains_diagrams": False,
            "handwriting_style": "mixed",
            "estimated_quality": 0.5,
            "page_density": 0.5,
        }


class OCRFactory:
    """Factory for creating vision-LLM provider instances."""

    _providers: dict[ProviderType, type[VisionLLMProvider]] = {
        "google": GoogleGeminiProvider,
    }

    @classmethod
    def create(
        self,
        provider: ProviderType = "google",
        api_key: str | None = None,
    ) -> VisionLLMProvider:
        """Create a vision-LLM provider instance.

        Args:
            provider: Provider type ('google', 'openai', 'anthropic').
            api_key: Optional API key. If None, loads from environment.

        Returns:
            Initialized provider instance.

        Raises:
            ValueError: If provider type is not supported.
        """
        if provider not in self._providers:
            supported = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unsupported provider: {provider}. Supported: {supported}"
            )

        provider_class = self._providers[provider]
        return provider_class(api_key)

    @classmethod
    def register_provider(
        self,
        name: ProviderType,
        provider_class: type[VisionLLMProvider],
    ) -> None:
        """Register a custom provider class.

        Args:
            name: Provider name/identifier.
            provider_class: Provider class to register.
        """
        self._providers[name] = provider_class
