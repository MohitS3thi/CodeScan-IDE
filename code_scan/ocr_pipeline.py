"""Phase 2 OCR Pipeline: Orchestrates multimodal recognition and disambiguation.

This module integrates vision-LLM, language detection, and disambiguation
into a complete handwritten code recognition pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from code_scan.disambiguation import HandwritingDisambiguator, IndentationAnalyzer
from code_scan.language_detection import DependencyExtractor, LanguageDetector, ProgrammingLanguage
from code_scan.vision_llm import OCRFactory, VisionLLMProvider

ImageLike = str | Path | Image.Image
LLMProviderType = Literal["google", "openai", "anthropic"]


@dataclass
class RecognitionResult:
    """Result from handwritten code recognition."""

    raw_text: str
    disambiguated_text: str
    language: ProgrammingLanguage
    confidence: float
    
    # Recognition details
    regions: list[dict[str, Any]]
    ambiguities: list[dict[str, Any]]
    corrections: list[dict[str, Any]]
    
    # Code structure
    dependencies: list[dict[str, Any]]
    indentation: dict[str, Any]
    
    # Metadata
    metadata: dict[str, Any]
    provider_used: str
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class OCRPipeline:
    """End-to-end OCR pipeline for handwritten code recognition."""

    def __init__(
        self,
        llm_provider: LLMProviderType = "google",
        api_key: str | None = None,
    ):
        """Initialize the OCR pipeline.

        Args:
            llm_provider: Vision-LLM provider to use ('google', 'openai', 'anthropic').
            api_key: Optional API key. If None, loads from environment.
        """
        self.llm_provider_type = llm_provider
        
        # Initialize components
        factory = OCRFactory()
        self.vision_llm: VisionLLMProvider = factory.create(
            provider=llm_provider,
            api_key=api_key,
        )
        
        self.disambiguator = HandwritingDisambiguator()
        self.language_detector = LanguageDetector()
        self.dependency_extractor = DependencyExtractor()
        self.indentation_analyzer = IndentationAnalyzer()

    def process(
        self,
        image: ImageLike,
        context: str | None = None,
        confidence_threshold: float = 0.7,
    ) -> RecognitionResult:
        """Process handwritten image through complete OCR pipeline.

        Args:
            image: Image file path, PIL Image, or base64 string.
            context: Optional contextual prompt (e.g., "Python code from whiteboard").
            confidence_threshold: Minimum confidence for recognized text.

        Returns:
            RecognitionResult containing all extracted and processed information.
        """
        # Step 1: Vision-LLM recognition
        recognition_data = self.vision_llm.recognize_text(
            image,
            context=context,
            confidence_threshold=confidence_threshold,
        )
        
        raw_text = recognition_data.get("text", "")
        regions = recognition_data.get("regions", [])
        vision_ambiguities = recognition_data.get("ambiguities", [])
        
        # Step 2: Language detection
        lang_result = self.language_detector.detect(raw_text)
        detected_language = lang_result["language"]
        
        # Step 3: Handwriting disambiguation
        disambig_result = self.disambiguator.disambiguate(
            raw_text,
            context_language=detected_language.value,
        )
        disambiguated_text = disambig_result["disambiguated_text"]
        corrections = [asdict(c) for c in disambig_result["corrections"]]
        
        # Step 4: Dependency extraction
        dependencies = self.dependency_extractor.extract(
            disambiguated_text,
            language=detected_language,
        )
        dependencies_list = [
            {
                "name": dep.name,
                "import_statement": dep.import_statement,
                "confidence": dep.confidence,
                "reason": dep.reason,
            }
            for dep in dependencies
        ]
        
        # Step 5: Indentation analysis
        indentation = self.indentation_analyzer.detect_indentation_style(
            disambiguated_text
        )
        
        # Step 6: Extract metadata
        metadata = self.vision_llm.extract_metadata(image)
        
        # Step 7: Additional vision-LLM disambiguation if needed
        if vision_ambiguities or corrections:
            vision_disambig = self.vision_llm.disambiguate_text(
                image,
                disambiguated_text,
            )
            additional_corrections = vision_disambig.get("corrections", [])
            disambiguated_text = vision_disambig.get("disambiguated_text", disambiguated_text)
            corrections.extend(additional_corrections)
        
        # Compute overall confidence
        overall_confidence = min(
            lang_result.get("confidence", 0.5),
            disambig_result.get("confidence", 0.7),
            recognition_data.get("confidence", 0.5),
            (metadata.get("estimated_quality", 0.5) if metadata else 0.5),
        )
        
        return RecognitionResult(
            raw_text=raw_text,
            disambiguated_text=disambiguated_text,
            language=detected_language,
            confidence=overall_confidence,
            regions=regions,
            ambiguities=vision_ambiguities,
            corrections=corrections,
            dependencies=dependencies_list,
            indentation=indentation,
            metadata=metadata,
            provider_used=self.llm_provider_type,
        )


class BatchOCRProcessor:
    """Process multiple images through OCR pipeline."""

    def __init__(self, pipeline: OCRPipeline):
        """Initialize batch processor.

        Args:
            pipeline: OCRPipeline instance to use for processing.
        """
        self.pipeline = pipeline

    def process_directory(
        self,
        directory: Path | str,
        pattern: str = "*.png",
        context: str | None = None,
    ) -> dict[str, RecognitionResult]:
        """Process all images in a directory.

        Args:
            directory: Directory containing images.
            pattern: Glob pattern for image files (default: "*.png").
            context: Optional context for all images.

        Returns:
            Dictionary mapping file paths to RecognitionResult objects.
        """
        directory = Path(directory)
        results = {}
        
        for image_path in directory.glob(pattern):
            try:
                result = self.pipeline.process(image_path, context=context)
                results[str(image_path)] = result
            except Exception as e:
                results[str(image_path)] = {
                    "error": str(e),
                    "file": str(image_path),
                }
        
        return results

    def process_list(
        self,
        images: list[ImageLike],
        contexts: list[str] | None = None,
    ) -> list[RecognitionResult]:
        """Process list of images.

        Args:
            images: List of image paths or PIL Images.
            contexts: Optional list of contexts (one per image).

        Returns:
            List of RecognitionResult objects.
        """
        results = []
        contexts = contexts or [None] * len(images)
        
        for image, context in zip(images, contexts):
            try:
                result = self.pipeline.process(image, context=context)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                })
        
        return results


class OCRPipelineBuilder:
    """Fluent builder for configuring OCR pipeline."""

    def __init__(self):
        """Initialize builder."""
        self.llm_provider = "google"
        self.api_key = None
        self.confidence_threshold = 0.7
        self.custom_context = None

    def with_provider(self, provider: LLMProviderType) -> OCRPipelineBuilder:
        """Set vision-LLM provider.

        Args:
            provider: Provider type ('google', 'openai', 'anthropic').

        Returns:
            Self for chaining.
        """
        self.llm_provider = provider
        return self

    def with_api_key(self, api_key: str) -> OCRPipelineBuilder:
        """Set API key.

        Args:
            api_key: API key for the provider.

        Returns:
            Self for chaining.
        """
        self.api_key = api_key
        return self

    def with_confidence_threshold(self, threshold: float) -> OCRPipelineBuilder:
        """Set confidence threshold.

        Args:
            threshold: Minimum confidence score (0-1).

        Returns:
            Self for chaining.
        """
        self.confidence_threshold = threshold
        return self

    def with_context(self, context: str) -> OCRPipelineBuilder:
        """Set default context.

        Args:
            context: Context prompt for recognition.

        Returns:
            Self for chaining.
        """
        self.custom_context = context
        return self

    def build(self) -> OCRPipeline:
        """Build OCRPipeline instance.

        Returns:
            Configured OCRPipeline.
        """
        return OCRPipeline(
            llm_provider=self.llm_provider,
            api_key=self.api_key,
        )


def create_default_pipeline() -> OCRPipeline:
    """Create pipeline with default configuration.

    Returns:
        OCRPipeline with default settings (Google Gemini provider).
    """
    return OCRPipeline(llm_provider="google")
