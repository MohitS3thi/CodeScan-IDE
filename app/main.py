from __future__ import annotations

import ast
import base64
import io
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load .env before any other import touches os.getenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PIL import Image
from starlette.requests import Request

from code_scan.preprocessing import is_supported_upload, preprocess_image
from code_scan.ocr_pipeline import create_default_pipeline
from code_scan.language_detection import LanguageDetector, DependencyExtractor


# ========== REQUEST MODELS ==========
class DetectLanguageRequest(BaseModel):
    text: str


class ExtractDependenciesRequest(BaseModel):
    text: str
    language: str = "python"


class ValidateSyntaxRequest(BaseModel):
    text: str
    language: str = "python"


BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CodeScan - Handwritten Code Recognition IDE")

# Allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Initialize components
language_detector = LanguageDetector()
dependency_extractor = DependencyExtractor()

# Supported image formats that Pillow can reliably open
_SAFE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Initialize OCR pipeline (lazy - only if API key is available)
_ocr_pipeline = None


def get_ocr_pipeline():
    """Lazily initialize OCR pipeline."""
    global _ocr_pipeline
    if _ocr_pipeline is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set — OCR pipeline will be unavailable.")
            return None
        try:
            _ocr_pipeline = create_default_pipeline()
        except Exception as e:
            logger.warning(f"Could not initialize OCR pipeline: {e}")
    return _ocr_pipeline


def _is_safely_openable(filename: str) -> bool:
    """Return True only for formats reliably handled by Pillow on all platforms."""
    return Path(filename).suffix.lower() in _SAFE_IMAGE_EXTS


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "CodeScan",
        },
    )


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    if not is_supported_upload(filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PNG, JPG, JPEG, or WEBP.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Validate the file can actually be opened as an image
    if not _is_safely_openable(filename):
        raise HTTPException(
            status_code=400,
            detail=(
                "PDF and HEIC formats require additional system libraries. "
                "Please convert to PNG, JPG, or WEBP first."
            ),
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.verify()  # Ensure the image is not corrupted
        image = Image.open(io.BytesIO(content))  # Re-open after verify
        processed = preprocess_image(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process image: {exc}") from exc

    buffered = io.BytesIO()
    processed.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "filename": filename,
        "format": image.format or "PNG",
        "processed_size": list(processed.size),
        "preview": f"data:image/png;base64,{encoded}",
    }


@app.post("/api/recognize")
async def recognize_handwritten_text(
    file: UploadFile = File(...),
    context: str | None = Form(None),  # Fix: was Query(None), must be Form for multipart
) -> dict[str, Any]:
    """Recognize handwritten text from image using Phase 2 OCR pipeline.

    Args:
        file: Image file to process.
        context: Optional context submitted as a form field alongside the file.

    Returns:
        RecognitionResult with recognized text, language, dependencies, etc.
    """
    pipeline = get_ocr_pipeline()
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "OCR pipeline not available. "
                "Please set GOOGLE_API_KEY in your .env file."
            ),
        )

    filename = file.filename or ""
    if not is_supported_upload(filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PNG, JPG, JPEG, or WEBP.",
        )

    if not _is_safely_openable(filename):
        raise HTTPException(
            status_code=400,
            detail=(
                "PDF and HEIC formats require additional system libraries. "
                "Please convert to PNG, JPG, or WEBP first."
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        image = Image.open(io.BytesIO(content))
        result = pipeline.process(image, context=context)
        result_dict = result.to_dict()
        # Ensure language is serialized as a string
        if "language" in result_dict and hasattr(result_dict["language"], "value"):
            result_dict["language"] = result_dict["language"].value
        return result_dict
    except Exception as exc:
        logger.error(f"Recognition error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Recognition failed: {str(exc)}",
        ) from exc


@app.post("/api/detect-language")
async def detect_language(request: DetectLanguageRequest) -> dict[str, Any]:
    """Detect programming language from code snippet.

    Args:
        request: Request containing code text.

    Returns:
        Dictionary with detected language and confidence score.
    """
    result = language_detector.detect(request.text)
    return {
        "language": result["language"].value,
        "confidence": result["confidence"],
        "patterns_matched": list(result["patterns_matched"]),
        "scores": {
            lang.value: score for lang, score in result["scores"].items()
        },
    }


@app.post("/api/extract-dependencies")
async def extract_dependencies(request: ExtractDependenciesRequest) -> dict[str, Any]:
    """Extract dependencies from code.

    Args:
        request: Request containing code text and language.

    Returns:
        List of detected dependencies with import statements.
    """
    from code_scan.language_detection import ProgrammingLanguage

    try:
        lang_enum = ProgrammingLanguage(request.language.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {request.language}",
        )

    dependencies = dependency_extractor.extract(request.text, language=lang_enum)
    return {
        "language": request.language,
        "dependencies": [
            {
                "name": dep.name,
                "import_statement": dep.import_statement,
                "confidence": dep.confidence,
                "reason": dep.reason,
            }
            for dep in dependencies
        ],
    }


@app.post("/api/validate-syntax")
async def validate_syntax(request: ValidateSyntaxRequest) -> dict[str, Any]:
    """Validate syntax of recognized code.

    Currently supports Python. JavaScript returns a best-effort check.

    Args:
        request: Code text and language.

    Returns:
        Dictionary with valid flag, error message, and line/column for errors.
    """
    lang = request.language.lower()
    text = request.text

    if lang == "python":
        try:
            ast.parse(text)
            return {"valid": True, "language": lang, "error": None, "line": None, "column": None}
        except SyntaxError as exc:
            return {
                "valid": False,
                "language": lang,
                "error": exc.msg,
                "line": exc.lineno,
                "column": exc.offset,
            }
    else:
        # For other languages, return a neutral result (no server-side JS parser)
        return {
            "valid": None,
            "language": lang,
            "error": f"Syntax validation is not yet supported for '{lang}'.",
            "line": None,
            "column": None,
        }


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Status information about available services.
    """
    pipeline = get_ocr_pipeline()
    api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "healthy",
        "version": "2.0.0",
        "api_key_configured": api_key_configured,
        "services": {
            "image_preprocessing": True,
            "language_detection": True,
            "dependency_extraction": True,
            "syntax_validation": True,
            "ocr_pipeline": pipeline is not None,
        },
    }


@app.get("/api/info")
async def service_info() -> dict[str, Any]:
    """Get information about available services.

    Returns:
        Information about Phase 1 and Phase 2 capabilities.
    """
    return {
        "product_name": "CodeScan - AI Agentic IDE for Handwritten Code",
        "phase": "2",
        "capabilities": {
            "phase1": {
                "description": "Image preprocessing and normalization",
                "features": [
                    "Image upload (PNG, JPEG, WEBP)",
                    "Grayscale conversion",
                    "EXIF orientation correction",
                    "Vectorized image resizing with bounds",
                    "Contrast enhancement",
                    "Fast vectorized deskew (NumPy-accelerated)",
                ],
            },
            "phase2": {
                "description": "Handwritten text recognition and disambiguation",
                "features": [
                    "Vision-LLM integration (Google Gemini)",
                    "Handwritten text OCR",
                    "Character disambiguation (0 vs O, 1 vs l vs I, etc.)",
                    "Programming language detection",
                    "Dependency extraction",
                    "Indentation analysis",
                    "Python syntax validation",
                ],
            },
        },
        "endpoints": {
            "POST /api/upload": "Phase 1: Upload and preprocess image",
            "POST /api/recognize": "Phase 2: Full OCR pipeline with recognition",
            "POST /api/detect-language": "Detect programming language from code",
            "POST /api/extract-dependencies": "Extract dependencies from code",
            "POST /api/validate-syntax": "Validate Python/JS code syntax",
            "GET /api/health": "Health check",
            "GET /api/info": "This endpoint",
        },
    }
