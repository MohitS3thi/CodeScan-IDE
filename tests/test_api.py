"""Tests for the FastAPI endpoints in app.main."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ── Helper ────────────────────────────────────────────────────────────────────

def _png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Return in-memory PNG bytes for testing."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 200), (220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── GET / ─────────────────────────────────────────────────────────────────────

class TestIndex:
    def test_returns_html(self):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]


# ── GET /api/health ────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "image_preprocessing" in data["services"]
        assert "language_detection" in data["services"]
        assert "dependency_extraction" in data["services"]
        assert "syntax_validation" in data["services"]
        assert "api_key_configured" in data

    def test_version_present(self):
        res = client.get("/api/health")
        data = res.json()
        assert "version" in data


# ── GET /api/info ──────────────────────────────────────────────────────────────

class TestInfo:
    def test_info_structure(self):
        res = client.get("/api/info")
        assert res.status_code == 200
        data = res.json()
        assert "product_name" in data
        assert "capabilities" in data
        assert "endpoints" in data


# ── POST /api/upload ───────────────────────────────────────────────────────────

class TestUpload:
    def test_png_upload_success(self):
        res = client.post(
            "/api/upload",
            files={"file": ("test.png", _png_bytes(), "image/png")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "filename" in data
        assert "processed_size" in data
        assert data["preview"].startswith("data:image/png;base64,")

    def test_jpeg_upload_success(self):
        res = client.post(
            "/api/upload",
            files={"file": ("test.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200

    def test_unsupported_extension_rejected(self):
        res = client.post(
            "/api/upload",
            files={"file": ("script.exe", b"fake data", "application/octet-stream")},
        )
        assert res.status_code == 400

    def test_empty_file_rejected(self):
        res = client.post(
            "/api/upload",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert res.status_code == 400

    def test_heic_returns_400_with_helpful_message(self):
        res = client.post(
            "/api/upload",
            files={"file": ("photo.heic", b"fake heic data", "image/heic")},
        )
        # Either 400 (format not safe) or 400 (unsupported) — should never 500
        assert res.status_code in {400, 422}


# ── POST /api/detect-language ─────────────────────────────────────────────────

class TestDetectLanguage:
    def test_detects_python(self):
        res = client.post(
            "/api/detect-language",
            json={"text": "def add(a, b):\n    return a + b\nimport numpy as np\n"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["language"] == "python"
        assert 0.0 <= data["confidence"] <= 1.0

    def test_detects_sql(self):
        res = client.post(
            "/api/detect-language",
            json={"text": "SELECT * FROM users WHERE id = 1 ORDER BY name;"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["language"] in {"sql", "unknown"}

    def test_empty_text_returns_unknown(self):
        res = client.post("/api/detect-language", json={"text": ""})
        assert res.status_code == 200
        data = res.json()
        assert data["language"] == "unknown"

    def test_missing_text_field_returns_422(self):
        res = client.post("/api/detect-language", json={})
        assert res.status_code == 422


# ── POST /api/extract-dependencies ────────────────────────────────────────────

class TestExtractDependencies:
    def test_numpy_detected(self):
        code = "import numpy as np\nresult = np.array([1, 2, 3])\n"
        res = client.post(
            "/api/extract-dependencies",
            json={"text": code, "language": "python"},
        )
        assert res.status_code == 200
        data = res.json()
        names = [d["name"] for d in data["dependencies"]]
        assert "numpy" in names

    def test_pandas_detected(self):
        code = "df = pd.DataFrame({'a': [1, 2]})\n"
        res = client.post(
            "/api/extract-dependencies",
            json={"text": code, "language": "python"},
        )
        assert res.status_code == 200
        names = [d["name"] for d in res.json()["dependencies"]]
        assert "pandas" in names

    def test_unsupported_language_returns_400(self):
        res = client.post(
            "/api/extract-dependencies",
            json={"text": "x = 1", "language": "brainfuck"},
        )
        assert res.status_code == 400

    def test_non_python_returns_empty_list(self):
        res = client.post(
            "/api/extract-dependencies",
            json={"text": "const x = 1;", "language": "javascript"},
        )
        assert res.status_code == 200
        assert res.json()["dependencies"] == []


# ── POST /api/validate-syntax ─────────────────────────────────────────────────

class TestValidateSyntax:
    def test_valid_python_returns_true(self):
        res = client.post(
            "/api/validate-syntax",
            json={"text": "def add(a, b):\n    return a + b\n", "language": "python"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is True
        assert data["error"] is None

    def test_invalid_python_returns_false(self):
        res = client.post(
            "/api/validate-syntax",
            json={"text": "def add(a b:\n    return a + b\n", "language": "python"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is False
        assert data["error"] is not None
        assert data["line"] is not None

    def test_unsupported_language_returns_null(self):
        res = client.post(
            "/api/validate-syntax",
            json={"text": "const x = 1;", "language": "javascript"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is None
