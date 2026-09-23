# CodeScan IDE

CodeScan is a FastAPI application for handwritten code recognition, language detection, preprocessing, and dependency extraction powered by Google Gemini.

## Features

- Upload and preprocess images (PNG, JPG, WEBP)
- Single-pass multimodal OCR — Google Gemini extracts code, language, dependencies, and metadata in **one API call** (~6-8s latency)
- Vectorized image preprocessing (NumPy-accelerated deskewing — 300x faster than v1)
- Programming language detection
- Dependency extraction for common libraries and frameworks
- Python syntax validation
- Browser-based IDE: tabbed code inspector, copy/download buttons, syntax badge, dual-pane layout
- Live camera capture for scanning code directly from a device camera
- Batch processing support with CSV export
- Health endpoint reporting API key status and service availability

## Runtime structure

```text
CodeScan/
├── app/
│   ├── __init__.py
│   └── main.py
├── code_scan/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── language_detection.py
│   ├── disambiguation.py
│   ├── ocr_pipeline.py
│   └── vision_llm.py
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
├── tests/
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_disambiguation.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── .venv/
└── .env
```

## Setup

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Configure environment variables:

```bash
copy .env.example .env
```

Then edit `.env` and set:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GEMINI_MODEL=gemini-3.6-flash
```

5. Run tests to verify everything works:

```bash
python -m pytest tests/ -v
```

6. Start the application:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

7. Open in browser:

```
http://localhost:8001/
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check with service status and API key flag |
| `GET` | `/api/info` | Capability information |
| `POST` | `/api/upload` | Preprocess and validate image |
| `POST` | `/api/recognize` | Full OCR pipeline (requires `GOOGLE_API_KEY`) |
| `POST` | `/api/detect-language` | Detect programming language from code text |
| `POST` | `/api/extract-dependencies` | Extract import statements from code |
| `POST` | `/api/validate-syntax` | Validate Python syntax (returns line/column on error) |

## Running Tests

```bash
python -m pytest tests/ -v
```

50 tests across preprocessing, disambiguation engine, and all API endpoints.
