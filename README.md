# CodeScan

CodeScan is a FastAPI application for handwritten code recognition, language detection, preprocessing, and dependency extraction.

## Features

- Upload and preprocess images
- OCR-based handwritten code recognition
- Programming language detection
- Dependency extraction for common libraries and frameworks
- Browser-based interface for image and code analysis
- Batch processing support

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

2. Activate it:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Configure environment variables:

Copy the example file and update it:

```bash
copy .env.example .env
```

Then set:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GEMINI_MODEL=gemini-3.6-flash
```

5. Start the application:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

6. Open the app in the browser:

```text
http://localhost:8001/
```

