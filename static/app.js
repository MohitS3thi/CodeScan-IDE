'use strict';
const { useEffect, useMemo, useRef, useState, useCallback } = React;

// ============================================================
// Tiny utility helpers
// ============================================================
function cls(...parts) {
  return parts.filter(Boolean).join(' ');
}

// ============================================================
// Sub-components
// ============================================================

/** Status pill that shows success / error / loading messages */
function StatusPill({ message, kind }) {
  if (!message) return null;
  return React.createElement('div', { className: cls('status-pill', kind) }, message);
}

/** Animated pulse dot for the health footer */
function StatusDot({ ok }) {
  return React.createElement('span', {
    className: cls('status-dot', ok ? 'ready' : 'unavailable'),
    title: ok ? 'Available' : 'Unavailable',
  });
}

/** React Error Boundary — prevents one broken panel from crashing everything */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: '' };
  }
  static getDerivedStateFromError(err) {
    return { hasError: true, message: String(err && err.message ? err.message : err) };
  }
  componentDidCatch(err, info) {
    console.error('ErrorBoundary caught:', err, info);
  }
  render() {
    if (this.state.hasError) {
      return React.createElement(
        'div',
        { className: 'error-boundary' },
        React.createElement('p', null, '⚠️ A rendering error occurred.'),
        React.createElement('code', null, this.state.message),
        React.createElement(
          'button',
          {
            className: 'secondary-button',
            style: { marginTop: 12 },
            onClick: () => this.setState({ hasError: false, message: '' }),
          },
          'Retry',
        ),
      );
    }
    return this.props.children;
  }
}

/** Syntax-highlighted code block (lightweight keyword colouring via CSS spans) */
function CodeBlock({ code, language, label }) {
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code || '').then(() => {
      // Briefly toggle a "Copied!" label — handled in parent state to keep pure
    });
  }, [code]);

  if (!code) return null;

  return React.createElement(
    'div',
    { className: 'code-block-wrap' },
    React.createElement(
      'div',
      { className: 'code-block-header' },
      React.createElement('span', { className: 'code-lang-badge' }, language || 'text'),
      label && React.createElement('span', { className: 'code-block-label' }, label),
      React.createElement(
        'button',
        { className: 'copy-button', type: 'button', onClick: handleCopy, title: 'Copy to clipboard' },
        '⎘ Copy',
      ),
    ),
    React.createElement('pre', { className: 'code-block' }, React.createElement('code', null, code)),
  );
}

/** Single tabbed panel used inside the result inspector */
function TabPanel({ tabs, activeTab, onTabChange, children }) {
  return React.createElement(
    'div',
    { className: 'tab-panel' },
    React.createElement(
      'div',
      { className: 'tab-nav', role: 'tablist' },
      tabs.map((tab) =>
        React.createElement(
          'button',
          {
            key: tab.id,
            role: 'tab',
            className: cls('tab-button', activeTab === tab.id && 'active'),
            onClick: () => onTabChange(tab.id),
            'aria-selected': activeTab === tab.id,
          },
          tab.icon && React.createElement('span', { className: 'tab-icon' }, tab.icon),
          tab.label,
          tab.count != null &&
            React.createElement('span', { className: 'tab-badge' }, tab.count),
        ),
      ),
    ),
    React.createElement('div', { className: 'tab-content' }, children),
  );
}

// ============================================================
// Main App
// ============================================================
function App() {
  // Theme
  const [theme, setTheme] = useState(() =>
    localStorage.getItem('cs-theme') === 'dark' ? 'dark' : 'light',
  );

  // Source file state
  const [sourceFile, setSourceFile] = useState(null);
  const [uploadMetadata, setUploadMetadata] = useState(null);
  const [uploadPreview, setUploadPreview] = useState('');
  const [uploadStatus, setUploadStatus] = useState({ message: '', kind: '' });

  // Camera
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraStatus, setCameraStatus] = useState({ message: '', kind: '' });
  const [cameraBusy, setCameraBusy] = useState(false);
  const videoRef = useRef(null);
  const cameraStreamRef = useRef(null);

  // OCR Recognition
  const [contextText, setContextText] = useState('');
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [recognitionStatus, setRecognitionStatus] = useState({ message: '', kind: '' });
  const [activeResultTab, setActiveResultTab] = useState('code');
  const [copiedCode, setCopiedCode] = useState(false);

  // Code analysis panel
  const [codeInput, setCodeInput] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState({ message: '', kind: '' });
  const [analysisResult, setAnalysisResult] = useState(null);
  const [syntaxStatus, setSyntaxStatus] = useState(null); // {valid, error, line, column}

  // Batch processing
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchStatus, setBatchStatus] = useState({ message: '', kind: '' });
  const [batchResults, setBatchResults] = useState([]);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, visible: false });
  const [batchPreprocess, setBatchPreprocess] = useState(true);
  const [batchOcr, setBatchOcr] = useState(true);
  const [batchDetectLang, setBatchDetectLang] = useState(false);

  // Health
  const [health, setHealth] = useState(null);
  const [healthModalOpen, setHealthModalOpen] = useState(false);

  // ── Effects ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('cs-theme', theme);
  }, [theme]);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(
    () => () => {
      if (cameraStreamRef.current) {
        cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      }
    },
    [],
  );

  // Auto-validate Python syntax when codeInput changes (debounced)
  useEffect(() => {
    setSyntaxStatus(null);
    if (!codeInput.trim()) return;
    const timer = setTimeout(() => {
      validateSyntaxAuto(codeInput);
    }, 800);
    return () => clearTimeout(timer);
  }, [codeInput]);

  // ── API calls ────────────────────────────────────────────────────────────────
  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      const data = await res.json();
      setHealth(data);
    } catch {
      setHealth({
        status: 'error',
        version: '2.0.0',
        api_key_configured: false,
        services: {
          image_preprocessing: false,
          language_detection: false,
          dependency_extraction: false,
          syntax_validation: false,
          ocr_pipeline: false,
        },
      });
    }
  };

  const validateSyntaxAuto = async (code) => {
    try {
      const res = await fetch('/api/validate-syntax', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: code, language: 'python' }),
      });
      if (res.ok) {
        const data = await res.json();
        setSyntaxStatus(data);
      }
    } catch {
      // silent — syntax check is best-effort
    }
  };

  // ── Camera ────────────────────────────────────────────────────────────────────
  const stopCamera = () => {
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks().forEach((t) => t.stop());
      cameraStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOpen(false);
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus({ message: '❌ Camera not supported by this browser.', kind: 'error' });
      return;
    }
    setCameraBusy(true);
    setCameraStatus({ message: '⏳ Requesting camera access...', kind: 'loading' });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' } },
        audio: false,
      });
      cameraStreamRef.current = stream;
      setCameraOpen(true);
      setCameraStatus({ message: '✅ Camera ready. Position code and capture.', kind: 'success' });
      window.requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      });
    } catch (err) {
      setCameraStatus({ message: `❌ Unable to access camera: ${err.message}`, kind: 'error' });
    } finally {
      setCameraBusy(false);
    }
  };

  const captureCameraImage = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setCameraStatus({ message: '❌ Camera still loading. Try again.', kind: 'error' });
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        setCameraStatus({ message: '❌ Could not capture image.', kind: 'error' });
        return;
      }
      setSourceFile(new File([blob], 'camera-capture.png', { type: 'image/png' }));
      setUploadMetadata(null);
      setUploadPreview('');
      setCameraStatus({ message: '✅ Captured. Preprocess or recognize it below.', kind: 'success' });
      stopCamera();
    }, 'image/png');
  };

  // ── Upload / Recognize ────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!sourceFile) {
      setUploadStatus({ message: '❌ Choose a file first.', kind: 'error' });
      return;
    }
    setUploadStatus({ message: '⏳ Processing image…', kind: 'loading' });
    const fd = new FormData();
    fd.append('file', sourceFile);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: fd });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || 'Upload failed.');
      setUploadMetadata(payload);
      setUploadPreview(payload.preview || '');
      setUploadStatus({ message: '✅ Image preprocessed successfully.', kind: 'success' });
    } catch (err) {
      setUploadStatus({ message: `❌ ${err.message}`, kind: 'error' });
    }
  };

  const handleRecognize = async () => {
    if (!sourceFile) {
      setRecognitionStatus({ message: '❌ Choose a file first.', kind: 'error' });
      return;
    }
    setRecognitionStatus({ message: '⏳ Recognizing handwriting… (this may take ~10s)', kind: 'loading' });
    const fd = new FormData();
    fd.append('file', sourceFile);
    if (contextText.trim()) fd.append('context', contextText.trim());
    try {
      const res = await fetch('/api/recognize', { method: 'POST', body: fd });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || 'Recognition failed.');
      setRecognitionResult(payload);
      setRecognitionStatus({ message: '✅ Text recognized successfully.', kind: 'success' });
      if (payload.disambiguated_text) {
        setCodeInput(payload.disambiguated_text);
      }
      setActiveResultTab('code');
    } catch (err) {
      setRecognitionStatus({ message: `❌ ${err.message}`, kind: 'error' });
    }
  };

  // ── Code analysis ─────────────────────────────────────────────────────────────
  const detectLanguageFromText = async (text) => {
    const res = await fetch('/api/detect-language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Language detection failed.');
    return data;
  };

  const handleDetectLanguage = async () => {
    const text = codeInput.trim();
    if (!text) { setAnalysisStatus({ message: '❌ Enter code first.', kind: 'error' }); return; }
    setAnalysisStatus({ message: '⏳ Detecting language…', kind: 'loading' });
    try {
      const payload = await detectLanguageFromText(text);
      setAnalysisResult({ type: 'language', payload });
      setAnalysisStatus({ message: '✅ Language detected.', kind: 'success' });
    } catch (err) {
      setAnalysisStatus({ message: `❌ ${err.message}`, kind: 'error' });
    }
  };

  const handleExtractDependencies = async () => {
    const text = codeInput.trim();
    if (!text) { setAnalysisStatus({ message: '❌ Enter code first.', kind: 'error' }); return; }
    setAnalysisStatus({ message: '⏳ Extracting dependencies…', kind: 'loading' });
    try {
      const detected = await detectLanguageFromText(text);
      const language = detected.language || 'python';
      const res = await fetch('/api/extract-dependencies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || 'Extraction failed.');
      setAnalysisResult({ type: 'dependencies', payload });
      setAnalysisStatus({ message: '✅ Dependencies extracted.', kind: 'success' });
    } catch (err) {
      setAnalysisStatus({ message: `❌ ${err.message}`, kind: 'error' });
    }
  };

  const handleDownloadCode = () => {
    if (!codeInput) return;
    const lang = recognitionResult?.language || 'txt';
    const ext = { python: 'py', javascript: 'js', typescript: 'ts', csharp: 'cs', java: 'java', cpp: 'cpp', sql: 'sql', r: 'r', julia: 'jl' }[lang] || 'txt';
    const blob = new Blob([codeInput], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `recognized_code.${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(codeInput || '').then(() => {
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 2000);
    });
  };

  // ── Batch processing ──────────────────────────────────────────────────────────
  const handleBatchProcess = async () => {
    if (!batchFiles.length) {
      setBatchStatus({ message: '❌ Choose files first.', kind: 'error' });
      return;
    }
    setBatchStatus({ message: '⏳ Processing batch…', kind: 'loading' });
    setBatchProgress({ current: 0, total: batchFiles.length, visible: true });
    setBatchResults([]);
    const nextResults = [];

    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i];
      const result = { filename: file.name, status: 'processing', language: '—', preview: '', rawData: null };

      try {
        if (batchPreprocess || batchOcr) {
          const fd = new FormData();
          fd.append('file', file);
          const endpoint = batchOcr ? '/api/recognize' : '/api/upload';
          const res = await fetch(endpoint, { method: 'POST', body: fd });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();
          result.rawData = data;

          if (endpoint.includes('upload')) {
            result.preview = data.preview ? 'Preprocessed ✓' : 'Processed';
          } else {
            const txt = data.disambiguated_text || data.text || '';
            result.preview = txt ? txt.slice(0, 60) + (txt.length > 60 ? '…' : '') : 'Recognized ✓';
            if (data.language) result.language = typeof data.language === 'object' ? data.language.value || '?' : data.language;
          }

          if (batchDetectLang && !result.language || result.language === '—') {
            const det = await detectLanguageFromText(data.disambiguated_text || data.text || '');
            result.language = det.language || '—';
          }

          result.status = 'success';
        } else {
          result.status = 'skipped';
        }
      } catch (err) {
        result.status = 'error';
        result.preview = err.message;
      }

      nextResults.push(result);
      setBatchProgress({ current: i + 1, total: batchFiles.length, visible: true });
      setBatchResults([...nextResults]);
    }

    setBatchStatus({ message: '✅ Batch complete.', kind: 'success' });
  };

  const handleExportCsv = () => {
    const headers = ['Filename', 'Status', 'Language', 'Preview'];
    const rows = batchResults.map((r) => [r.filename, r.status, r.language, (r.preview || '').replace(/"/g, '""')]);
    const csv = [headers, ...rows].map((r) => r.map((c) => `"${c}"`).join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'batch-results.csv';
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // ── Render helpers ────────────────────────────────────────────────────────────
  const metaSummary = useMemo(() => {
    if (!uploadMetadata) return null;
    return [
      { label: 'File', value: uploadMetadata.filename },
      { label: 'Format', value: uploadMetadata.format },
      { label: 'Size', value: `${uploadMetadata.processed_size[0]} × ${uploadMetadata.processed_size[1]} px` },
    ];
  }, [uploadMetadata]);

  const renderResultTabs = () => {
    if (!recognitionResult) {
      return React.createElement(
        'div',
        { className: 'empty-state panel-empty' },
        React.createElement('span', { className: 'empty-icon' }, '🔍'),
        React.createElement('p', null, 'Recognition results appear here'),
        React.createElement('p', { className: 'empty-sub' }, 'Upload an image and click "Recognize text"'),
      );
    }

    const r = recognitionResult;
    const lang = typeof r.language === 'object' ? (r.language.value || 'text') : (r.language || 'text');
    const corrCount = Array.isArray(r.corrections) ? r.corrections.length : 0;
    const depCount = Array.isArray(r.dependencies) ? r.dependencies.length : 0;

    const tabs = [
      { id: 'code', label: 'Code', icon: '💻' },
      { id: 'raw', label: 'Raw OCR', icon: '🔍' },
      { id: 'corrections', label: 'Fixes', icon: '🔧', count: corrCount },
      { id: 'deps', label: 'Dependencies', icon: '📦', count: depCount },
      { id: 'meta', label: 'Metadata', icon: '📋' },
    ];

    let tabContent;
    if (activeResultTab === 'code') {
      tabContent = React.createElement(
        'div',
        { className: 'result-code-pane' },
        React.createElement(
          'div',
          { className: 'result-code-header' },
          React.createElement(
            'span',
            { className: 'lang-badge' },
            lang.toUpperCase(),
          ),
          React.createElement(
            'span',
            { className: 'confidence-badge' },
            `${((r.confidence || 0) * 100).toFixed(0)}% confidence`,
          ),
        ),
        React.createElement('pre', { className: 'code-block full' },
          React.createElement('code', null, r.disambiguated_text || r.raw_text || '(no text)')
        ),
      );
    } else if (activeResultTab === 'raw') {
      tabContent = React.createElement(
        'div',
        null,
        r.raw_text
          ? React.createElement('pre', { className: 'code-block full' }, React.createElement('code', null, r.raw_text))
          : React.createElement('p', { className: 'muted' }, 'No raw text available.'),
      );
    } else if (activeResultTab === 'corrections') {
      tabContent = corrCount > 0
        ? React.createElement(
            'ul',
            { className: 'correction-list' },
            r.corrections.map((c, i) => {
              // Safely extract fields whether they're nested objects or flat
              const orig = c.original ?? (c.from ?? '?');
              const corr = c.corrected ?? (c.to ?? '?');
              const conf = typeof c.confidence === 'number' ? `${(c.confidence * 100).toFixed(0)}%` : '';
              return React.createElement(
                'li',
                { key: i, className: 'correction-item' },
                React.createElement('code', { className: 'orig-char' }, orig),
                React.createElement('span', { className: 'arrow' }, ' → '),
                React.createElement('code', { className: 'corr-char' }, corr),
                conf && React.createElement('span', { className: 'conf-chip' }, conf),
              );
            }),
          )
        : React.createElement('p', { className: 'muted' }, 'No character corrections applied.');
    } else if (activeResultTab === 'deps') {
      tabContent = depCount > 0
        ? React.createElement(
            'ul',
            { className: 'dep-list' },
            r.dependencies.map((dep, i) => {
              // Fix: dep is a plain object — extract fields explicitly
              const name = dep.name || '?';
              const importStmt = dep.import_statement || dep.name || '?';
              const conf = typeof dep.confidence === 'number' ? `${(dep.confidence * 100).toFixed(0)}%` : '';
              const reason = dep.reason || '';
              return React.createElement(
                'li',
                { key: i, className: 'dep-item' },
                React.createElement(
                  'div',
                  { className: 'dep-import' },
                  React.createElement('code', null, importStmt),
                ),
                React.createElement(
                  'div',
                  { className: 'dep-meta' },
                  conf && React.createElement('span', { className: 'conf-chip' }, conf),
                  reason && React.createElement('span', { className: 'dep-reason' }, reason),
                ),
              );
            }),
          )
        : React.createElement('p', { className: 'muted' }, 'No dependencies detected.');
    } else if (activeResultTab === 'meta') {
      const meta = r.metadata || {};
      const indent = r.indentation || {};
      const entries = [
        ['Handwriting style', meta.handwriting_style],
        ['Estimated quality', meta.estimated_quality != null ? `${(meta.estimated_quality * 100).toFixed(0)}%` : null],
        ['Contains math', meta.contains_math != null ? String(meta.contains_math) : null],
        ['Contains diagrams', meta.contains_diagrams != null ? String(meta.contains_diagrams) : null],
        ['Page density', meta.page_density != null ? `${(meta.page_density * 100).toFixed(0)}%` : null],
        ['Indent style', indent.style],
        ['Indent size', indent.indent_size != null ? String(indent.indent_size) : null],
        ['Provider', r.provider_used],
      ].filter(([, v]) => v != null);
      tabContent = React.createElement(
        'table',
        { className: 'meta-table' },
        React.createElement(
          'tbody',
          null,
          entries.map(([k, v]) =>
            React.createElement(
              'tr',
              { key: k },
              React.createElement('th', null, k),
              React.createElement('td', null, v),
            ),
          ),
        ),
      );
    }

    return React.createElement(
      TabPanel,
      { tabs, activeTab: activeResultTab, onTabChange: setActiveResultTab },
      tabContent,
    );
  };

  const renderAnalysisResult = () => {
    if (!analysisResult) return React.createElement('p', { className: 'muted' }, 'Analysis results appear here…');

    if (analysisResult.type === 'language') {
      const { payload } = analysisResult;
      return React.createElement(
        'div',
        { className: 'analysis-result' },
        React.createElement(
          'div',
          { className: 'analysis-highlight' },
          React.createElement('span', { className: 'lang-badge large' }, payload.language.toUpperCase()),
          React.createElement('span', { className: 'confidence-badge large' }, `${(payload.confidence * 100).toFixed(1)}% confidence`),
        ),
        payload.patterns_matched?.length > 0
          ? React.createElement(
              'div',
              { className: 'pattern-chips' },
              payload.patterns_matched.map((p) => React.createElement('span', { key: p, className: 'pattern-chip' }, p)),
            )
          : null,
      );
    }

    // Dependencies
    const { payload } = analysisResult;
    return React.createElement(
      'div',
      { className: 'analysis-result' },
      React.createElement('p', null,
        React.createElement('strong', null, '📦 Language: '),
        payload.language,
      ),
      payload.dependencies?.length > 0
        ? React.createElement(
            'ul',
            { className: 'dep-list' },
            payload.dependencies.map((dep, i) =>
              React.createElement(
                'li',
                { key: i, className: 'dep-item' },
                React.createElement('code', null, dep.import_statement || dep.name),
                dep.confidence != null &&
                  React.createElement('span', { className: 'conf-chip' }, `${(dep.confidence * 100).toFixed(0)}%`),
              ),
            ),
          )
        : React.createElement('p', { className: 'muted' }, 'No dependencies detected.'),
    );
  };

  // ── Syntax badge ────────────────────────────────────────────────────────────
  const syntaxBadge = syntaxStatus == null
    ? null
    : syntaxStatus.valid === true
      ? React.createElement('span', { className: 'syntax-badge ok', title: 'No syntax errors' }, '✓ Syntax OK')
      : syntaxStatus.valid === false
        ? React.createElement(
            'span',
            {
              className: 'syntax-badge error',
              title: syntaxStatus.error + (syntaxStatus.line ? ` (line ${syntaxStatus.line})` : ''),
            },
            `✗ Line ${syntaxStatus.line || '?'}: ${syntaxStatus.error}`,
          )
        : null;

  // ── Top-level render ──────────────────────────────────────────────────────────
  const ocrReady = health?.services?.ocr_pipeline;
  const apiKeyConfigured = health?.api_key_configured;

  return React.createElement(
    'div',
    { className: 'app-shell' },
    React.createElement(
      'div',
      { className: 'shell' },

      // ── Header ──────────────────────────────────────────────────────────────
      React.createElement(
        'header',
        { className: 'topbar' },
        React.createElement(
          'div',
          { className: 'brand' },
          React.createElement('div', { className: 'brand-logo' }, '⟨/⟩'),
          React.createElement(
            'div',
            null,
            React.createElement('h1', null, 'CodeScan'),
            React.createElement('p', { className: 'subtitle' }, 'AI-powered handwritten code recognition'),
          ),
        ),
        React.createElement(
          'div',
          { className: 'header-actions' },
          !apiKeyConfigured && React.createElement(
            'span',
            { className: 'api-warning', title: 'Set GOOGLE_API_KEY in your .env file to enable OCR' },
            '⚠ API key not set',
          ),
          React.createElement(
            'button',
            {
              id: 'health-btn',
              className: 'icon-button',
              type: 'button',
              onClick: () => setHealthModalOpen(true),
              title: 'System health',
            },
            '⚙️',
          ),
          React.createElement(
            'button',
            {
              id: 'theme-toggle-btn',
              className: 'icon-button',
              type: 'button',
              onClick: () => setTheme(theme === 'dark' ? 'light' : 'dark'),
              title: 'Toggle theme',
            },
            theme === 'dark' ? '☀️' : '🌙',
          ),
        ),
      ),

      // ── Main workspace ────────────────────────────────────────────────────────
      React.createElement(
        'main',
        { className: 'workspace-grid' },

        // ─ Column 1: Input ─────────────────────────────────────────────────────
        React.createElement(
          'section',
          { className: 'card', 'aria-label': 'Image input' },
          React.createElement('h2', null, '📁 Image Input'),

          // Drop zone
          React.createElement(
            'label',
            { className: 'dropzone', htmlFor: 'source-input' },
            React.createElement('input', {
              id: 'source-input',
              type: 'file',
              accept: '.png,.jpg,.jpeg,.webp',
              onChange: (e) => {
                setSourceFile(e.target.files?.[0] ?? null);
                setUploadMetadata(null);
                setUploadPreview('');
                setUploadStatus({ message: '', kind: '' });
              },
            }),
            React.createElement(
              'div',
              { className: 'dropzone-content' },
              React.createElement('span', { className: 'dropzone-icon' }, '📸'),
              React.createElement(
                'span',
                { className: 'dropzone-text' },
                sourceFile ? sourceFile.name : 'Choose image or drag & drop',
              ),
              React.createElement('span', { className: 'dropzone-hint' }, '.png  .jpg  .webp'),
            ),
          ),

          // Camera controls
          React.createElement(
            'div',
            { className: 'camera-controls' },
            cameraOpen
              ? React.createElement(
                  React.Fragment,
                  null,
                  React.createElement('video', {
                    ref: videoRef,
                    className: 'camera-preview',
                    autoPlay: true,
                    playsInline: true,
                    muted: true,
                  }),
                  React.createElement(
                    'div',
                    { className: 'button-row' },
                    React.createElement(
                      'button',
                      { className: 'primary-button', type: 'button', onClick: captureCameraImage },
                      '📷 Capture',
                    ),
                    React.createElement(
                      'button',
                      { className: 'secondary-button', type: 'button', onClick: stopCamera },
                      '✕ Close',
                    ),
                  ),
                )
              : React.createElement(
                  'button',
                  {
                    id: 'camera-btn',
                    className: 'secondary-button camera-button',
                    type: 'button',
                    onClick: startCamera,
                    disabled: cameraBusy,
                  },
                  cameraBusy ? '⏳ Starting…' : '📷 Use camera',
                ),
          ),
          StatusPill(cameraStatus),

          // Context
          React.createElement(
            'div',
            { className: 'field-group' },
            React.createElement('label', { htmlFor: 'context-input' }, 'Context hint'),
            React.createElement('input', {
              id: 'context-input',
              type: 'text',
              className: 'text-input',
              placeholder: "e.g. 'Python ML code on whiteboard'",
              value: contextText,
              onChange: (e) => setContextText(e.target.value),
            }),
          ),

          // Action buttons
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement(
              'button',
              { id: 'preprocess-btn', className: 'secondary-button', type: 'button', onClick: handleUpload },
              '⚙️ Preprocess',
            ),
            React.createElement(
              'button',
              {
                id: 'recognize-btn',
                className: 'primary-button',
                type: 'button',
                onClick: handleRecognize,
                disabled: !ocrReady,
                title: ocrReady ? '' : 'OCR pipeline unavailable — check API key',
              },
              '🤖 Recognize text',
            ),
          ),
          StatusPill(uploadStatus),
          StatusPill(recognitionStatus),

          // Image preview
          React.createElement(
            'div',
            { className: 'image-stage' },
            uploadPreview
              ? React.createElement('img', {
                  src: uploadPreview,
                  alt: 'Processed preview',
                  className: 'preview-image',
                })
              : React.createElement(
                  'div',
                  { className: 'empty-state' },
                  React.createElement('span', null, '🖼️'),
                  React.createElement('p', null, 'Preprocessed preview appears here'),
                ),
          ),

          // Metadata
          metaSummary &&
            React.createElement(
              'div',
              { className: 'meta-box' },
              React.createElement('h3', null, 'Image details'),
              React.createElement(
                'table',
                { className: 'meta-table compact' },
                React.createElement(
                  'tbody',
                  null,
                  metaSummary.map(({ label, value }) =>
                    React.createElement(
                      'tr',
                      { key: label },
                      React.createElement('th', null, label),
                      React.createElement('td', null, value),
                    ),
                  ),
                ),
              ),
            ),
        ),

        // ─ Column 2: Recognition Results ────────────────────────────────────────
        React.createElement(
          'section',
          { className: 'card', 'aria-label': 'Recognition results' },
          React.createElement('h2', null, '🤖 Recognition Results'),
          React.createElement(
            ErrorBoundary,
            null,
            renderResultTabs(),
          ),
        ),

        // ─ Column 3: Code Analysis ───────────────────────────────────────────────
        React.createElement(
          'section',
          { className: 'card', 'aria-label': 'Code analysis' },
          React.createElement('h2', null, '🔍 Code Analysis'),
          React.createElement(
            'div',
            { className: 'field-group' },
            React.createElement(
              'div',
              { className: 'code-label-row' },
              React.createElement('label', { htmlFor: 'code-input' }, 'Code editor'),
              syntaxBadge,
            ),
            React.createElement('textarea', {
              id: 'code-input',
              className: 'code-textarea',
              value: codeInput,
              onChange: (e) => setCodeInput(e.target.value),
              placeholder: 'Recognized code auto-fills here, or paste manually…',
              spellCheck: false,
            }),
          ),
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement(
              'button',
              {
                id: 'copy-btn',
                className: 'secondary-button icon-left',
                type: 'button',
                onClick: handleCopyCode,
                disabled: !codeInput,
              },
              copiedCode ? '✅ Copied!' : '⎘ Copy',
            ),
            React.createElement(
              'button',
              {
                id: 'download-btn',
                className: 'secondary-button icon-left',
                type: 'button',
                onClick: handleDownloadCode,
                disabled: !codeInput,
              },
              '⬇ Download',
            ),
          ),
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement(
              'button',
              { id: 'detect-lang-btn', className: 'secondary-button', type: 'button', onClick: handleDetectLanguage },
              '🌐 Detect language',
            ),
            React.createElement(
              'button',
              { id: 'extract-deps-btn', className: 'secondary-button', type: 'button', onClick: handleExtractDependencies },
              '📦 Extract deps',
            ),
          ),
          StatusPill(analysisStatus),
          React.createElement(
            'div',
            { className: 'result-panel' },
            React.createElement(ErrorBoundary, null, renderAnalysisResult()),
          ),
        ),
      ),

      // ── Batch Processing ────────────────────────────────────────────────────
      React.createElement(
        'section',
        { className: 'card batch-card', 'aria-label': 'Batch processing' },
        React.createElement('h2', null, '📁 Batch Processing'),
        React.createElement(
          'div',
          { className: 'batch-layout' },
          React.createElement(
            'div',
            { className: 'batch-controls' },
            React.createElement(
              'div',
              { className: 'field-group' },
              React.createElement('label', { htmlFor: 'batch-files' }, 'Select multiple images'),
              React.createElement('input', {
                id: 'batch-files',
                type: 'file',
                multiple: true,
                accept: '.png,.jpg,.jpeg,.webp',
                onChange: (e) => setBatchFiles(e.target.files ? Array.from(e.target.files) : []),
              }),
            ),
            React.createElement(
              'div',
              { className: 'checkbox-row' },
              React.createElement(
                'label',
                { className: 'checkbox-label' },
                React.createElement('input', {
                  type: 'checkbox',
                  checked: batchPreprocess,
                  onChange: (e) => setBatchPreprocess(e.target.checked),
                }),
                'Preprocess',
              ),
              React.createElement(
                'label',
                { className: 'checkbox-label' },
                React.createElement('input', {
                  type: 'checkbox',
                  checked: batchOcr,
                  onChange: (e) => setBatchOcr(e.target.checked),
                }),
                'Run OCR',
              ),
              React.createElement(
                'label',
                { className: 'checkbox-label' },
                React.createElement('input', {
                  type: 'checkbox',
                  checked: batchDetectLang,
                  onChange: (e) => setBatchDetectLang(e.target.checked),
                }),
                'Detect language',
              ),
            ),
            React.createElement(
              'div',
              { className: 'button-row' },
              React.createElement(
                'button',
                {
                  id: 'batch-start-btn',
                  className: 'primary-button',
                  type: 'button',
                  onClick: handleBatchProcess,
                  disabled: !batchFiles.length,
                },
                '▶ Start batch',
              ),
              React.createElement(
                'button',
                {
                  id: 'export-csv-btn',
                  className: 'secondary-button',
                  type: 'button',
                  onClick: handleExportCsv,
                  disabled: !batchResults.length,
                },
                '💾 Export CSV',
              ),
            ),
            batchProgress.visible &&
              React.createElement(
                'div',
                { className: 'progress-wrap' },
                React.createElement(
                  'div',
                  { className: 'progress-bar' },
                  React.createElement('div', {
                    className: 'progress-fill',
                    style: { width: `${(batchProgress.current / Math.max(batchProgress.total, 1)) * 100}%` },
                  }),
                ),
                React.createElement(
                  'p',
                  { className: 'progress-text' },
                  `${batchProgress.current} / ${batchProgress.total} processed`,
                ),
              ),
            StatusPill(batchStatus),
          ),
          batchResults.length > 0 &&
            React.createElement(
              'div',
              { className: 'batch-results-wrap' },
              React.createElement(
                'table',
                { className: 'results-table' },
                React.createElement(
                  'thead',
                  null,
                  React.createElement(
                    'tr',
                    null,
                    React.createElement('th', null, 'File'),
                    React.createElement('th', null, 'Status'),
                    React.createElement('th', null, 'Language'),
                    React.createElement('th', null, 'Preview'),
                  ),
                ),
                React.createElement(
                  'tbody',
                  null,
                  batchResults.map((r, i) =>
                    React.createElement(
                      'tr',
                      { key: `${r.filename}-${i}` },
                      React.createElement('td', null, r.filename),
                      React.createElement(
                        'td',
                        null,
                        React.createElement('span', { className: cls('status-badge', r.status) }, r.status),
                      ),
                      React.createElement('td', null, r.language),
                      React.createElement('td', { className: 'preview-cell' }, r.preview || '—'),
                    ),
                  ),
                ),
              ),
            ),
        ),
      ),

      // ── Footer ─────────────────────────────────────────────────────────────
      React.createElement(
        'footer',
        { className: 'status-footer' },
        React.createElement(
          'div',
          { className: 'footer-item' },
          React.createElement(StatusDot, { ok: ocrReady }),
          ocrReady ? 'OCR pipeline ready' : 'OCR unavailable',
        ),
        React.createElement(
          'div',
          { className: 'footer-item' },
          React.createElement(StatusDot, { ok: apiKeyConfigured }),
          apiKeyConfigured ? 'API key configured' : 'API key missing',
        ),
        React.createElement(
          'div',
          { className: 'footer-item muted' },
          `CodeScan v${health?.version || '2.0.0'}`,
        ),
      ),

      // ── Health Modal ────────────────────────────────────────────────────────
      healthModalOpen && health &&
        React.createElement(
          'div',
          {
            className: 'modal-backdrop',
            onClick: () => setHealthModalOpen(false),
            role: 'dialog',
            'aria-modal': 'true',
            'aria-label': 'System health',
          },
          React.createElement(
            'div',
            { className: 'modal-dialog', onClick: (e) => e.stopPropagation() },
            React.createElement(
              'div',
              { className: 'modal-header' },
              React.createElement('h3', null, '⚙️ System Health'),
              React.createElement(
                'button',
                { className: 'icon-button small', type: 'button', onClick: () => setHealthModalOpen(false), 'aria-label': 'Close' },
                '✕',
              ),
            ),
            React.createElement(
              'div',
              { className: 'modal-body' },
              React.createElement(
                'table',
                { className: 'meta-table' },
                React.createElement(
                  'tbody',
                  null,
                  React.createElement(
                    'tr',
                    null,
                    React.createElement('th', null, 'Status'),
                    React.createElement('td', null, health.status),
                  ),
                  React.createElement(
                    'tr',
                    null,
                    React.createElement('th', null, 'Version'),
                    React.createElement('td', null, health.version),
                  ),
                  React.createElement(
                    'tr',
                    null,
                    React.createElement('th', null, 'API Key'),
                    React.createElement('td', null, health.api_key_configured ? '✅ Configured' : '❌ Not set'),
                  ),
                  ...Object.entries(health.services || {}).map(([key, val]) =>
                    React.createElement(
                      'tr',
                      { key },
                      React.createElement('th', null, key.replace(/_/g, ' ')),
                      React.createElement('td', null, val ? '✅ Available' : '❌ Unavailable'),
                    ),
                  ),
                ),
              ),
              React.createElement(
                'button',
                { className: 'primary-button', style: { marginTop: 16 }, type: 'button', onClick: () => { checkHealth(); setHealthModalOpen(false); } },
                '🔄 Refresh',
              ),
            ),
          ),
        ),
    ),
  );
}

document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('app');
  ReactDOM.createRoot(root).render(React.createElement(App));
});
