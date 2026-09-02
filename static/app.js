const { useEffect, useMemo, useState } = React;

function StatusPill({ message, kind }) {
  if (!message) return null;
  return React.createElement('div', { className: `status-pill ${kind || ''}` }, message);
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') === 'dark' ? 'dark' : 'light');
  const [sourceFile, setSourceFile] = useState(null);
  const [uploadMetadata, setUploadMetadata] = useState(null);
  const [uploadPreview, setUploadPreview] = useState('');
  const [uploadStatus, setUploadStatus] = useState({ message: '', kind: '' });

  const [contextText, setContextText] = useState('');
  const [recognitionResult, setRecognitionResult] = useState(null);
  const [recognitionStatus, setRecognitionStatus] = useState({ message: '', kind: '' });

  const [codeInput, setCodeInput] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState({ message: '', kind: '' });
  const [analysisResult, setAnalysisResult] = useState(null);

  const [batchFiles, setBatchFiles] = useState([]);
  const [batchStatus, setBatchStatus] = useState({ message: '', kind: '' });
  const [batchResults, setBatchResults] = useState([]);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, visible: false });
  const [health, setHealth] = useState(null);
  const [healthModalOpen, setHealthModalOpen] = useState(false);

  useEffect(() => {
    document.body.classList.toggle('dark-theme', theme === 'dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    checkHealth();
  }, []);

  const currentThemeLabel = theme === 'dark' ? '☀️' : '🌙';

  const metaSummary = useMemo(() => {
    if (!uploadMetadata) return ['No image processed yet.'];
    return [
      `File: ${uploadMetadata.filename}`,
      `Format: ${uploadMetadata.format}`,
      `Size: ${uploadMetadata.processed_size[0]} x ${uploadMetadata.processed_size[1]} px`,
    ];
  }, [uploadMetadata]);

  const handleUpload = async () => {
    if (!sourceFile) {
      setUploadStatus({ message: '❌ Please choose a file first.', kind: 'error' });
      return;
    }

    setUploadStatus({ message: '⏳ Processing image...', kind: 'loading' });
    const formData = new FormData();
    formData.append('file', sourceFile);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || 'Upload failed.');
      }

      setUploadMetadata(payload);
      setUploadPreview(payload.preview || '');
      setUploadStatus({ message: '✅ Image processed successfully.', kind: 'success' });
    } catch (error) {
      setUploadStatus({ message: `❌ ${error.message}`, kind: 'error' });
    }
  };

  const handleRecognize = async () => {
    if (!sourceFile) {
      setRecognitionStatus({ message: '❌ Please choose a file first.', kind: 'error' });
      return;
    }

    setRecognitionStatus({ message: '⏳ Recognizing handwriting...', kind: 'loading' });
    const formData = new FormData();
    formData.append('file', sourceFile);

    if (contextText.trim()) {
      formData.append('context', contextText.trim());
    }

    try {
      const response = await fetch('/api/recognize', {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail || 'Recognition failed.');
      }

      setRecognitionResult(payload);
      setRecognitionStatus({ message: '✅ Text recognized successfully.', kind: 'success' });

      if (payload.disambiguated_text) {
        setCodeInput(payload.disambiguated_text);
      }
    } catch (error) {
      setRecognitionStatus({ message: `❌ ${error.message}`, kind: 'error' });
    }
  };

  const detectLanguageFromText = async (text) => {
    const response = await fetch('/api/detect-language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || 'Language detection failed.');
    }

    return payload;
  };

  const handleDetectLanguage = async () => {
    const text = codeInput.trim();
    if (!text) {
      setAnalysisStatus({ message: '❌ Please enter code first.', kind: 'error' });
      return;
    }

    setAnalysisStatus({ message: '⏳ Detecting language...', kind: 'loading' });

    try {
      const payload = await detectLanguageFromText(text);
      setAnalysisResult({ type: 'language', payload });
      setAnalysisStatus({ message: '✅ Language detected.', kind: 'success' });
    } catch (error) {
      setAnalysisStatus({ message: `❌ ${error.message}`, kind: 'error' });
    }
  };

  const handleExtractDependencies = async () => {
    const text = codeInput.trim();
    if (!text) {
      setAnalysisStatus({ message: '❌ Please enter code first.', kind: 'error' });
      return;
    }

    setAnalysisStatus({ message: '⏳ Extracting dependencies...', kind: 'loading' });

    try {
      const detected = await detectLanguageFromText(text);
      const language = detected.language || 'python';

      const response = await fetch('/api/extract-dependencies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
      });

      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || 'Dependency extraction failed.');
      }

      setAnalysisResult({ type: 'dependencies', payload });
      setAnalysisStatus({ message: '✅ Dependencies extracted.', kind: 'success' });
    } catch (error) {
      setAnalysisStatus({ message: `❌ ${error.message}`, kind: 'error' });
    }
  };

  const handleBatchProcess = async () => {
    if (!batchFiles.length) {
      setBatchStatus({ message: '❌ Please choose files first.', kind: 'error' });
      return;
    }

    const preprocess = document.getElementById('batch-preprocess').checked;
    const ocr = document.getElementById('batch-ocr').checked;
    const detectLang = document.getElementById('batch-language').checked;

    setBatchStatus({ message: '⏳ Processing batch...', kind: 'loading' });
    setBatchProgress({ current: 0, total: batchFiles.length, visible: true });
    setBatchResults([]);

    const nextResults = [];

    for (let index = 0; index < batchFiles.length; index += 1) {
      const file = batchFiles[index];
      const result = {
        filename: file.name,
        status: 'processing',
        language: '-',
        preview: '',
        rawData: null,
      };

      try {
        if (preprocess || ocr) {
          const formData = new FormData();
          formData.append('file', file);
          const endpoint = ocr ? '/api/recognize' : '/api/upload';
          const response = await fetch(endpoint, { method: 'POST', body: formData });

          if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
          }

          const data = await response.json();
          result.rawData = data;

          if (endpoint.includes('/api/upload')) {
            result.preview = data.preview ? `${data.preview.slice(0, 48)}...` : 'Processed';
          } else {
            result.preview = data.disambiguated_text ? `${data.disambiguated_text.slice(0, 48)}...` : 'Recognized';
            if (data.language) result.language = data.language;
          }

          if (detectLang && data && data.language) {
            result.language = data.language;
          }

          result.status = 'success';
        } else {
          result.status = 'skipped';
        }
      } catch (error) {
        result.status = 'error';
        result.preview = error.message;
      }

      nextResults.push(result);
      setBatchProgress({
        current: index + 1,
        total: batchFiles.length,
        visible: true,
      });
      setBatchResults([...nextResults]);
    }

    setBatchResults(nextResults);
    setBatchStatus({ message: '✅ Batch processing complete.', kind: 'success' });
    setBatchProgress({ current: batchFiles.length, total: batchFiles.length, visible: true });
  };

  const handleExportCsv = () => {
    const headers = ['Filename', 'Status', 'Language', 'Preview'];
    const rows = batchResults.map((result) => [
      result.filename,
      result.status,
      result.language,
      (result.preview || '').replace(/"/g, '""'),
    ]);

    const csv = [
      headers.map((header) => `"${header}"`).join(','),
      ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'batch-results.csv';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const checkHealth = async () => {
    try {
      const response = await fetch('/api/health');
      const data = await response.json();
      setHealth(data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth({
        status: 'error',
        version: '2.0.0',
        services: {
          image_preprocessing: false,
          language_detection: false,
          dependency_extraction: false,
          ocr_pipeline: false,
        },
      });
    }
  };

  const renderAnalysisCard = () => {
    if (!analysisResult) {
      return React.createElement('p', { className: 'placeholder' }, 'Analysis results will appear here...');
    }

    if (analysisResult.type === 'language') {
      const { payload } = analysisResult;
      return React.createElement(
        'div',
        { className: 'result-stack' },
        React.createElement('p', null, React.createElement('strong', null, '🌐 Detected language: '), payload.language.toUpperCase()),
        React.createElement('p', null, React.createElement('strong', null, '🎯 Confidence: '), `${(payload.confidence * 100).toFixed(1)}%`),
        payload.patterns_matched && payload.patterns_matched.length > 0
          ? React.createElement('p', null, React.createElement('strong', null, '📋 Patterns matched: '), payload.patterns_matched.join(', '))
          : null,
        payload.scores
          ? React.createElement('pre', { className: 'code-block compact' }, JSON.stringify(payload.scores, null, 2))
          : null,
      );
    }

    const { payload } = analysisResult;
    return React.createElement(
      'div',
      { className: 'result-stack' },
      React.createElement('p', null, React.createElement('strong', null, '📦 Language: '), payload.language),
      payload.dependencies && payload.dependencies.length > 0
        ? React.createElement(
            'ul',
            { className: 'simple-list' },
            ...payload.dependencies.map((dep) =>
              React.createElement(
                'li',
                { key: dep.import_statement || dep.name },
                React.createElement('code', null, dep.import_statement || dep.name),
                React.createElement('span', { className: 'muted' }, ` Confidence ${(dep.confidence * 100).toFixed(0)}%`),
              ),
            ),
          )
        : React.createElement('p', { className: 'muted' }, 'No dependencies detected.'),
    );
  };

  const renderRecognitionCard = () => {
    if (!recognitionResult) {
      return React.createElement('p', { className: 'placeholder' }, 'Recognized text will appear here...');
    }

    return React.createElement(
      'div',
      { className: 'result-stack' },
      recognitionResult.raw_text
        ? React.createElement(
            'div',
            null,
            React.createElement('p', null, React.createElement('strong', null, '🔍 Raw text')),
            React.createElement('pre', { className: 'code-block' }, recognitionResult.raw_text),
          )
        : null,
      recognitionResult.disambiguated_text
        ? React.createElement(
            'div',
            null,
            React.createElement('p', null, React.createElement('strong', null, '✨ Disambiguated text')),
            React.createElement('pre', { className: 'code-block' }, recognitionResult.disambiguated_text),
          )
        : null,
      recognitionResult.language
        ? React.createElement(
            'p',
            null,
            React.createElement('strong', null, '🌐 Language: '),
            `${recognitionResult.language} (${(recognitionResult.confidence * 100).toFixed(1)}% confidence)`,
          )
        : null,
      recognitionResult.corrections && recognitionResult.corrections.length > 0
        ? React.createElement('p', null, React.createElement('strong', null, '🔧 Corrections applied: '), recognitionResult.corrections.length)
        : null,
      recognitionResult.dependencies && recognitionResult.dependencies.length > 0
        ? React.createElement(
            'div',
            null,
            React.createElement('p', null, React.createElement('strong', null, '📦 Dependencies found: '), recognitionResult.dependencies.length),
            React.createElement(
              'ul',
              { className: 'simple-list compact' },
              ...recognitionResult.dependencies.map((dep) => React.createElement('li', { key: dep }, React.createElement('code', null, dep))),
            ),
          )
        : null,
    );
  };

  return React.createElement(
    'div',
    { className: `app-shell ${theme === 'dark' ? 'dark-theme' : 'light-theme'}` },
    React.createElement(
      'div',
      { className: 'shell' },
      React.createElement(
        'header',
        { className: 'topbar' },
        React.createElement(
          'div',
          null,
          React.createElement('h1', null, 'CodeScan'),
          React.createElement('p', { className: 'subtitle' }, 'Handwritten code recognition and analysis pipeline'),
        ),
        React.createElement(
          'div',
          { className: 'header-actions' },
          React.createElement(
            'button',
            { className: 'icon-button', type: 'button', onClick: () => setHealthModalOpen(true) },
            '⚙️',
          ),
          React.createElement(
            'button',
            { className: 'icon-button', type: 'button', onClick: () => setTheme(theme === 'dark' ? 'light' : 'dark') },
            currentThemeLabel,
          ),
        ),
      ),
      React.createElement(
        'main',
        { className: 'workspace-grid' },
        React.createElement(
          'section',
          { className: 'card' },
          React.createElement('h2', null, '📁 Input, preprocessing & recognition'),
          React.createElement(
            'label',
            { className: 'dropzone', htmlFor: 'source-input' },
            React.createElement('input', {
              id: 'source-input',
              type: 'file',
              accept: '.png,.jpg,.jpeg,.webp,.heic,.pdf',
              onChange: (event) => setSourceFile(event.target.files && event.target.files[0] ? event.target.files[0] : null),
            }),
            React.createElement(
              'div',
              { className: 'dropzone-content' },
              React.createElement('span', { className: 'dropzone-icon' }, '📸'),
              React.createElement('span', { className: 'dropzone-text' }, sourceFile ? sourceFile.name : 'Choose image or drag it here'),
              React.createElement('span', { className: 'dropzone-hint' }, '.png, .jpg, .webp, .heic, .pdf'),
            ),
          ),
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement('button', { className: 'primary-button', type: 'button', onClick: handleUpload }, '⚙️ Preprocess image'),
            React.createElement('button', { className: 'secondary-button', type: 'button', onClick: handleRecognize }, '🤖 Recognize text'),
          ),
          StatusPill(uploadStatus),
          StatusPill(recognitionStatus),
          React.createElement(
            'div',
            { className: 'field-group' },
            React.createElement('label', { htmlFor: 'context-input' }, 'Optional context'),
            React.createElement('input', {
              id: 'context-input',
              type: 'text',
              className: 'text-input',
              placeholder: "e.g. 'Python code from a whiteboard'",
              value: contextText,
              onChange: (event) => setContextText(event.target.value),
            }),
          ),
          React.createElement(
            'div',
            { className: 'image-stage' },
            uploadPreview
              ? React.createElement('img', { src: uploadPreview, alt: 'Processed preview', className: 'preview-image' })
              : React.createElement('div', { className: 'empty-state' }, 'Processed preview appears here'),
          ),
          React.createElement(
            'div',
            { className: 'meta-box' },
            React.createElement('h3', null, 'Output details'),
            React.createElement(
              'ul',
              { className: 'meta-list' },
              ...metaSummary.map((item) => React.createElement('li', { key: item }, item)),
            ),
          ),
          React.createElement('div', { className: 'result-panel' }, renderRecognitionCard()),
        ),
        React.createElement(
          'section',
          { className: 'card' },
          React.createElement('h2', null, '🔍 Code analysis'),
          React.createElement(
            'div',
            { className: 'field-group' },
            React.createElement('label', { htmlFor: 'code-input' }, 'Code snippet'),
            React.createElement('textarea', {
              id: 'code-input',
              className: 'code-textarea',
              value: codeInput,
              onChange: (event) => setCodeInput(event.target.value),
              placeholder: 'Paste or auto-fill recognized code here...',
            }),
          ),
          React.createElement(
            'div',
            { className: 'button-row' },
            React.createElement('button', { className: 'secondary-button', type: 'button', onClick: handleDetectLanguage }, '🌐 Detect language'),
            React.createElement('button', { className: 'secondary-button', type: 'button', onClick: handleExtractDependencies }, '📦 Extract dependencies'),
          ),
          StatusPill(analysisStatus),
          React.createElement('div', { className: 'result-panel' }, renderAnalysisCard()),
        ),
      ),
      React.createElement(
        'section',
        { className: 'card batch-card' },
        React.createElement('h2', null, '📁 Batch processing'),
        React.createElement(
          'div',
          { className: 'field-group' },
          React.createElement('label', { htmlFor: 'batch-files' }, 'Select multiple images'),
          React.createElement('input', {
            id: 'batch-files',
            type: 'file',
            multiple: true,
            accept: '.png,.jpg,.jpeg,.webp,.heic,.pdf',
            onChange: (event) => setBatchFiles(event.target.files ? Array.from(event.target.files) : []),
          }),
        ),
        React.createElement(
          'div',
          { className: 'checkbox-row' },
          React.createElement('label', { className: 'checkbox-label' },
            React.createElement('input', { id: 'batch-preprocess', type: 'checkbox', defaultChecked: true }),
            'Run preprocessing'),
          React.createElement('label', { className: 'checkbox-label' },
            React.createElement('input', { id: 'batch-ocr', type: 'checkbox', defaultChecked: true }),
            'Run OCR'),
          React.createElement('label', { className: 'checkbox-label' },
            React.createElement('input', { id: 'batch-language', type: 'checkbox' }),
            'Detect language'),
        ),
        React.createElement(
          'div',
          { className: 'button-row' },
          React.createElement('button', { className: 'primary-button', type: 'button', onClick: handleBatchProcess }, '▶️ Start batch'),
          React.createElement(
            'button',
            { className: 'secondary-button', type: 'button', onClick: handleExportCsv, disabled: batchResults.length === 0 },
            '💾 Export CSV',
          ),
        ),
        batchProgress.visible
          ? React.createElement(
              'div',
              { className: 'progress-wrap' },
              React.createElement('div', { className: 'progress-bar' },
                React.createElement('div', {
                  className: 'progress-fill',
                  style: { width: `${(batchProgress.current / Math.max(batchProgress.total, 1)) * 100}%` },
                }),
              ),
              React.createElement('p', { className: 'progress-text' }, `${batchProgress.current} / ${batchProgress.total} processed`),
            )
          : null,
        StatusPill(batchStatus),
        batchResults.length > 0
          ? React.createElement(
              'div',
              { className: 'batch-results-table-wrap' },
              React.createElement('table', { className: 'results-table' },
                React.createElement('thead', null,
                  React.createElement('tr', null,
                    React.createElement('th', null, 'File'),
                    React.createElement('th', null, 'Status'),
                    React.createElement('th', null, 'Language'),
                    React.createElement('th', null, 'Preview'),
                  ),
                ),
                React.createElement('tbody', null,
                  ...batchResults.map((result, index) =>
                    React.createElement('tr', { key: `${result.filename}-${index}` },
                      React.createElement('td', null, result.filename),
                      React.createElement('td', null, React.createElement('span', { className: `status-badge ${result.status}` }, result.status)),
                      React.createElement('td', null, result.language),
                      React.createElement('td', null, result.preview || '—'),
                    ),
                  ),
                ),
              ),
            )
          : null,
      ),
      React.createElement(
        'footer',
        { className: 'status-footer' },
        React.createElement('div', { className: 'footer-item' }, health && health.services && health.services.ocr_pipeline ? '✅ OCR pipeline ready' : '⚠️ OCR pipeline unavailable'),
        React.createElement('div', { className: 'footer-item' }, `Version: ${health ? health.version || '2.0.0' : '2.0.0'}`),
      ),
      healthModalOpen && health
        ? React.createElement(
            'div',
            { className: 'modal-backdrop', onClick: () => setHealthModalOpen(false) },
            React.createElement(
              'div',
              { className: 'modal-dialog', onClick: (event) => event.stopPropagation() },
              React.createElement('div', { className: 'modal-header' },
                React.createElement('h3', null, 'System health'),
                React.createElement('button', { className: 'icon-button small', type: 'button', onClick: () => setHealthModalOpen(false) }, '✕'),
              ),
              React.createElement(
                'div',
                { className: 'modal-body' },
                React.createElement('p', null, React.createElement('strong', null, 'Status: '), health.status),
                React.createElement('p', null, React.createElement('strong', null, 'Version: '), health.version),
                React.createElement(
                  'ul',
                  { className: 'simple-list' },
                  ...Object.entries(health.services || {}).map(([key, value]) =>
                    React.createElement('li', { key: key }, `${key.replace(/_/g, ' ')}: ${value ? 'available' : 'unavailable'}`),
                  ),
                ),
              ),
            ),
          )
        : null,
    ),
  );
}

document.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('app');
  ReactDOM.createRoot(root).render(React.createElement(App));
});
