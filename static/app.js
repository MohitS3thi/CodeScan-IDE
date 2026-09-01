// ========== STATE & DOM REFERENCES ==========
let batchResults = [];
let currentFile = null;

// Phase 1 Elements
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');
const previewImage = document.getElementById('preview-image');
const phase1Metadata = document.getElementById('phase1-metadata');
const phase1Status = document.getElementById('phase1-status');
const dropzone = document.getElementById('dropzone');

// Phase 2 Elements
const ocrFileInput = document.getElementById('ocr-file-input');
const contextInput = document.getElementById('context-input');
const recognizeButton = document.getElementById('recognize-button');
const phase2Status = document.getElementById('phase2-status');
const recognitionResults = document.getElementById('recognition-results');
const ocrDropzone = document.getElementById('ocr-dropzone');

// Code Analysis Elements
const codeInput = document.getElementById('code-input');
const detectLangBtn = document.getElementById('detect-lang-btn');
const extractDepsBtn = document.getElementById('extract-deps-btn');
const analysisStatus = document.getElementById('analysis-status');
const analysisResults = document.getElementById('analysis-results');

// Batch Processing Elements
const batchFilesInput = document.getElementById('batch-files-input');
const batchProcessBtn = document.getElementById('batch-process-btn');
const batchExportBtn = document.getElementById('batch-export-btn');
const batchStatus = document.getElementById('batch-status');
const batchProgress = document.getElementById('batch-progress');
const batchProgressFill = document.getElementById('batch-progress-fill');
const batchProgressText = document.getElementById('batch-progress-text');
const batchResultsDiv = document.getElementById('batch-results');
const batchResultsBody = document.getElementById('batch-results-body');

// Modal Elements
const healthCheckBtn = document.getElementById('health-check-btn');
const healthModal = document.getElementById('health-modal');
const healthBody = document.getElementById('health-body');
const themeToggle = document.getElementById('theme-toggle');

// ========== UTILITY FUNCTIONS ==========

const setStatus = (elementId, message, type = '') => {
  const element = document.getElementById(elementId);
  if (element) {
    element.textContent = message;
    element.className = `status ${type}`.trim();
  }
};

const showLoading = (elementId) => {
  setStatus(elementId, '⏳ Processing...', 'loading');
};

const copyToClipboard = (text) => {
  navigator.clipboard.writeText(text).then(() => {
    console.log('Copied to clipboard');
  }).catch(err => {
    console.error('Failed to copy:', err);
  });
};

const formatJSON = (obj) => {
  return JSON.stringify(obj, null, 2);
};

const escapeHtml = (text) => {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

// ========== TAB NAVIGATION ==========

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    const tabName = e.target.dataset.tab;
    
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
      tab.classList.remove('active');
    });
    
    // Deactivate all buttons
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    e.target.classList.add('active');
  });
});

// ========== THEME TOGGLE ==========

function toggleTheme() {
  document.body.classList.toggle('dark-theme');
  themeToggle.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
  localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
}

// Load saved theme
if (localStorage.getItem('theme') === 'dark') {
  document.body.classList.add('dark-theme');
  themeToggle.textContent = '☀️';
}

// ========== PHASE 1: IMAGE PREPROCESSING ==========

// Drag and drop
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
  }
});

uploadButton.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus('phase1-status', '❌ Please choose a file first.', 'error');
    return;
  }

  showLoading('phase1-status');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || 'Upload failed.');
    }

    currentFile = payload;
    previewImage.src = payload.preview;
    previewImage.style.display = 'block';

    phase1Metadata.innerHTML = `
      <p><strong>📄 File:</strong> ${escapeHtml(payload.filename)}</p>
      <p><strong>📸 Format:</strong> ${payload.format}</p>
      <p><strong>📐 Original:</strong> ${payload.processed_size[0]} x ${payload.processed_size[1]}px</p>
    `;

    setStatus('phase1-status', '✅ Image processed successfully.', 'success');
  } catch (error) {
    setStatus('phase1-status', `❌ ${error.message}`, 'error');
  }
});

// ========== PHASE 2: OCR RECOGNITION ==========

// Drag and drop for OCR
ocrDropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  ocrDropzone.classList.add('dragover');
});

ocrDropzone.addEventListener('dragleave', () => {
  ocrDropzone.classList.remove('dragover');
});

ocrDropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  ocrDropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    ocrFileInput.files = e.dataTransfer.files;
  }
});

recognizeButton.addEventListener('click', async () => {
  const file = ocrFileInput.files[0];
  if (!file) {
    setStatus('phase2-status', '❌ Please choose a file first.', 'error');
    return;
  }

  showLoading('phase2-status');

  const formData = new FormData();
  formData.append('file', file);
  
  const context = contextInput.value.trim();
  if (context) {
    formData.append('context', context);
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

    displayRecognitionResult(payload);
    setStatus('phase2-status', '✅ Text recognized successfully.', 'success');
    
    // Auto-fill code analysis
    if (payload.disambiguated_text) {
      codeInput.value = payload.disambiguated_text;
    }
  } catch (error) {
    setStatus('phase2-status', `❌ ${error.message}`, 'error');
  }
});

function displayRecognitionResult(result) {
  let html = '<div class="result-item">';
  
  if (result.raw_text) {
    html += `<p><strong>🔍 Raw Text:</strong></p><div class="result-text">${escapeHtml(result.raw_text)}</div>`;
  }
  
  if (result.disambiguated_text) {
    html += `<p><strong>✨ Disambiguated Text:</strong></p><div class="result-text">${escapeHtml(result.disambiguated_text)}</div>`;
  }
  
  if (result.language) {
    html += `<p><strong>🌐 Language:</strong> ${result.language} (${(result.confidence * 100).toFixed(1)}% confidence)</p>`;
  }
  
  if (result.corrections && result.corrections.length > 0) {
    html += `<p><strong>🔧 Corrections Applied:</strong> ${result.corrections.length}</p>`;
  }
  
  if (result.dependencies && result.dependencies.length > 0) {
    html += `<p><strong>📦 Dependencies Found:</strong> ${result.dependencies.length}</p>`;
    html += '<ul>';
    result.dependencies.forEach(dep => {
      html += `<li><code>${escapeHtml(dep)}</code></li>`;
    });
    html += '</ul>';
  }
  
  html += '</div>';
  recognitionResults.innerHTML = html;
  recognitionResults.classList.add('populated');
}

// ========== CODE ANALYSIS ==========

detectLangBtn.addEventListener('click', async () => {
  const text = codeInput.value.trim();
  if (!text) {
    setStatus('analysis-status', '❌ Please enter code.', 'error');
    return;
  }

  showLoading('analysis-status');

  try {
    const response = await fetch('/api/detect-language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || 'Language detection failed.');
    }

    displayLanguageResult(payload);
    setStatus('analysis-status', '✅ Language detected.', 'success');
  } catch (error) {
    setStatus('analysis-status', `❌ ${error.message}`, 'error');
  }
});

extractDepsBtn.addEventListener('click', async () => {
  const text = codeInput.value.trim();
  if (!text) {
    setStatus('analysis-status', '❌ Please enter code.', 'error');
    return;
  }

  showLoading('analysis-status');

  try {
    const detectedLang = await getDetectedLanguage(text);
    const language = detectedLang || 'python';

    const response = await fetch('/api/extract-dependencies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || 'Dependency extraction failed.');
    }

    displayDependencyResult(payload);
    setStatus('analysis-status', '✅ Dependencies extracted.', 'success');
  } catch (error) {
    setStatus('analysis-status', `❌ ${error.message}`, 'error');
  }
});

async function getDetectedLanguage(text) {
  try {
    const response = await fetch('/api/detect-language', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    return data.language;
  } catch (error) {
    console.error('Error detecting language:', error);
    return 'python';
  }
}

function displayLanguageResult(payload) {
  let html = '<div class="result-item">';
  html += `<p><strong>🌐 Detected Language:</strong> <span style="color: var(--accent)">${payload.language.toUpperCase()}</span></p>`;
  html += `<p><strong>🎯 Confidence:</strong> ${(payload.confidence * 100).toFixed(1)}%</p>`;
  
  if (payload.patterns_matched && payload.patterns_matched.length > 0) {
    html += `<p><strong>📋 Patterns Matched:</strong> ${payload.patterns_matched.join(', ')}</p>`;
  }
  
  if (payload.scores) {
    html += '<p><strong>📊 Scores:</strong></p><pre style="font-size: 0.8rem;">' + formatJSON(payload.scores) + '</pre>';
  }
  
  html += '</div>';
  analysisResults.innerHTML = html;
  analysisResults.classList.add('populated');
}

function displayDependencyResult(payload) {
  let html = '<div class="result-item">';
  html += `<p><strong>📦 Language:</strong> ${payload.language}</p>`;
  
  if (payload.dependencies && payload.dependencies.length > 0) {
    html += `<p><strong>🔗 Found ${payload.dependencies.length} Dependencies:</strong></p>`;
    html += '<ul style="margin: 8px 0;">';
    
    payload.dependencies.forEach(dep => {
      html += `<li>
        <code style="background: rgba(45, 212, 191, 0.15); padding: 2px 6px; border-radius: 3px;">${escapeHtml(dep.import_statement)}</code>
        <br /><small style="color: var(--muted);">Confidence: ${(dep.confidence * 100).toFixed(0)}%</small>
      </li>`;
    });
    
    html += '</ul>';
  } else {
    html += '<p style="color: var(--muted);">No dependencies detected.</p>';
  }
  
  html += '</div>';
  analysisResults.innerHTML = html;
  analysisResults.classList.add('populated');
}

// ========== BATCH PROCESSING ==========

batchProcessBtn.addEventListener('click', async () => {
  const files = batchFilesInput.files;
  if (files.length === 0) {
    setStatus('batch-status', '❌ Please select files.', 'error');
    return;
  }

  const preprocess = document.getElementById('batch-preprocess').checked;
  const ocr = document.getElementById('batch-ocr').checked;
  const detectLang = document.getElementById('batch-language').checked;

  batchResults = [];
  batchResultsDiv.style.display = 'none';
  batchExportBtn.disabled = true;
  batchProgress.style.display = 'block';
  showLoading('batch-status');

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const progressPercent = ((i + 1) / files.length) * 100;
    batchProgressFill.style.width = `${progressPercent}%`;
    batchProgressText.textContent = `${i + 1} / ${files.length} processed`;

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

        const apiEndpoint = ocr ? '/api/recognize' : '/api/upload';
        const response = await fetch(apiEndpoint, {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        result.rawData = data;

        if (response.url.includes('/api/upload')) {
          result.preview = data.preview ? data.preview.substring(0, 50) + '...' : 'Processed';
        } else {
          result.preview = data.disambiguated_text ? data.disambiguated_text.substring(0, 50) + '...' : 'Recognized';
          if (data.language) result.language = data.language;
        }

        result.status = 'success';
      } else {
        result.status = 'skipped';
      }
    } catch (error) {
      result.status = 'error';
      result.preview = error.message;
    }

    batchResults.push(result);
  }

  displayBatchResults();
  batchExportBtn.disabled = false;
  setStatus('batch-status', '✅ Batch processing complete.', 'success');
});

function displayBatchResults() {
  batchResultsBody.innerHTML = '';
  
  batchResults.forEach((result, index) => {
    const row = document.createElement('tr');
    
    const statusBadge = {
      success: '<span class="status-badge success">✅ Success</span>',
      error: '<span class="status-badge error">❌ Error</span>',
      processing: '<span class="status-badge processing">⏳ Processing</span>',
      skipped: '<span class="status-badge">⊘ Skipped</span>',
    }[result.status] || '<span class="status-badge">?</span>';
    
    row.innerHTML = `
      <td>${escapeHtml(result.filename)}</td>
      <td>${statusBadge}</td>
      <td>${result.language}</td>
      <td><code style="font-size: 0.75rem;">${escapeHtml(result.preview)}</code></td>
      <td><button class="icon-btn" onclick="viewBatchResult(${index})" style="width: 32px; height: 32px; font-size: 0.9rem;">👁</button></td>
    `;
    
    batchResultsBody.appendChild(row);
  });
  
  batchResultsDiv.style.display = 'block';
}

function viewBatchResult(index) {
  const result = batchResults[index];
  const content = JSON.stringify(result.rawData, null, 2);
  alert(`Batch Result: ${result.filename}\n\n${content}`);
}

batchExportBtn.addEventListener('click', () => {
  const csv = generateCSV(batchResults);
  downloadCSV(csv, 'batch-results.csv');
});

function generateCSV(results) {
  const headers = ['Filename', 'Status', 'Language', 'Preview'];
  const rows = results.map(r => [
    r.filename,
    r.status,
    r.language,
    r.preview.replace(/"/g, '""'),
  ]);
  
  const csv = [
    headers.map(h => `"${h}"`).join(','),
    ...rows.map(r => r.map(cell => `"${cell}"`).join(',')),
  ].join('\n');
  
  return csv;
}

function downloadCSV(csv, filename) {
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
}

// ========== HEALTH CHECK & SYSTEM STATUS ==========

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    displayHealthModal(data);
    updateFooterStatus(data);
  } catch (error) {
    console.error('Health check failed:', error);
  }
}

function displayHealthModal(data) {
  let html = '<div>';
  
  html += `<p><strong>Version:</strong> ${data.version}</p>`;
  html += `<p><strong>Status:</strong> ${data.status}</p>`;
  
  if (data.services) {
    html += '<h4>Services:</h4>';
    Object.entries(data.services).forEach(([name, status]) => {
      const statusClass = status ? 'ok' : 'unavailable';
      html += `
        <div class="health-item">
          <div class="health-status ${statusClass}"></div>
          <span>${name.replace(/_/g, ' ')}:</span>
          <strong>${status ? '✅ Available' : '❌ Unavailable'}</strong>
        </div>
      `;
    });
  }
  
  html += '</div>';
  healthBody.innerHTML = html;
  healthModal.classList.add('active');
}

function closeHealthModal() {
  healthModal.classList.remove('active');
}

function updateFooterStatus(data) {
  const ocrItem = document.getElementById('footer-ocr');
  const statusDot = ocrItem.querySelector('.status-dot');
  const isOcrReady = data.services && data.services.ocr_pipeline;
  
  if (isOcrReady) {
    statusDot.classList.add('ready');
    ocrItem.innerHTML = '<span class="status-dot ready"></span> OCR Pipeline: Ready';
  } else {
    statusDot.classList.remove('ready');
    ocrItem.innerHTML = '<span class="status-dot"></span> OCR Pipeline: Unavailable';
  }
}

// Initialize health check on load
window.addEventListener('load', () => {
  checkHealth();
});

// Close modal on outside click
healthModal.addEventListener('click', (e) => {
  if (e.target === healthModal) {
    closeHealthModal();
  }
});
