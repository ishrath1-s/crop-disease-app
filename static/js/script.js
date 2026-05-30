let selectedFile = null;
let currentMode = 'leaf';

const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewArea = document.getElementById('previewArea');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const uploadCard = document.getElementById('uploadCard');

function setMode(mode) {
    currentMode = mode;
    document.getElementById('leafBtn').classList.toggle('active', mode === 'leaf');
    document.getElementById('fruitBtn').classList.toggle('active', mode === 'fruit');
    document.getElementById('modeInfo').innerHTML = mode === 'leaf'
        ? '🍃 Upload a <strong>leaf image</strong> to detect crop diseases'
        : '🍎 Upload a <strong>fruit image</strong> to detect fruit diseases';
    document.getElementById('uploadIcon').textContent = mode === 'leaf' ? '🍃' : '🍎';
    document.getElementById('resultsSection').style.display = 'none';
    resetUpload();
}

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadCard.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadCard.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadCard.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    uploadArea.style.display = 'block';
    previewArea.style.display = 'none';
    analyzeBtn.disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
}

async function predict() {
    if (!selectedFile) return;
    document.getElementById('btnText').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const endpoint = currentMode === 'leaf' ? '/predict/leaf' : '/predict/fruit';
        const response = await fetch(endpoint, { method: 'POST', body: formData });
        const data = await response.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        displayResults(data.predictions);
    } catch (err) {
        alert('Something went wrong. Please try again.');
    } finally {
        document.getElementById('btnText').style.display = 'inline';
        document.getElementById('spinner').style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function getSeverityClass(severity) {
    const map = { 'None': 'sev-none', 'Moderate': 'sev-moderate', 'High': 'sev-high', 'Critical': 'sev-critical' };
    return map[severity] || 'sev-moderate';
}

function getSeverityEmoji(severity) {
    const map = { 'None': '✅', 'Moderate': '⚠️', 'High': '🔴', 'Critical': '🚨' };
    return map[severity] || '⚠️';
}

function displayResults(predictions) {
    const top = predictions[0];
    const parts = top.name.split(' → ');
    const main = parts[0] || top.name;
    const sub = parts[1] || '';

    document.getElementById('topResult').innerHTML = `
        <div>
            <div class="disease-name">${main}</div>
            <div class="crop-name">${sub ? '🔍 ' + sub : ''}</div>
            <span class="severity-badge ${getSeverityClass(top.severity)}">${getSeverityEmoji(top.severity)} ${top.severity} Severity</span>
        </div>
        <div class="confidence-ring">
            <div class="num">${top.confidence}%</div>
            <div class="label">Confidence</div>
        </div>
    `;

    document.getElementById('detailsGrid').innerHTML = `
        <div class="detail-card">
            <div class="icon">💊</div>
            <h4>Treatment</h4>
            <p>${top.treatment}</p>
        </div>
        <div class="detail-card">
            <div class="icon">🛡️</div>
            <h4>Prevention</h4>
            <p>${top.prevention}</p>
        </div>
    `;

    document.getElementById('otherPredictions').innerHTML = `
        <h3>Top 3 Predictions</h3>
        ${predictions.map(p => `
            <div class="pred-item">
                <span class="pred-name">${p.name}</span>
                <div class="pred-bar-wrap"><div class="pred-bar" style="width:${p.confidence}%"></div></div>
                <span class="pred-conf">${p.confidence}%</span>
            </div>
        `).join('')}
    `;

    const section = document.getElementById('resultsSection');
    section.style.display = 'block';
    section.scrollIntoView({ behavior: 'smooth' });
}