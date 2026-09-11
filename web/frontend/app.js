/**
 * Swim Analyzer — Web Frontend Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const uploadSection = document.getElementById('upload-section');
  const progressSection = document.getElementById('progress-section');
  const resultsSection = document.getElementById('results-section');

  const dropZone = document.getElementById('drop-zone');
  const videoInput = document.getElementById('video-file-input');
  const btnBrowse = document.getElementById('btn-browse-file');
  const filePreview = document.getElementById('file-info-preview');
  const fileNameLabel = document.getElementById('file-name-label');
  const fileSizeLabel = document.getElementById('file-size-label');
  const btnRemoveFile = document.getElementById('btn-remove-file');
  const dropContent = dropZone.querySelector('.drop-content');

  const analysisForm = document.getElementById('analysis-form');
  const btnStartAnalysis = document.getElementById('btn-start-analysis');

  // Progress Elements
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressPctLabel = document.getElementById('progress-pct-label');
  const progressStatusText = document.getElementById('progress-status-text');
  const btnCancelJob = document.getElementById('btn-cancel-job');

  // Results Elements
  const btnDownloadCsv = document.getElementById('btn-download-csv');
  const btnRestartAnalysis = document.getElementById('btn-restart-analysis');
  const btnHeaderNew = document.getElementById('btn-header-new');

  // Chart instances
  let velocityChartInstance = null;
  let strokeChartInstance = null;

  // Active Job State
  let currentJobId = null;
  let pollInterval = null;
  let selectedFile = null;

  // -------------------------------------------------------------------------
  // File Drop / Selection Handlers
  // -------------------------------------------------------------------------
  btnBrowse.addEventListener('click', () => videoInput.click());

  videoInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    clearSelectedFile();
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileNameLabel.textContent = file.name;
    fileSizeLabel.textContent = `${(file.size / (1024 * 1024)).toFixed(1)} MB`;

    dropContent.style.display = 'none';
    filePreview.style.display = 'flex';
    btnStartAnalysis.disabled = false;
  }

  function clearSelectedFile() {
    selectedFile = null;
    videoInput.value = '';
    dropContent.style.display = 'block';
    filePreview.style.display = 'none';
    btnStartAnalysis.disabled = true;
  }

  // -------------------------------------------------------------------------
  // Form Submission & Job Creation
  // -------------------------------------------------------------------------
  analysisForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData(analysisForm);
    formData.append('video', selectedFile);

    // Switch to progress view
    showSection(progressSection);
    resetProgressView();

    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Failed to start analysis');
      }

      const data = await response.json();
      currentJobId = data.job_id;
      startPolling(currentJobId);

    } catch (err) {
      alert(`Error starting analysis: ${err.message}`);
      showSection(uploadSection);
    }
  });

  // -------------------------------------------------------------------------
  // Polling Engine
  // -------------------------------------------------------------------------
  function startPolling(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error('Status poll failed');
        const data = await res.json();

        updateProgressUI(data);

        if (data.status === 'completed') {
          clearInterval(pollInterval);
          setTimeout(() => renderResults(data), 600);
        } else if (data.status === 'failed') {
          clearInterval(pollInterval);
          alert(`Analysis failed: ${data.error || 'Unknown error'}`);
          showSection(uploadSection);
        } else if (data.status === 'cancelled') {
          clearInterval(pollInterval);
          showSection(uploadSection);
        }
      } catch (err) {
        console.warn('Poll error:', err);
      }
    }, 1200);
  }

  function updateProgressUI(data) {
    const pct = data.progress.percent || 0;
    const msg = data.progress.message || 'Processing…';

    progressBarFill.style.width = `${pct}%`;
    progressPctLabel.textContent = `${pct}%`;
    progressStatusText.textContent = msg;

    // Stepper updates
    setStepActive(1, pct >= 5);
    setStepActive(2, pct >= 15);
    setStepActive(3, pct >= 60 || msg.includes('Gemini'));
    setStepActive(4, pct >= 80);
    setStepActive(5, pct >= 100);
  }

  function setStepActive(num, isActive) {
    const el = document.getElementById(`step-node-${num}`);
    if (el) {
      if (isActive) el.classList.add('active');
      else el.classList.remove('active');
    }
  }

  btnCancelJob.addEventListener('click', async () => {
    if (!currentJobId) return;
    if (confirm('Cancel this running analysis?')) {
      await fetch(`/api/jobs/${currentJobId}`, { method: 'DELETE' });
      clearInterval(pollInterval);
      showSection(uploadSection);
    }
  });

  // -------------------------------------------------------------------------
  // Results Rendering
  // -------------------------------------------------------------------------
  function renderResults(jobData) {
    const res = jobData.result;
    if (!res) return;

    showSection(resultsSection);
    btnHeaderNew.style.display = 'inline-block';

    // Header badge
    document.getElementById('result-stroke-badge').textContent = `${res.event_distance_m}m ${res.stroke.toUpperCase()}`;

    // KPI Cards
    const totalTimeFormatted = res.total_time_s ? `${res.total_time_s.toFixed(2)}s` : 'N/A';
    document.getElementById('kpi-total-time').textContent = totalTimeFormatted;
    document.getElementById('kpi-start-method').textContent = `Start: ${res.start_method}`;

    document.getElementById('kpi-avg-velocity').textContent = res.avg_velocity_mps ? `${res.avg_velocity_mps.toFixed(2)} m/s` : 'N/A';

    document.getElementById('kpi-start-15m').textContent = res.start_15m_time_s ? `${res.start_15m_time_s.toFixed(2)}s` : 'N/A';
    document.getElementById('kpi-start-reaction').textContent = res.start_reaction_time_s ? `Reaction: ${res.start_reaction_time_s.toFixed(2)}s` : 'No block sensor';

    document.getElementById('kpi-turn-time').textContent = res.turn_time_s ? `${res.turn_time_s.toFixed(2)}s` : 'None / Straight';

    document.getElementById('kpi-gemini-frames').textContent = `${res.gemini_assisted_frames} AI / ${res.cv_only_frames} CV`;
    document.getElementById('kpi-uncertain-frames').textContent = `${res.discrepancy_count} corrected frames`;

    // Render Splits Table
    const tbody = document.getElementById('splits-table-body');
    tbody.innerHTML = '';
    res.splits.forEach(s => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${s.marker_m} m</strong></td>
        <td>${s.time_s.toFixed(3)}</td>
        <td>${s.split_time_s ? s.split_time_s.toFixed(3) : '-'}</td>
        <td><span class="badge ${s.interpolated ? 'badge-warning' : 'badge-success'}">${s.interpolated ? 'Interpolated' : 'Measured'}</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Render Telemetry & Low Confidence Segments
    const totalFrames = (res.gemini_assisted_frames + res.cv_only_frames) || 1;
    document.getElementById('t-cv-val').textContent = `${((res.cv_only_frames / totalFrames) * 100).toFixed(1)}%`;
    document.getElementById('t-gemini-val').textContent = `${((res.gemini_assisted_frames / totalFrames) * 100).toFixed(1)}%`;
    document.getElementById('t-corrected-val').textContent = res.discrepancy_count;

    const segBody = document.getElementById('segments-table-body');
    segBody.innerHTML = '';
    if (res.low_confidence_segments && res.low_confidence_segments.length > 0) {
      res.low_confidence_segments.forEach(seg => {
        const tr = document.createElement('tr');
        const resSource = seg.resolved_by ? seg.resolved_by.toUpperCase() : 'UNRESOLVED';
        const badgeClass = seg.resolved_by === 'gemini' ? 'badge-warning' : (seg.resolved_by ? 'badge-success' : 'badge-danger');
        tr.innerHTML = `
          <td>${seg.start_frame}</td>
          <td>${seg.end_frame}</td>
          <td>${seg.start_time_s.toFixed(2)}s – ${seg.end_time_s.toFixed(2)}s</td>
          <td>${seg.reason.toUpperCase()}</td>
          <td><span class="badge ${badgeClass}">${resSource}</span></td>
        `;
        segBody.appendChild(tr);
      });
    } else {
      segBody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#95a5a6;">Clean track throughout race — no extended splash dropouts.</td></tr>';
    }

    // Render Charts
    renderCharts(res);

    // Render AI & Benchmark comparison
    renderAiAnalysis(res);
  }

  function renderCharts(res) {
    // 1. Velocity Profile Chart
    const velCanvas = document.getElementById('velocity-chart');
    if (velocityChartInstance) velocityChartInstance.destroy();

    const velData = res.velocity_profile.map(p => ({ x: p[0], y: p[1] }));

    velocityChartInstance = new Chart(velCanvas, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Tracked Velocity (m/s)',
          data: velData,
          borderColor: '#00d2d3',
          backgroundColor: 'rgba(0, 210, 211, 0.15)',
          fill: true,
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 2,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Distance (m)', color: '#95a5a6' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#95a5a6' }
          },
          y: {
            title: { display: true, text: 'Velocity (m/s)', color: '#95a5a6' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#95a5a6' }
          }
        },
        plugins: {
          legend: { labels: { color: '#fff' } }
        }
      }
    });

    // 2. Stroke Rate & Length Chart
    const strokeCanvas = document.getElementById('stroke-chart');
    if (strokeChartInstance) strokeChartInstance.destroy();

    const srData = res.stroke_rates.map(p => ({ x: p[0], y: p[1] }));
    const slData = res.stroke_lengths.map(p => ({ x: p[0], y: p[1] }));

    strokeChartInstance = new Chart(strokeCanvas, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Stroke Rate (cycles/min)',
            data: srData,
            borderColor: '#e74c3c',
            yAxisID: 'ySR',
            tension: 0.2,
            borderWidth: 2,
            pointRadius: 2,
          },
          {
            label: 'Stroke Length (m/cycle)',
            data: slData,
            borderColor: '#2ecc71',
            yAxisID: 'ySL',
            tension: 0.2,
            borderWidth: 2,
            pointRadius: 2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'linear',
            title: { display: true, text: 'Time (s)', color: '#95a5a6' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#95a5a6' }
          },
          ySR: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Rate (cpm)', color: '#e74c3c' },
            ticks: { color: '#e74c3c' },
            grid: { color: 'rgba(255,255,255,0.05)' },
          },
          ySL: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Distance (m)', color: '#2ecc71' },
            ticks: { color: '#2ecc71' },
            grid: { drawOnChartArea: false },
          }
        },
        plugins: {
          legend: { labels: { color: '#fff' } }
        }
      }
    });
  }

  function renderAiAnalysis(res) {
    const techBox = document.getElementById('ai-technique-content');
    const benchBox = document.getElementById('ai-benchmark-content');

    if (res.ai_result && res.ai_result.technique_analysis) {
      const t = res.ai_result.technique_analysis;
      let html = `<div style="margin-bottom:16px;">
        <h4>Overall Rating: <span style="color:#00d2d3">${t.overall_rating ? t.overall_rating.toUpperCase() : 'COMPLETE'}</span></h4>
        <p>Stroke: <strong>${t.stroke_identified || res.stroke}</strong></p>
      </div>`;

      if (t.top_priorities && t.top_priorities.length > 0) {
        html += '<h4>Key Biomechanical Priorities:</h4>';
        t.top_priorities.forEach(p => {
          html += `
            <div class="priority-item">
              <h4>#${p.priority} ${p.element}</h4>
              <p><strong>Impact:</strong> ${p.expected_impact}</p>
              <p><strong>Drill:</strong> <em>${p.drill}</em></p>
            </div>
          `;
        });
      }
      techBox.innerHTML = html;
    } else {
      techBox.innerHTML = '<p class="empty-notice">AI analysis completed. No critical technique faults detected.</p>';
    }

    if (res.benchmark_comparison) {
      benchBox.innerHTML = `
        <div class="priority-item" style="border-left-color:#f39c12;">
          <h4>Olympic & Elite Comparison</h4>
          <p style="white-space: pre-line;">${res.benchmark_comparison}</p>
        </div>
      `;
    } else {
      benchBox.innerHTML = '<p class="empty-notice">No verified benchmark match found for this distance and stroke.</p>';
    }
  }

  // -------------------------------------------------------------------------
  // Tab Switching
  // -------------------------------------------------------------------------
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // -------------------------------------------------------------------------
  // CSV Export & Navigation
  // -------------------------------------------------------------------------
  btnDownloadCsv.addEventListener('click', () => {
    if (!currentJobId) return;
    window.location.href = `/api/jobs/${currentJobId}/csv`;
  });

  btnRestartAnalysis.addEventListener('click', () => {
    showSection(uploadSection);
    clearSelectedFile();
    btnHeaderNew.style.display = 'none';
  });

  btnHeaderNew.addEventListener('click', () => {
    showSection(uploadSection);
    clearSelectedFile();
    btnHeaderNew.style.display = 'none';
  });

  function showSection(sectionEl) {
    [uploadSection, progressSection, resultsSection].forEach(s => s.classList.remove('active'));
    sectionEl.classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function resetProgressView() {
    progressBarFill.style.width = '0%';
    progressPctLabel.textContent = '0%';
    progressStatusText.textContent = 'Submitting video upload…';
    for (let i = 1; i <= 5; i++) setStepActive(i, false);
    setStepActive(1, true);
  }
});
