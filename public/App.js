/* ============================================================
   KIU CrowdSense Dashboard — App.js
   People Detection & Counting System using YOLOv8 + ByteTrack
   FYP — BSc Computer Science, Kampala International University
   ============================================================ */

'use strict';

// ── Constants ────────────────────────────────────────────────
const PYTHON_API = 'http://localhost:8000';
const REFRESH_INTERVAL = 5000; // ms
const LIVE_CAMERA_STATS_INTERVAL = 1000; // ms (IN/OUT/NOW/FPS on camera cards)
const LIVE_DETECTION_REFRESH_INTERVAL = 5000; // ms (metrics + events)

// KIU campus camera zones
const CAMERA_COORDS = {
  'kiu-webcam':        { lat: 0.2858, lng: 32.5734, label: 'Live Webcam (USB)' },
  'kiu-main-entrance': { lat: 0.2858, lng: 32.5734, label: 'Main Entrance (Demo)' },
  'kiu-library':       { lat: 0.2861, lng: 32.5738, label: 'Library Block (Demo)' },
};

function cameraLabel(id) {
  return (CAMERA_COORDS[id] || {}).label || id;
}

// ── Chart.js defaults ────────────────────────────────────────
function getThemeColor(cssVarName, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(cssVarName).trim();
  return value || fallback;
}

Chart.defaults.color = getThemeColor('--muted', '#64748b');
Chart.defaults.borderColor = getThemeColor('--border', '#e2e8f0');
Chart.defaults.font.family = 'Inter, sans-serif';
Chart.defaults.plugins.legend.display = false;

function lineDataset(label, color) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color + '22',
    fill: true,
    tension: 0.35,
    pointRadius: 3,
    pointHoverRadius: 5,
  };
}

function makeChart(id, datasets, yLabel) {
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: { title: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { beginAtZero: true, title: { display: !!yLabel, text: yLabel || '' } },
      },
      plugins: { legend: { display: datasets.length > 1 } },
    },
  });
}

function pushToChart(chart, label, values) {
  if (!chart) return;
  chart.data.labels.push(label);
  values.forEach((v, i) => chart.data.datasets[i].data.push(v));
  if (chart.data.labels.length > 20) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(d => d.data.shift());
  }
  chart.update('none');
}

// ── Navigation ───────────────────────────────────────────────
const SECTION_TITLES = {
  'overview':       'Overview',
  'live-detection': 'Live Detection',
  'live-map':       'Live Map',
  'shops':          'Cameras',
  'analytics':      'Detection Analytics',
  'top-shops':      'Zone Rankings',
  'heatmap':        'Occupancy',
  'history':        'Sessions',
  'health':         'System Health',
};

let currentSection = 'overview';

function navigate(section) {
  document.querySelectorAll('.section').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));
  const sec = document.getElementById('sec-' + section);
  if (sec) sec.classList.add('active');
  const nav = document.querySelector(`.nav-item[data-section="${section}"]`);
  if (nav) nav.classList.add('active');
  document.getElementById('topbarTitle').textContent = SECTION_TITLES[section] || section;
  currentSection = section;
  closeSidebar();
  // Lazy load section data
  if (section === 'shops')          loadShops();
  if (section === 'top-shops')      loadTopShops();
  if (section === 'heatmap')        loadHeatmap();
  if (section === 'health')         loadHealth();
  if (section === 'live-detection') loadDetectionData();
  if (section === 'history')        loadHistory();
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
  document.getElementById('sidebarOverlay').classList.toggle('open');
}
function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('open');
}
document.getElementById('sidebarOverlay').addEventListener('click', closeSidebar);

// ── Fetch helpers ────────────────────────────────────────────
async function apiFetch(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

// ── Overview + Map charts (shared) ──────────────────────────
let ovPeakChart, ovAvgChart, ovForecastChart;
let anPeakChart, anAvgChart, anForecastChart;
let recentData = [];
let leafletMap, mapMarkers = {}, mapCircles = {};

function initCharts() {
  ovPeakChart     = makeChart('ovPeakChart',     [lineDataset('Total IN',    '#4f8ef7')]);
  ovAvgChart      = makeChart('ovAvgChart',      [lineDataset('Total OUT',   '#22c55e')]);
  ovForecastChart = makeChart('ovForecastChart', [lineDataset('Net Occupancy', '#f59e0b')]);
  anPeakChart     = makeChart('anPeakChart',     [lineDataset('Total IN',    '#4f8ef7')]);
  anAvgChart      = makeChart('anAvgChart',      [lineDataset('Net Occupancy', '#22c55e')]);
  anForecastChart = makeChart('anForecastChart', [lineDataset('Current Count', '#f59e0b')]);
}

function initMap() {
  // Centre on KIU campus, Kampala
  leafletMap = L.map('map').setView([0.2858, 32.5734], 17);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
  }).addTo(leafletMap);
}

function crowdColor(count) {
  if (count >= 16) return '#ef4444';
  if (count >= 6)  return '#f59e0b';
  return '#22c55e';
}

function crowdBadgeClass(count) {
  if (count >= 16) return 'red';
  if (count >= 6)  return 'amber';
  return 'green';
}

function crowdLabel(count) {
  if (count >= 16) return 'High';
  if (count >= 6)  return 'Moderate';
  return 'Low';
}

function createDivIcon(color) {
  return L.divIcon({
    className: '',
    html: `<div style="width:18px;height:18px;border-radius:50%;background:${color};border:2px solid rgba(255,255,255,0.6);box-shadow:0 0 6px ${color}88"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

function updateMapMarkers(cameras) {
  Object.values(mapCircles).forEach(cs => cs.forEach(c => leafletMap.removeLayer(c)));
  mapCircles = {};

  cameras.forEach(cam => {
    const info = CAMERA_COORDS[cam.camera_id] || {};
    const lat  = info.lat || 0;
    const lng  = info.lng || 0;
    const key  = cam.camera_id;
    const name = info.label || cam.camera_id;
    const count = cam.current_count || 0;
    const color = crowdColor(count);
    const lbl   = crowdLabel(count);

    const popup = `
      <div style="min-width:150px">
        <strong>${name}</strong><br/>
        Current: <b>${count}</b>
        <span style="margin-left:6px;padding:2px 8px;border-radius:20px;font-size:11px;font-weight:600;
          background:${color}22;color:${color}">${lbl}</span><br/>
        IN: ${cam.count_in || 0} &nbsp; OUT: ${cam.count_out || 0}<br/>
        <span style="font-size:11px;color:var(--muted)">FPS: ${(cam.fps || 0).toFixed(1)}</span>
      </div>`;

    if (!mapMarkers[key]) {
      mapMarkers[key] = L.marker([lat, lng], { icon: createDivIcon(color) })
        .addTo(leafletMap)
        .bindPopup(popup);
    } else {
      mapMarkers[key].setIcon(createDivIcon(color));
      mapMarkers[key].setPopupContent(popup);
    }

    const radius = Math.max(40, count * 12);
    mapCircles[key] = [
      L.circle([lat, lng], { color, fillColor: color, fillOpacity: 0.18, radius, stroke: false }).addTo(leafletMap),
    ];
  });
}

// ── Main data fetch (detection API) ─────────────────────────
async function fetchCrowdData() {
  try {
    const [metrics, cameras] = await Promise.all([
      apiFetch(`${PYTHON_API}/api/v1/metrics`).catch(() => null),
      apiFetch(`${PYTHON_API}/api/v1/cameras`).catch(() => []),
    ]);

    const totalIn       = metrics ? (metrics.total_in       || 0) : 0;
    const netOccupancy  = metrics ? (metrics.net_occupancy  || 0) : 0;
    const activeCameras = metrics ? (metrics.cameras_active || cameras.length || 0) : cameras.length;
    const avgFps        = metrics ? (metrics.avg_fps        || 0) : 0;
    const totalEvents   = metrics ? (metrics.events_logged  || 0) : 0;
    const now = new Date().toLocaleTimeString();

    // Overview KPI cards
    document.getElementById('kpiPeak').textContent      = totalIn;
    document.getElementById('kpiAvg').textContent       = netOccupancy;
    document.getElementById('kpiLocations').textContent = activeCameras;
    document.getElementById('kpiPreferred').textContent = avgFps.toFixed(1);
    document.getElementById('kpiBusiest').textContent   = totalEvents;

    // Map stats
    document.getElementById('mapPeak').textContent = totalIn;
    document.getElementById('mapAvg').textContent  = netOccupancy;

    // Least occupied zone
    if (cameras && cameras.length) {
      const least = cameras.reduce((a, b) =>
        (a.current_count || 0) <= (b.current_count || 0) ? a : b);
      document.getElementById('mapBest').textContent =
        `${cameraLabel(least.camera_id)} (${least.current_count || 0})`;
    }

    // Push to charts
    const totalOut = metrics ? (metrics.total_out || 0) : 0;
    pushToChart(ovPeakChart,     now, [totalIn]);
    pushToChart(ovAvgChart,      now, [totalOut]);
    pushToChart(ovForecastChart, now, [netOccupancy]);
    pushToChart(anPeakChart,     now, [totalIn]);
    pushToChart(anAvgChart,      now, [netOccupancy]);
    pushToChart(anForecastChart, now, [cameras.reduce((s, c) => s + (c.current_count || 0), 0)]);

    // Map markers
    if (leafletMap && cameras.length) updateMapMarkers(cameras);

    // Overview camera status table
    recentData = cameras;
    renderRecentTable(cameras);

    setBadge('ok');
  } catch (err) {
    console.error('fetchCrowdData error:', err);
    setBadge('error');
  }
}

function setBadge(status) {
  const badge = document.getElementById('systemBadge');
  const text  = document.getElementById('systemBadgeText');
  badge.className = 'topbar-badge';
  if (status === 'ok') {
    badge.classList.add('topbar-badge');
    text.textContent = 'Live';
  } else {
    badge.classList.add('warning');
    text.textContent = 'Degraded';
  }
}

// ── Overview camera status table ─────────────────────────────
function renderRecentTable(cameras) {
  const el = document.getElementById('recentTableContainer');
  if (!cameras || cameras.length === 0) {
    el.innerHTML = `<div class="state-box"><i class="fas fa-video-slash"></i><p>No cameras active. Start the Python detection server.</p></div>`;
    return;
  }
  el.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Camera Zone</th>
          <th>IN</th>
          <th>OUT</th>
          <th>Current</th>
          <th>FPS</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${cameras.map(cam => {
          const bc  = crowdBadgeClass(cam.current_count || 0);
          const bl  = crowdLabel(cam.current_count || 0);
          return `<tr>
            <td><strong>${cameraLabel(cam.camera_id)}</strong><br/>
              <span style="font-size:11px;color:var(--muted)">${cam.camera_id}</span></td>
            <td><strong>${cam.count_in || 0}</strong></td>
            <td><strong>${cam.count_out || 0}</strong></td>
            <td><strong>${cam.current_count || 0}</strong></td>
            <td style="font-size:12px">${(cam.fps || 0).toFixed(1)}</td>
            <td><span class="badge ${bc}">${bl}</span></td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

function filterRecentTable() {
  const q = document.getElementById('recentSearch').value.trim().toLowerCase();
  if (!q) { renderRecentTable(recentData); return; }
  renderRecentTable(recentData.filter(cam =>
    cameraLabel(cam.camera_id).toLowerCase().includes(q) ||
    cam.camera_id.toLowerCase().includes(q)
  ));
}

// ── Map search ───────────────────────────────────────────────
function searchOnMap() {
  const q = document.getElementById('mapSearchInput').value.trim().toLowerCase();
  const resultEl = document.getElementById('mapSearchResult');
  if (!q) { resultEl.textContent = 'Please enter a camera zone name.'; return; }

  for (const [id, info] of Object.entries(CAMERA_COORDS)) {
    if (info.label.toLowerCase().includes(q) || id.toLowerCase().includes(q)) {
      leafletMap.setView([info.lat, info.lng], 18);
      if (mapMarkers[id]) mapMarkers[id].openPopup();
      resultEl.textContent = `Found: ${info.label} (${id})`;
      return;
    }
  }
  resultEl.textContent = 'No camera zone found with that name.';
}

// ── Cameras section ──────────────────────────────────────────
let shopsData = [];

async function loadShops() {
  const el = document.getElementById('shopsTableContainer');
  el.innerHTML = `<div class="state-box"><div class="spinner"></div><p>Loading cameras…</p></div>`;
  try {
    const cameras = await apiFetch(`${PYTHON_API}/api/v1/cameras`);
    shopsData = cameras;
    renderShopsTable(cameras);
  } catch (err) {
    console.error('loadShops error:', err);
    el.innerHTML = `<div class="state-box"><i class="fas fa-circle-exclamation"></i><p>Failed to load camera data. Is the Python API running?</p></div>`;
  }
}

function renderShopsTable(cameras) {
  const el = document.getElementById('shopsTableContainer');
  if (!cameras || cameras.length === 0) {
    el.innerHTML = `<div class="state-box"><i class="fas fa-video-slash"></i><p>No cameras configured.</p></div>`;
    return;
  }
  el.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Camera Zone</th>
          <th>Source</th>
          <th>IN</th>
          <th>OUT</th>
          <th>Now</th>
          <th>FPS</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${cameras.map(cam => {
          const bc  = crowdBadgeClass(cam.current_count || 0);
          const bl  = crowdLabel(cam.current_count || 0);
          const src = cam.source || cam.camera_id;
          return `<tr>
            <td>
              <strong>${cameraLabel(cam.camera_id)}</strong><br/>
              <span style="font-size:11px;color:var(--muted)">${cam.camera_id}</span>
            </td>
            <td style="font-size:12px;color:var(--muted)">
              ${String(src).split('/').pop()}
            </td>
            <td><strong>${cam.count_in || 0}</strong></td>
            <td><strong>${cam.count_out || 0}</strong></td>
            <td><strong>${cam.current_count || 0}</strong></td>
            <td style="font-size:12px">${(cam.fps || 0).toFixed(1)}</td>
            <td><span class="badge ${bc}">${bl}</span></td>
            <td>
              <button class="btn-sm secondary" style="padding:4px 10px;font-size:11px"
                onclick="resetCamera('${cam.camera_id}')">Reset</button>
            </td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

function filterShopsTable() {
  const q = document.getElementById('shopSearch').value.trim().toLowerCase();
  if (!q) { renderShopsTable(shopsData); return; }
  renderShopsTable(shopsData.filter(cam =>
    cameraLabel(cam.camera_id).toLowerCase().includes(q) ||
    cam.camera_id.toLowerCase().includes(q)
  ));
}

// ── Zone Rankings ────────────────────────────────────────────
let topShopsChart = null;

async function loadTopShops() {
  const el = document.getElementById('topShopsTableContainer');
  el.innerHTML = `<div class="state-box"><div class="spinner"></div><p>Loading…</p></div>`;
  try {
    const cameras = await apiFetch(`${PYTHON_API}/api/v1/cameras`);
    renderTopShops(cameras);
  } catch (err) {
    console.error('loadTopShops error:', err);
    el.innerHTML = `<div class="state-box"><i class="fas fa-circle-exclamation"></i><p>Failed to load zone data. Is the Python API running?</p></div>`;
  }
}

function renderTopShops(cameras) {
  const el = document.getElementById('topShopsTableContainer');
  if (!cameras || cameras.length === 0) {
    el.innerHTML = `<div class="state-box"><i class="fas fa-database"></i><p>No camera zones available.</p></div>`;
    return;
  }

  // Sort by total IN descending
  const sorted = [...cameras].sort((a, b) => (b.count_in || 0) - (a.count_in || 0));
  const labels = sorted.map(c => cameraLabel(c.camera_id));
  const values = sorted.map(c => c.count_in || 0);

  if (topShopsChart) topShopsChart.destroy();
  const ctx = document.getElementById('topShopsChart');
  if (ctx) {
    topShopsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Total IN',
          data: values,
          backgroundColor: ['#4f8ef7', '#22c55e', '#f59e0b', '#ef4444', '#7c5cbf'],
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } },
      },
    });
  }

  const totalIn = values.reduce((a, b) => a + b, 0);
  el.innerHTML = `
    <table>
      <thead>
        <tr><th>#</th><th>Zone</th><th>Total IN</th><th>Total OUT</th><th>Net Occ.</th><th>FPS</th><th>Share</th></tr>
      </thead>
      <tbody>
        ${sorted.map((cam, i) => {
          const pct = totalIn > 0 ? (((cam.count_in || 0) / totalIn) * 100).toFixed(1) : 0;
          const bc  = crowdBadgeClass(cam.current_count || 0);
          return `<tr>
            <td><strong>#${i + 1}</strong></td>
            <td>${cameraLabel(cam.camera_id)}</td>
            <td><strong>${cam.count_in || 0}</strong></td>
            <td>${cam.count_out || 0}</td>
            <td><span class="badge ${bc}">${cam.current_count || 0}</span></td>
            <td style="font-size:12px">${(cam.fps || 0).toFixed(1)}</td>
            <td style="min-width:120px">
              <div style="display:flex;align-items:center;gap:8px">
                <div style="flex:1;height:6px;border-radius:3px;background:var(--border)">
                  <div style="width:${pct}%;height:100%;border-radius:3px;background:var(--accent)"></div>
                </div>
                <span style="font-size:12px;color:var(--muted)">${pct}%</span>
              </div>
            </td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

// ── Occupancy section (was Heatmap) ──────────────────────────
async function loadHeatmap() {
  const el = document.getElementById('heatmapContainer');
  el.innerHTML = `<div class="state-box"><div class="spinner"></div><p>Loading occupancy data…</p></div>`;
  try {
    const cameras = await apiFetch(`${PYTHON_API}/api/v1/cameras`);
    if (!cameras || cameras.length === 0) {
      el.innerHTML = `<div class="state-box"><i class="fas fa-fire"></i><p>No occupancy data. Start the detection server.</p></div>`;
      return;
    }
    const maxCount = Math.max(...cameras.map(c => c.current_count || 0), 1);
    el.innerHTML = `<div class="heatmap-list">
      ${cameras.map(cam => {
        const count = cam.current_count || 0;
        const pct   = ((count / maxCount) * 100).toFixed(0);
        const color = crowdColor(count);
        const lbl   = crowdLabel(count);
        return `<div class="heatmap-item">
          <div class="coord">${cam.camera_id}</div>
          <div style="font-weight:600;margin-bottom:4px;font-size:13px">${cameraLabel(cam.camera_id)}</div>
          <div class="count" style="color:${color}">${count}</div>
          <div style="height:6px;border-radius:3px;background:var(--border);margin-top:8px;overflow:hidden">
            <div style="width:${pct}%;height:100%;border-radius:3px;background:${color}"></div>
          </div>
          <div style="font-size:11px;color:var(--muted);margin-top:4px">
            <span style="color:${color}">${lbl}</span> &bull; ${pct}% of peak &bull; IN: ${cam.count_in || 0}
          </div>
        </div>`;
      }).join('')}
    </div>`;
  } catch (err) {
    console.error('loadHeatmap error:', err);
    el.innerHTML = `<div class="state-box"><i class="fas fa-circle-exclamation"></i><p>Failed to load occupancy data.</p></div>`;
  }
}

// ── Session History ──────────────────────────────────────────
async function loadHistory() {
  const cameraFilter = document.getElementById('histShop').value;
  const tableEl = document.getElementById('histTableContainer');
  tableEl.innerHTML = `<div class="state-box"><div class="spinner"></div><p>Loading sessions…</p></div>`;

  try {
    // Load MAE evaluation summary
    try {
      const evalData = await apiFetch('/api/evaluation');
      document.getElementById('maeCount').textContent = evalData.sessions_evaluated ?? '—';
      document.getElementById('maeValue').textContent =
        evalData.mae != null ? evalData.mae.toFixed(2) : '—';
    } catch (_) {}

    // Load sessions
    const url = cameraFilter
      ? `/api/sessions?camera_id=${encodeURIComponent(cameraFilter)}`
      : '/api/sessions';
    const sessions = await apiFetch(url);
    renderSessionsTable(sessions);
  } catch (err) {
    console.error('loadHistory error:', err);
    tableEl.innerHTML = `<div class="state-box"><i class="fas fa-circle-exclamation"></i><p>Failed to fetch sessions. Is the Node.js server running?</p></div>`;
  }
}

function renderSessionsTable(sessions) {
  const tableEl = document.getElementById('histTableContainer');

  if (!Array.isArray(sessions) || sessions.length === 0) {
    tableEl.innerHTML = `<div class="state-box"><i class="fas fa-inbox"></i><p>No sessions found.</p></div>`;
    return;
  }

  tableEl.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Camera</th>
          <th>Started</th>
          <th>IN</th>
          <th>OUT</th>
          <th>Ground Truth</th>
          <th>Abs. Error</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        ${sessions.map(s => {
          const started = s.started_at ? new Date(s.started_at).toLocaleString() : '—';
          const gt = s.ground_truth != null ? s.ground_truth : '—';
          const err = (s.ground_truth != null)
            ? Math.abs((s.count_in || 0) - s.ground_truth) : '—';
          return `<tr>
            <td><strong>${cameraLabel(s.camera_id) || s.camera_id || '—'}</strong></td>
            <td style="font-size:12px">${started}</td>
            <td>${s.count_in || 0}</td>
            <td>${s.count_out || 0}</td>
            <td>${gt}</td>
            <td>${err !== '—' ? `<span class="badge ${err === 0 ? 'green' : 'amber'}">${err}</span>` : '—'}</td>
            <td>
              <button class="btn-sm secondary" style="padding:4px 10px;font-size:11px"
                onclick="openGtForm('${s._id || s.id}', ${s.ground_truth != null ? s.ground_truth : 'null'})">
                Set GT
              </button>
            </td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

function openGtForm(sessionId, currentGt) {
  document.getElementById('gtFormWrap').style.display = 'block';
  document.getElementById('gtSessionId').value = sessionId;
  document.getElementById('gtValue').value = currentGt != null ? currentGt : '';
  document.getElementById('gtFormWrap').scrollIntoView({ behavior: 'smooth' });
}

async function saveGroundTruth() {
  const id  = document.getElementById('gtSessionId').value;
  const val = parseInt(document.getElementById('gtValue').value, 10);
  if (!id || isNaN(val)) { alert('Please enter a valid count.'); return; }
  try {
    const resp = await fetch(`/api/sessions/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ground_truth: val }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    document.getElementById('gtFormWrap').style.display = 'none';
    loadHistory();
  } catch (err) {
    console.error('saveGroundTruth error:', err);
    alert('Failed to save ground truth.');
  }
}

// ── Analytics — Camera Comparison ───────────────────────────
let compareChart = null;

async function loadCompareChart() {
  const cam1 = document.getElementById('compareShop1').value;
  const cam2 = document.getElementById('compareShop2').value;

  const name1 = cameraLabel(cam1);
  const name2 = cameraLabel(cam2);

  try {
    // Pull recent events for each camera from the detection API
    const [d1, d2] = await Promise.all([
      apiFetch(`${PYTHON_API}/api/v1/cameras/${cam1}/events`).catch(() => []),
      apiFetch(`${PYTHON_API}/api/v1/cameras/${cam2}/events`).catch(() => []),
    ]);

    const toSeries = (evts) => {
      if (!Array.isArray(evts)) return { labels: [], data: [] };
      const sorted = [...evts].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      let cumIn = 0;
      return {
        labels: sorted.map(e => new Date(e.timestamp).toLocaleTimeString()),
        data:   sorted.map(e => { if (e.direction === 'in') cumIn++; return cumIn; }),
      };
    };

    const s1 = toSeries(d1);
    const s2 = toSeries(d2);
    const labels = s1.labels.length >= s2.labels.length ? s1.labels : s2.labels;

    if (compareChart) compareChart.destroy();
    const ctx = document.getElementById('anCompareChart');
    if (ctx) {
      compareChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            { label: name1, data: s1.data, borderColor: '#4f8ef7', backgroundColor: '#4f8ef722', fill: true, tension: 0.35, pointRadius: 3 },
            { label: name2, data: s2.data, borderColor: '#22c55e', backgroundColor: '#22c55e22', fill: true, tension: 0.35, pointRadius: 3 },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: true } },
          plugins: { legend: { display: true, labels: { color: getThemeColor('--text', '#0f172a') } } },
        },
      });
    }
  } catch (err) {
    console.error('loadCompareChart error:', err);
  }
}

// ── Health ───────────────────────────────────────────────────
async function loadHealth() {
  await Promise.allSettled([loadNodeHealth(), loadPyHealth(), loadOrchHealth(), loadDetEngineHealth()]);
}

async function loadNodeHealth() {
  const badge = document.getElementById('nodeHealthBadge');
  const body  = document.getElementById('nodeHealthBody');
  try {
    const data = await apiFetch('/health');
    badge.className = 'badge green';
    badge.textContent = data.status === 'ok' ? 'OK' : 'Degraded';
    body.innerHTML = renderHealthRows({
      Status:    data.status,
      Version:   data.version || '—',
      Service:   data.service || '—',
      Timestamp: data.timestamp ? new Date(data.timestamp).toLocaleString() : '—',
    });
  } catch (err) {
    badge.className = 'badge red';
    badge.textContent = 'Unreachable';
    body.innerHTML = `<div class="state-box" style="padding:20px"><i class="fas fa-circle-xmark"></i><p>Could not reach /health</p></div>`;
  }
}

async function loadPyHealth() {
  const badge = document.getElementById('pyHealthBadge');
  const body  = document.getElementById('pyHealthBody');
  try {
    const data = await apiFetch(`${PYTHON_API}/health`);
    const overall = data.status || 'unknown';
    badge.className = overall === 'ok' ? 'badge green' : 'badge amber';
    badge.textContent = overall === 'ok' ? 'OK' : 'Degraded';
    const rows = { 'Overall Status': overall };
    if (data.providers && typeof data.providers === 'object') {
      Object.entries(data.providers).forEach(([k, v]) => {
        rows[k] = typeof v === 'object' ? (v.status || JSON.stringify(v)) : v;
      });
    }
    body.innerHTML = renderHealthRows(rows);
  } catch (err) {
    badge.className = 'badge muted';
    badge.textContent = 'Offline';
    body.innerHTML = `<div class="state-box" style="padding:20px"><i class="fas fa-plug-circle-xmark"></i><p>Python API not reachable (port 8000).<br/>Start with: <code style="font-size:11px">uvicorn api_server:app --port 8000</code></p></div>`;
  }
}

async function loadOrchHealth() {
  const badge = document.getElementById('orchHealthBadge');
  const body  = document.getElementById('orchHealthBody');
  try {
    const data = await apiFetch(`${PYTHON_API}/api/v1/orchestrator/health`);
    badge.className = 'badge blue';
    badge.textContent = 'Active';
    const rows = {};
    if (data.success_rate  !== undefined) rows['Success Rate']  = `${(data.success_rate * 100).toFixed(1)}%`;
    if (data.fallback_rate !== undefined) rows['Fallback Rate'] = `${(data.fallback_rate * 100).toFixed(1)}%`;
    if (data.total_calls   !== undefined) rows['Total Calls']   = data.total_calls;
    if (data.providers && typeof data.providers === 'object') {
      Object.entries(data.providers).forEach(([k, v]) => {
        rows[`Provider: ${k}`] = typeof v === 'object' ? (v.state || JSON.stringify(v)) : v;
      });
    }
    body.innerHTML = Object.keys(rows).length ? renderHealthRows(rows) :
      `<div class="state-box" style="padding:20px"><i class="fas fa-check-circle"></i><p>Orchestrator active</p></div>`;
  } catch (err) {
    badge.className = 'badge muted';
    badge.textContent = 'Offline';
    body.innerHTML = `<div class="state-box" style="padding:20px"><i class="fas fa-plug-circle-xmark"></i><p>Orchestrator API not reachable (port 8000).</p></div>`;
  }
}

function renderHealthRows(rows) {
  return Object.entries(rows).map(([k, v]) => `
    <div class="health-row">
      <span class="health-key">${k}</span>
      <span class="health-val">${v}</span>
    </div>`).join('');
}

async function loadDetEngineHealth() {
  const badge = document.getElementById('detEngineHealthBadge');
  const body  = document.getElementById('detEngineHealthBody');
  if (!badge || !body) return;
  try {
    const data = await apiFetch(`${PYTHON_API}/api/v1/metrics`);
    const active = data.cameras_active || 0;
    badge.className = active > 0 ? 'badge green' : 'badge amber';
    badge.textContent = active > 0 ? 'Running' : 'Idle';
    body.innerHTML = renderHealthRows({
      'Model':           data.model || 'yolov8n',
      'Cameras Active':  active,
      'Avg FPS':         (data.avg_fps || 0).toFixed(1),
      'Total IN':        data.total_in || 0,
      'Net Occupancy':   data.net_occupancy || 0,
      'Events Logged':   data.events_logged || 0,
    });
  } catch (err) {
    badge.className = 'badge muted';
    badge.textContent = 'Offline';
    body.innerHTML = `<div class="state-box" style="padding:20px"><i class="fas fa-plug-circle-xmark"></i><p>Detection engine not reachable (port 8000).</p></div>`;
  }
}

// ── Global refresh ───────────────────────────────────────────
function refreshAll() {
  fetchCrowdData();
  if (currentSection === 'shops')          loadShops();
  if (currentSection === 'top-shops')      loadTopShops();
  if (currentSection === 'heatmap')        loadHeatmap();
  if (currentSection === 'health')         loadHealth();
  if (currentSection === 'live-detection') loadDetectionData();
  if (currentSection === 'history')        loadHistory();
}

// ── Live Detection ───────────────────────────────────────────
let _detectionPollTimer = null;

async function loadDetectionData() {
  await Promise.all([loadCameraFeeds(), loadDetectionMetrics(), loadCrossingEvents()]);
}

async function loadDetectionMetrics() {
  try {
    const m = await fetch(`${PYTHON_API}/api/v1/metrics`).then(r => r.json());
    document.getElementById('metCameras').textContent   = m.cameras_active ?? '—';
    document.getElementById('metTotalIn').textContent   = m.total_in ?? '—';
    document.getElementById('metTotalOut').textContent  = m.total_out ?? '—';
    document.getElementById('metOccupancy').textContent = m.net_occupancy ?? '—';
    document.getElementById('metFps').textContent       = (m.avg_fps ?? '—') + (m.avg_fps != null ? ' fps' : '');
  } catch (_) {
    ['metCameras','metTotalIn','metTotalOut','metOccupancy','metFps']
      .forEach(id => document.getElementById(id).textContent = '—');
  }
}

async function loadCameraFeeds() {
  const grid = document.getElementById('cameraGrid');
  try {
    const cameras = await fetch(`${PYTHON_API}/api/v1/cameras`).then(r => r.json());
    if (!cameras.length) {
      grid.innerHTML = `<div class="state-box"><i class="fas fa-camera-slash"></i><p>No cameras configured. Check cameras.json and restart the Python API.</p></div>`;
      return;
    }
    // Keep the USB live webcam featured at the top; demo feeds follow below.
    const orderedCameras = [...cameras].sort((a, b) => {
      const aLive = a.camera_id === 'kiu-webcam' ? 0 : 1;
      const bLive = b.camera_id === 'kiu-webcam' ? 0 : 1;
      return aLive - bLive;
    });

    const cards = Array.from(grid.querySelectorAll('.camera-card'));
    const renderedIds = cards.map(card => card.id.replace('card-', ''));
    const expectedIds = orderedCameras.map(c => c.camera_id);
    const needsRender =
      cards.length === 0 ||
      renderedIds.length !== expectedIds.length ||
      renderedIds.some((id, idx) => id !== expectedIds[idx]);

    if (needsRender) {
      grid.innerHTML = orderedCameras.map(c => _renderCameraCard(c)).join('');

      // Wire up image error → offline state
      grid.querySelectorAll('.cam-feed-img').forEach(img => {
        img.onerror = () => {
          img.style.display = 'none';
          if (img.nextElementSibling) img.nextElementSibling.style.display = 'flex';
        };
      });
    }

    // Real-time values update without recreating stream elements.
    orderedCameras.forEach(c => _updateCameraCard(c));
  } catch (_) {
    grid.innerHTML = `<div class="state-box"><i class="fas fa-plug-circle-xmark"></i><p>Python API unreachable. Start with: <code>uvicorn api_server:app --port 8000</code></p></div>`;
  }
}

function _toOneDecimal(n) {
  const v = Number(n);
  return Number.isFinite(v) ? v.toFixed(1) : '0.0';
}

function _updateCameraCard(c) {
  const card = document.getElementById(`card-${c.camera_id}`);
  if (!card) return;

  const streamUrl = `${PYTHON_API}/stream/${c.camera_id}`;
  const isRunning = !!c.running;

  const badge = card.querySelector('.cam-badge');
  if (badge) {
    badge.className = `cam-badge ${isRunning ? 'live' : 'off'}`;
    badge.innerHTML = isRunning ? '<span class="dot"></span> LIVE' : 'OFFLINE';
  }

  const img = card.querySelector('.cam-feed-img');
  const offline = card.querySelector('.feed-offline');
  if (img && offline) {
    if (isRunning) {
      if (!img.getAttribute('src')) img.setAttribute('src', streamUrl);
      img.style.display = '';
      offline.style.display = 'none';
    } else {
      img.style.display = 'none';
      offline.style.display = 'flex';
    }
  }

  const inEl = card.querySelector('[data-field="count_in"]');
  const outEl = card.querySelector('[data-field="count_out"]');
  const nowEl = card.querySelector('[data-field="current_count"]');
  const fpsEl = card.querySelector('[data-field="fps"]');
  if (inEl) inEl.textContent = `${c.count_in ?? 0}`;
  if (outEl) outEl.textContent = `${c.count_out ?? 0}`;
  if (nowEl) nowEl.textContent = `${c.current_count ?? 0}`;
  if (fpsEl) fpsEl.textContent = _toOneDecimal(c.fps);

  const sessionEl = card.querySelector('.session-time');
  if (sessionEl) {
    sessionEl.textContent = c.session_start ? c.session_start.slice(0, 19).replace('T', ' ') : '—';
  }
}

function _renderCameraCard(c) {
  const streamUrl = `${PYTHON_API}/stream/${c.camera_id}`;
  const isRunning = c.running;
  const isLiveWebcam = c.camera_id === 'kiu-webcam';
  return `
    <div class="camera-card${isLiveWebcam ? ' featured-live' : ''}" id="card-${c.camera_id}">
      <div class="camera-card-header">
        <i class="fas fa-video" style="color:var(--accent)"></i>
        <h3>${c.label}</h3>
        <span class="cam-badge ${isRunning ? 'live' : 'off'}">
          ${isRunning ? '<span class="dot"></span> LIVE' : 'OFFLINE'}
        </span>
      </div>
      <div class="camera-feed">
        <img class="cam-feed-img"
             src="${isRunning ? streamUrl : ''}"
             alt="${c.label} stream"
             style="${isRunning ? '' : 'display:none'}">
        <div class="feed-offline" style="${isRunning ? 'display:none' : ''}">
          <i class="fas fa-video-slash" style="font-size:32px"></i>
          <span>Camera offline</span>
        </div>
      </div>
      <div class="camera-stats">
        <div class="cam-stat">
          <div class="cam-stat-label">IN</div>
          <div class="cam-stat-value in" data-field="count_in">${c.count_in ?? 0}</div>
        </div>
        <div class="cam-stat">
          <div class="cam-stat-label">OUT</div>
          <div class="cam-stat-value out" data-field="count_out">${c.count_out ?? 0}</div>
        </div>
        <div class="cam-stat">
          <div class="cam-stat-label">NOW</div>
          <div class="cam-stat-value now" data-field="current_count">${c.current_count ?? 0}</div>
        </div>
        <div class="cam-stat">
          <div class="cam-stat-label">FPS</div>
          <div class="cam-stat-value fps" data-field="fps">${_toOneDecimal(c.fps)}</div>
        </div>
      </div>
      <div class="camera-footer">
        <span style="font-size:11px;color:var(--muted)">
          <i class="fas fa-clock"></i> Session: <span class="session-time">${c.session_start ? c.session_start.slice(0,19).replace('T',' ') : '—'}</span>
        </span>
        <button class="btn-sm secondary" style="margin-left:auto"
                onclick="resetCameracounts('${c.camera_id}')">
          <i class="fas fa-arrow-rotate-left"></i> Reset
        </button>
      </div>
    </div>`;
}

async function resetCameracounts(cameraId) {
  try {
    await fetch(`${PYTHON_API}/api/v1/cameras/${cameraId}/reset`, { method: 'POST' });
    loadDetectionData();
  } catch (e) {
    alert('Could not reset — is the Python API running?');
  }
}

async function loadCrossingEvents() {
  const container = document.getElementById('eventsTableContainer');
  const countEl   = document.getElementById('eventCount');
  try {
    // Fetch per-camera events and merge
    const cameras = await apiFetch(`${PYTHON_API}/api/v1/cameras`).catch(() => []);
    const all = await Promise.all(
      cameras.map(c =>
        apiFetch(`${PYTHON_API}/api/v1/cameras/${c.camera_id}/events?limit=20`).catch(() => [])
      )
    );
    const events = all.flat()
      .sort((a, b) => (b.timestamp || '').localeCompare(a.timestamp || ''))
      .slice(0, 50);

    countEl.textContent = `${events.length} event${events.length !== 1 ? 's' : ''}`;
    if (!events.length) {
      container.innerHTML = `<div class="state-box"><i class="fas fa-arrow-right-arrow-left"></i><p>No crossing events yet.</p></div>`;
      return;
    }
    container.innerHTML = `
      <table>
        <thead><tr><th>Camera</th><th>Track ID</th><th>Direction</th><th>Time</th></tr></thead>
        <tbody>
          ${events.map(e => `
            <tr>
              <td>${e.camera_id}</td>
              <td>#${e.track_id}</td>
              <td>
                <span class="badge ${e.direction === 'in' ? 'green' : 'red'}">
                  ${e.direction === 'in' ? '→ IN' : '← OUT'}
                </span>
              </td>
              <td style="color:var(--muted);font-size:12px">${e.timestamp.slice(0,19).replace('T',' ')}</td>
            </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (_) {
    container.innerHTML = `<div class="state-box"><i class="fas fa-plug-circle-xmark"></i><p>Python API unreachable.</p></div>`;
  }
}

// Auto-refresh Live Detection when it's the active section
setInterval(() => {
  if (currentSection === 'live-detection') {
    // Fast lane: update camera card IN / OUT / NOW / FPS every second.
    loadCameraFeeds();
  }
}, LIVE_CAMERA_STATS_INTERVAL);

setInterval(() => {
  if (currentSection === 'live-detection') {
    // Slow lane: heavier requests at 5s cadence.
    Promise.all([loadDetectionMetrics(), loadCrossingEvents()]);
  }
}, LIVE_DETECTION_REFRESH_INTERVAL);

// ── Initialise ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initMap();
  fetchCrowdData();
  setInterval(fetchCrowdData, REFRESH_INTERVAL);
});
