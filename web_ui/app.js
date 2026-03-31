const panelArea = document.getElementById('panel-area');
const installArea = document.getElementById('install-area');
const metricsSummary = document.getElementById('metrics-summary');
const viserFrame = document.getElementById('viser-frame');
const resetPanelsBtn = document.getElementById('reset-panels');
const resetLayoutBtn = document.getElementById('reset-layout');
const themeToggle = document.getElementById('theme-toggle');
const appRoot = document.querySelector('.app');
const desktop = document.getElementById('desktop');
const windowDock = document.getElementById('window-dock');
const taskbar = document.getElementById('taskbar');
const primarySelect = document.getElementById('primary-method');
const secondarySelect = document.getElementById('secondary-method');
const renderMethodSelect = document.getElementById('render-method-select');
const panelMethodSelect = document.getElementById('panel-method');
const addPanelBtn = document.getElementById('add-panel');
const loadSceneBtn = document.getElementById('load-scene');
const liveRenderBtn = document.getElementById('live-render');
const generateReportBtn = document.getElementById('generate-report');
const refreshHistoryBtn = document.getElementById('refresh-history');
const compareSelectedBtn = document.getElementById('compare-selected');
const historyBody = document.getElementById('history-body');
const historyFilterMethod = document.getElementById('history-filter-method');
const historyFilterStatus = document.getElementById('history-filter-status');
const historyFilterDataset = document.getElementById('history-filter-dataset');
const experimentVisuals = document.getElementById('experiment-visuals');
const sceneDatasetSelect = document.getElementById('scene-dataset');
const sceneSplitSelect = document.getElementById('scene-split');
const sceneFrameSelect = document.getElementById('scene-frame');
const renderImage = document.getElementById('render-image');
const renderFallback = document.getElementById('render-fallback');
const renderStatus = document.getElementById('render-status');
const activeMetrics = document.getElementById('active-metrics');
const panelCollapse = document.getElementById('panel-collapse');
const panelBody = document.getElementById('panel-body');
const snapOverlay = document.getElementById('snap-overlay');
const windowElements = Array.from(document.querySelectorAll('.os-window'));

const WINDOW_LAYOUT_KEY = 'nvs-window-layout-v1';
const SNAP_THRESHOLD = 30;
const CORNER_SNAP_RATIO = 0.5;

const state = {
  panels: [],
  metrics: {},
  metricsError: '',
  metricsFile: '/artifacts/metrics/latest_preview.json',
  artifactsRoot: '/artifacts',
  catalog: { datasets: [], methods: [], notes: [] },
  methodLabels: new Map(),
  installedMethods: [],
  scenes: [],
  selectedSceneId: '',
  experiments: [],
  selectedRunIds: new Set(),
  experimentsFilters: {
    method: '',
    status: '',
    dataset: '',
  },
  windows: {},
  zSeed: 20,
};

const WINDOW_TITLES = {
  preview: 'Previa espacial',
  render: 'Render da camera',
  controls: 'Controles rapidos',
  panels: 'Paineis',
  install: 'Instalacao',
};

const toFiniteNumber = (value) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const formatMetric = (value, digits = 2) => {
  const num = toFiniteNumber(value);
  return num === null ? '--' : num.toFixed(digits);
};

const sanitizeLooseJson = (text) => {
  if (!text) return '{}';
  return text
    .replace(/\bNaN\b/g, 'null')
    .replace(/\bInfinity\b/g, 'null')
    .replace(/\b-Infinity\b/g, 'null');
};

const normalizeWebPath = (value, fallback) => {
  if (!value || typeof value !== 'string') return fallback;
  if (value.startsWith('http://') || value.startsWith('https://')) return value;
  return value.startsWith('/') ? value : `/${value}`;
};

const setTheme = (mode) => {
  appRoot.dataset.theme = mode;
  localStorage.setItem('nvs-theme', mode);
};

const initTheme = () => {
  const saved = localStorage.getItem('nvs-theme') || 'auto';
  themeToggle.value = saved;
  if (saved === 'auto') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  } else {
    setTheme(saved);
  }
};

const buildImageCandidates = (method, kind) => {
  const base = state.artifactsRoot;
  return [
    `${base}/custom-${method}/${method}/${kind}/frame_0000.png`,
    `${base}/${method}/${kind}/frame_0000.png`,
  ];
};

const resolveImage = (method, kind) => {
  return buildImageCandidates(method, kind)[0];
};

const setImageWithFallback = (img, candidates) => {
  const urls = (candidates || []).filter(Boolean);
  if (!urls.length) {
    img.removeAttribute('src');
    return;
  }
  let index = 0;
  const tryNext = () => {
    if (index >= urls.length) {
      img.removeAttribute('src');
      return;
    }
    img.src = urls[index++];
  };
  img.onerror = () => tryNext();
  tryNext();
};

const normalizeMetricEntry = (method, payload) => {
  const data = payload && typeof payload === 'object' ? payload : {};
  return {
    method,
    kind: data.kind || 'benchmark',
    psnr: toFiniteNumber(data.psnr),
    ssim: toFiniteNumber(data.ssim),
    lpips: toFiniteNumber(data.lpips),
    fps: toFiniteNumber(data.fps),
    frame_time_ms: toFiniteNumber(data.frame_time_ms),
    vram_gb: toFiniteNumber(data.vram_gb),
    render_url: resolveImage(method, 'renders'),
    ref_url: resolveImage(method, 'references'),
  };
};

const buildMethodLabelMap = (catalog) => {
  const map = new Map();
  (catalog.methods || []).forEach((item) => {
    map.set(item.item_id, item.label || item.item_id);
  });
  return map;
};

const getMethodLabel = (methodId) => {
  return state.methodLabels.get(methodId) || methodId;
};

const populateSelect = (select, options, fallbackLabel) => {
  select.innerHTML = '';
  if (!options.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = fallbackLabel;
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  options.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = getMethodLabel(item);
    select.appendChild(opt);
  });
  select.disabled = false;
};

const buildMetricsMini = (title, entry) => {
  return `
    <div class="metrics-header">${title}</div>
    <div>PSNR <strong>${formatMetric(entry?.psnr, 2)}</strong></div>
    <div>SSIM <strong>${formatMetric(entry?.ssim, 3)}</strong></div>
    <div>LPIPS <strong>${formatMetric(entry?.lpips, 3)}</strong></div>
    <div>FPS <strong>${formatMetric(entry?.fps, 2)}</strong></div>
    <div>VRAM <strong>${formatMetric(entry?.vram_gb, 2)} GB</strong></div>
  `;
};

const updateActiveMetrics = () => {
  const primary = primarySelect.value;
  const secondary = secondarySelect.value;
  const primaryEntry = state.metrics[primary];
  const secondaryEntry = state.metrics[secondary];

  activeMetrics.innerHTML = `
    ${buildMetricsMini(`Base: ${getMethodLabel(primary)}`, primaryEntry)}
    <div class="metrics-divider"></div>
    ${buildMetricsMini(`Comparar: ${getMethodLabel(secondary)}`, secondaryEntry)}
  `;
};

const updateRender = () => {
  const methodId = renderMethodSelect?.value || primarySelect.value;
  if (!methodId) {
    renderStatus.textContent = 'Nenhum modelo instalado.';
    renderFallback.textContent = 'Nenhum modelo instalado. Instale um modelo para visualizar renders.';
    renderImage.removeAttribute('src');
    renderImage.classList.add('hidden');
    renderFallback.classList.remove('hidden');
    return;
  }
  const label = getMethodLabel(methodId);
  const candidates = buildImageCandidates(methodId, 'renders');

  renderStatus.textContent = `Modelo ativo: ${label}`;
  renderFallback.textContent = 'Nenhum render disponivel para este metodo.';
  setImageWithFallback(renderImage, candidates);
  renderImage.alt = `Render ${label}`;
};

renderImage.addEventListener('error', () => {
  renderImage.classList.add('hidden');
  renderFallback.classList.remove('hidden');
});

renderImage.addEventListener('load', () => {
  renderImage.classList.remove('hidden');
  renderFallback.classList.add('hidden');
});

const renderPanelGrid = () => {
  panelArea.innerHTML = '';
  if (!state.panels.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'Nenhum painel criado. Selecione um modelo instalado e clique em "Adicionar painel".';
    panelArea.appendChild(empty);
    return;
  }

  state.panels.forEach((methodId) => {
    const entry = state.metrics[methodId];
    const card = document.createElement('div');
    card.className = 'panel-card';
    card.innerHTML = `
      <div class="panel-card-header">
        <div class="panel-card-title">${getMethodLabel(methodId)}</div>
        <div class="panel-card-actions">
          <button class="icon-btn" data-remove="${methodId}" title="Remover">x</button>
        </div>
      </div>
      <img alt="Render ${getMethodLabel(methodId)}" />
      <div class="metrics-mini">
        ${buildMetricsMini('Metricas', entry)}
      </div>
    `;

    const img = card.querySelector('img');
    setImageWithFallback(img, buildImageCandidates(methodId, 'renders'));
    img.addEventListener('error', () => {
      img.replaceWith(Object.assign(document.createElement('div'), {
        className: 'empty-state',
        textContent: 'Render nao encontrado para este metodo.',
      }));
    });

    card.querySelector('[data-remove]').addEventListener('click', () => {
      state.panels = state.panels.filter((id) => id !== methodId);
      renderPanelGrid();
    });

    panelArea.appendChild(card);
  });
};

const resetPanels = (options) => {
  state.panels = [...options];
  renderPanelGrid();
};

const loadMetrics = async () => {
  state.metricsError = '';
  const response = await fetch(state.metricsFile, { cache: 'no-store' });
  if (!response.ok) {
    state.metricsError = `Arquivo de metricas nao encontrado: ${state.metricsFile}`;
    return {};
  }
  const rawText = await response.text();
  try {
    return JSON.parse(rawText);
  } catch {
    try {
      return JSON.parse(sanitizeLooseJson(rawText));
    } catch {
      state.metricsError = `Falha ao interpretar metricas (JSON invalido): ${state.metricsFile}`;
      return {};
    }
  }
};

const loadCatalog = async () => {
  const response = await fetch('/api/catalog');
  if (!response.ok) return { datasets: [], methods: [], notes: [] };
  return response.json();
};

const loadScenes = async () => {
  const response = await fetch('/api/scenes');
  if (!response.ok) return [];
  const payload = await response.json();
  return Array.isArray(payload.scenes) ? payload.scenes : [];
};

const loadExperiments = async (filters = {}) => {
  const params = new URLSearchParams();
  params.set('limit', '80');
  if (filters.method) params.set('method', filters.method);
  if (filters.status) params.set('status', filters.status);
  if (filters.dataset) params.set('dataset', filters.dataset);

  const response = await fetch(`/api/experiments?${params.toString()}`, { cache: 'no-store' });
  if (!response.ok) return [];
  const payload = await response.json();
  return Array.isArray(payload.experiments) ? payload.experiments : [];
};

const populateHistoryMethodFilter = () => {
  if (!historyFilterMethod) return;
  const methods = Array.from(new Set(state.experiments.map((item) => item.method).filter(Boolean)));
  const previous = historyFilterMethod.value;
  historyFilterMethod.innerHTML = '<option value="">Todos os metodos</option>';
  methods.forEach((methodId) => {
    const option = document.createElement('option');
    option.value = methodId;
    option.textContent = getMethodLabel(methodId);
    historyFilterMethod.appendChild(option);
  });
  if (methods.includes(previous)) {
    historyFilterMethod.value = previous;
  }
};

const renderExperimentVisuals = () => {
  if (!experimentVisuals) return;
  experimentVisuals.innerHTML = '';
  const selected = state.experiments.filter((item) => state.selectedRunIds.has(String(item.run_id || '')));
  if (selected.length < 2) {
    const hint = document.createElement('div');
    hint.className = 'history-empty';
    hint.textContent = 'Selecione pelo menos 2 runs para comparação visual.';
    experimentVisuals.appendChild(hint);
    return;
  }

  const winner = selected
    .slice()
    .sort((a, b) => (toFiniteNumber(b?.metrics_summary?.psnr) || -1) - (toFiniteNumber(a?.metrics_summary?.psnr) || -1))[0];

  selected.forEach((entry) => {
    const method = entry.method || '--';
    const runId = String(entry.run_id || '--');
    const summary = entry.metrics_summary || {};
    const card = document.createElement('article');
    card.className = 'experiment-card';
    if (winner && winner.run_id === entry.run_id) {
      card.classList.add('is-winner');
    }

    const imgBlock = entry.render_preview_url
      ? `<img src="${entry.render_preview_url}" alt="Render ${runId}" />`
      : '<div class="history-empty">Render indisponivel</div>';

    card.innerHTML = `
      <header>
        <strong>${getMethodLabel(method)}</strong>
        <span>${runId.slice(0, 26)}</span>
      </header>
      <div class="experiment-image">${imgBlock}</div>
      <div class="experiment-metrics">
        <span>PSNR <strong>${formatMetric(summary.psnr, 2)}</strong></span>
        <span>SSIM <strong>${formatMetric(summary.ssim, 3)}</strong></span>
        <span>LPIPS <strong>${formatMetric(summary.lpips, 3)}</strong></span>
        <span>FPS <strong>${formatMetric(summary.fps, 2)}</strong></span>
      </div>
    `;
    experimentVisuals.appendChild(card);
  });
};

const renderExperimentHistory = () => {
  if (!historyBody) return;
  historyBody.innerHTML = '';

  if (!state.experiments.length) {
    const row = document.createElement('tr');
    row.innerHTML = '<td colspan="6" class="history-empty">Nenhum experimento registrado.</td>';
    historyBody.appendChild(row);
    renderExperimentVisuals();
    return;
  }

  state.experiments.forEach((entry) => {
    const runId = String(entry.run_id || '');
    const method = entry.method || '--';
    const dataset = entry.dataset || '--';
    const summary = entry.metrics_summary || {};
    const psnr = toFiniteNumber(summary.psnr);
    const ssim = toFiniteNumber(summary.ssim);

    const row = document.createElement('tr');
    row.innerHTML = `
      <td><input type="checkbox" class="history-check" data-run-id="${runId}" /></td>
      <td title="${runId}">${runId.slice(0, 22)}</td>
      <td>${getMethodLabel(method)}</td>
      <td>${dataset}</td>
      <td>${psnr === null ? '--' : psnr.toFixed(2)}</td>
      <td>${ssim === null ? '--' : ssim.toFixed(3)}</td>
    `;
    historyBody.appendChild(row);
  });

  historyBody.querySelectorAll('.history-check').forEach((checkbox) => {
    checkbox.checked = state.selectedRunIds.has(checkbox.dataset.runId);
    checkbox.addEventListener('change', () => {
      const runId = checkbox.dataset.runId;
      if (!runId) return;
      if (checkbox.checked) {
        state.selectedRunIds.add(runId);
      } else {
        state.selectedRunIds.delete(runId);
      }
      if (compareSelectedBtn) {
        compareSelectedBtn.disabled = state.selectedRunIds.size < 2;
      }
      renderExperimentVisuals();
    });
  });

  renderExperimentVisuals();
};

const requestExperimentsComparison = async () => {
  const runIds = Array.from(state.selectedRunIds);
  if (runIds.length < 2) {
    window.alert('Selecione pelo menos 2 experimentos para comparar.');
    return;
  }

  const response = await fetch('/api/report-compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      run_ids: runIds,
      generate_pdf: false,
      report_name: `experiments_comparison_${Date.now()}`,
    }),
  });
  const payload = await response.json();
  if (!response.ok || payload.error) {
    window.alert(`Falha ao gerar comparação: ${payload.error || response.status}`);
    return;
  }

  const htmlPath = payload.html_path;
  if (typeof htmlPath === 'string' && htmlPath) {
    const reportUrl = htmlPath.startsWith('/') ? htmlPath : `/${htmlPath.replace(/^\.?\//, '')}`;
    window.open(reportUrl, '_blank', 'noopener,noreferrer');
  }
};

const selectedScene = () => state.scenes.find((item) => item.id === state.selectedSceneId) || null;

const selectedSceneSplit = () => {
  const scene = selectedScene();
  if (!scene) return null;
  const split = sceneSplitSelect?.value || '';
  return (scene.splits || []).find((item) => item.name === split) || null;
};

const populateSceneControls = (scenes) => {
  state.scenes = Array.isArray(scenes) ? scenes : [];
  sceneDatasetSelect.innerHTML = '';

  if (!state.scenes.length) {
    sceneDatasetSelect.disabled = true;
    sceneSplitSelect.disabled = true;
    sceneFrameSelect.disabled = true;
    loadSceneBtn.disabled = true;
    const empty = document.createElement('option');
    empty.value = '';
    empty.textContent = 'Nenhuma cena encontrada';
    sceneDatasetSelect.appendChild(empty);
    sceneSplitSelect.innerHTML = '<option value="">Sem split</option>';
    sceneFrameSelect.innerHTML = '<option value="">Sem imagem</option>';
    return;
  }

  state.scenes.forEach((scene) => {
    const option = document.createElement('option');
    option.value = scene.id;
    option.textContent = scene.label || scene.id;
    sceneDatasetSelect.appendChild(option);
  });

  state.selectedSceneId = state.scenes[0].id;
  sceneDatasetSelect.value = state.selectedSceneId;
  sceneDatasetSelect.disabled = false;
  loadSceneBtn.disabled = false;
  refreshSceneSplitOptions();
};

const refreshSceneSplitOptions = () => {
  const scene = selectedScene();
  sceneSplitSelect.innerHTML = '';
  if (!scene || !Array.isArray(scene.splits) || !scene.splits.length) {
    sceneSplitSelect.disabled = true;
    sceneFrameSelect.disabled = true;
    sceneSplitSelect.innerHTML = '<option value="">Sem split</option>';
    sceneFrameSelect.innerHTML = '<option value="">Sem imagem</option>';
    return;
  }

  scene.splits.forEach((split) => {
    const option = document.createElement('option');
    option.value = split.name;
    option.textContent = `${split.name} (${(split.frames || []).length})`;
    sceneSplitSelect.appendChild(option);
  });
  sceneSplitSelect.disabled = false;
  sceneSplitSelect.value = scene.splits[0].name;
  refreshSceneFrameOptions();
};

const refreshSceneFrameOptions = () => {
  const split = selectedSceneSplit();
  sceneFrameSelect.innerHTML = '';
  if (!split || !Array.isArray(split.frames) || !split.frames.length) {
    sceneFrameSelect.disabled = true;
    sceneFrameSelect.innerHTML = '<option value="">Sem imagem</option>';
    return;
  }

  split.frames.forEach((frame) => {
    const option = document.createElement('option');
    option.value = frame.web_path;
    option.textContent = frame.label || frame.web_path;
    sceneFrameSelect.appendChild(option);
  });
  sceneFrameSelect.disabled = false;
  sceneFrameSelect.value = split.frames[0].web_path;
};

const loadSelectedSceneFrame = () => {
  const scene = selectedScene();
  const split = selectedSceneSplit();
  const framePath = sceneFrameSelect?.value || '';

  if (!scene || !split || !framePath) {
    renderFallback.textContent = 'Nao foi possivel carregar a imagem selecionada da cena.';
    renderFallback.classList.remove('hidden');
    renderImage.classList.add('hidden');
    return;
  }

  renderImage.src = framePath;
  renderImage.alt = `Cena ${scene.label || scene.id} - ${split.name}`;
  renderStatus.textContent = `Cena: ${scene.label || scene.id} | Split: ${split.name}`;
  renderFallback.textContent = `A imagem selecionada nao pode ser carregada: ${framePath}`;

  openWindow('render');
  focusWindow('render');
  openWindow('preview');
};

const installItem = async (itemId) => {
  const response = await fetch('/api/install', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ item_id: itemId }),
  });
  const payload = await response.json();
  return payload;
};

const copyText = async (value) => {
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    return false;
  }
};

const renderInstallCards = (catalog) => {
  installArea.innerHTML = '';

  const makeSection = (title) => {
    const section = document.createElement('div');
    section.className = 'install-section';
    const heading = document.createElement('h3');
    heading.textContent = title;
    section.appendChild(heading);
    const grid = document.createElement('div');
    grid.className = 'install-grid';
    section.appendChild(grid);
    return section;
  };

  const makeCard = (item, kind) => {
    const card = document.createElement('div');
    card.className = 'install-card';
    card.innerHTML = `
      <h3>${item.label}</h3>
      <div class="install-meta">Path: ${item.path || '--'}</div>
      <div class="install-meta">Size: ${item.size_mb ? item.size_mb + ' MB' : '--'}</div>
      <div class="install-meta">Command: ${item.command || '--'}</div>
    `;

    const actions = document.createElement('div');
    actions.className = 'install-actions';

    const installBtn = document.createElement('button');
    installBtn.className = 'btn';
    installBtn.textContent = kind === 'dataset' ? 'Baixar' : 'Clone';
    installBtn.disabled = item.installed || !item.command;

    const openBtn = document.createElement('button');
    openBtn.className = 'btn secondary';
    openBtn.textContent = 'Abrir link';
    openBtn.disabled = !item.url;

    const copyBtn = document.createElement('button');
    copyBtn.className = 'btn secondary';
    copyBtn.textContent = 'Copiar comando';
    copyBtn.disabled = !item.command;

    installBtn.addEventListener('click', async () => {
      installBtn.disabled = true;
      const result = await installItem(item.item_id);
      const status = document.createElement('div');
      status.className = 'install-status';
      status.textContent = (result.messages || result.error || 'Erro').toString();
      card.appendChild(status);
      installBtn.disabled = false;
      const updated = await loadCatalog();
      renderInstallCards(updated);
    });

    openBtn.addEventListener('click', () => {
      if (item.url) {
        window.open(item.url, '_blank');
      }
    });

    copyBtn.addEventListener('click', async () => {
      const ok = await copyText(item.command || '');
      const status = document.createElement('div');
      status.className = 'install-status';
      status.textContent = ok ? 'Comando copiado.' : 'Falha ao copiar.';
      card.appendChild(status);
    });

    actions.appendChild(installBtn);
    actions.appendChild(openBtn);
    actions.appendChild(copyBtn);
    card.appendChild(actions);

    const status = document.createElement('div');
    status.className = 'install-status';
    if (item.installed) {
      status.textContent = 'Status: pronto (ja instalado).';
    } else if (!item.command) {
      status.textContent = 'Status: comando nao configurado.';
    } else {
      status.textContent = 'Status: pronto para instalar.';
    }
    card.appendChild(status);

    return card;
  };

  const datasets = Array.isArray(catalog.datasets) ? catalog.datasets : [];
  const methods = Array.isArray(catalog.methods) ? catalog.methods : [];

  const datasetSection = makeSection('Datasets');
  const datasetGrid = datasetSection.querySelector('.install-grid');
  datasets.forEach((item) => datasetGrid.appendChild(makeCard(item, 'dataset')));
  installArea.appendChild(datasetSection);

  const methodSection = makeSection('Modelos');
  const methodGrid = methodSection.querySelector('.install-grid');
  methods.forEach((item) => methodGrid.appendChild(makeCard(item, 'method')));
  installArea.appendChild(methodSection);

  if (Array.isArray(catalog.notes) && catalog.notes.length) {
    const notes = document.createElement('div');
    notes.className = 'install-notes';
    notes.textContent = catalog.notes.join(' | ');
    installArea.appendChild(notes);
  }
};

const renderMetricsSummary = (panels, metricsError) => {
  metricsSummary.innerHTML = '';

  if (metricsError) {
    const warn = document.createElement('div');
    warn.className = 'summary-alert';
    warn.textContent = metricsError;
    metricsSummary.appendChild(warn);
  }

  if (!panels.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'Sem metricas disponiveis. Gere um snapshot para comparar metodos.';
    metricsSummary.appendChild(empty);
    return;
  }

  const sorted = [...panels].sort((a, b) => {
    const pa = toFiniteNumber(a.psnr) ?? -1;
    const pb = toFiniteNumber(b.psnr) ?? -1;
    return pb - pa;
  });

  sorted.slice(0, 6).forEach((item, index) => {
    const card = document.createElement('div');
    card.className = 'summary-card';
    card.innerHTML = `
      <div class="summary-method">${getMethodLabel(item.method)}</div>
      <div class="summary-rank">#${index + 1} em PSNR</div>
      <div class="summary-grid">
        <span>PSNR <strong>${formatMetric(item.psnr, 2)}</strong></span>
        <span>SSIM <strong>${formatMetric(item.ssim, 3)}</strong></span>
        <span>LPIPS <strong>${formatMetric(item.lpips, 3)}</strong></span>
        <span>FPS <strong>${formatMetric(item.fps, 2)}</strong></span>
      </div>
    `;
    metricsSummary.appendChild(card);
  });
};

const buildPanels = (metrics) => {
  const entries = Object.entries(metrics || {}).filter((entry) => typeof entry[1] === 'object');
  if (entries.length === 0) {
    return [];
  }
  return entries.map(([method, payload]) => normalizeMetricEntry(method, payload));
};

const isMobileMode = () => window.innerWidth < 920;
const isTiledDesktopMode = () => false;

const getDesktopBounds = () => {
  const taskbarHeight = (taskbar?.offsetHeight || 0) + 12;
  return {
    width: Math.max(320, desktop.clientWidth),
    height: Math.max(320, desktop.clientHeight - taskbarHeight),
  };
};

const showSnapOverlay = (rect) => {
  if (!snapOverlay || isMobileMode() || !rect) return;
  snapOverlay.classList.remove('hidden');
  snapOverlay.style.transform = `translate(${Math.round(rect.x)}px, ${Math.round(rect.y)}px)`;
  snapOverlay.style.width = `${Math.round(rect.width)}px`;
  snapOverlay.style.height = `${Math.round(rect.height)}px`;
};

const hideSnapOverlay = () => {
  if (!snapOverlay) return;
  snapOverlay.classList.add('hidden');
};

const getSnapRect = (target, bounds) => {
  const EDGE_BUFFER = 40;  // conservative buffer to prevent overflow
  const maxWidth = bounds.width - EDGE_BUFFER;
  const halfWidth = Math.floor(maxWidth * CORNER_SNAP_RATIO);
  const halfHeight = Math.floor(bounds.height * CORNER_SNAP_RATIO);
  if (target === 'left') return { x: 0, y: 0, width: Math.floor(maxWidth / 2), height: bounds.height };
  if (target === 'right') return { x: Math.ceil(maxWidth / 2), y: 0, width: Math.floor(maxWidth / 2), height: bounds.height };
  if (target === 'top-left') return { x: 0, y: 0, width: halfWidth, height: halfHeight };
  if (target === 'top-right') return { x: maxWidth - halfWidth, y: 0, width: halfWidth, height: halfHeight };
  if (target === 'bottom-left') return { x: 0, y: bounds.height - halfHeight, width: halfWidth, height: halfHeight };
  if (target === 'bottom-right') return { x: maxWidth - halfWidth, y: bounds.height - halfHeight, width: halfWidth, height: halfHeight };
  return { x: 0, y: 0, width: maxWidth, height: bounds.height };
};

const detectSnapTarget = (win, bounds) => {
  const EDGE_BUFFER = 40;  // conservative buffer to prevent overflow
  const maxWidth = bounds.width - EDGE_BUFFER;
  const nearTop = win.y <= SNAP_THRESHOLD;
  const nearBottom = win.y + win.height >= bounds.height - SNAP_THRESHOLD;
  const nearLeft = win.x <= SNAP_THRESHOLD;
  const nearRight = win.x + win.width >= maxWidth - SNAP_THRESHOLD;

  if (nearTop && nearLeft) return 'top-left';
  if (nearTop && nearRight) return 'top-right';
  if (nearBottom && nearLeft) return 'bottom-left';
  if (nearBottom && nearRight) return 'bottom-right';
  if (nearTop) return 'fullscreen';
  if (nearLeft) return 'left';
  if (nearRight) return 'right';
  return '';
};

const buildDefaultWindowLayout = () => {
  const { width, height } = getDesktopBounds();
  const colGap = 14;
  const rowGap = 14;
  const padH = 32;
  const padV = 32;
  const maxWidth = Math.max(600, width - padH);
  const maxHeight = Math.max(400, height - padV);

  const leftWidth = Math.max(260, Math.floor(maxWidth * 0.25));
  const rightWidth = Math.max(260, Math.floor(maxWidth * 0.22));
  const centerWidth = Math.max(300, maxWidth - leftWidth - rightWidth - colGap * 2);

  const topHeight = Math.max(260, Math.floor(maxHeight * 0.62));
  const bottomHeight = Math.max(180, maxHeight - topHeight - rowGap);

  const xLeft = 0;
  const xCenter = leftWidth + colGap;
  const xRight = Math.max(xCenter + centerWidth + colGap, maxWidth - rightWidth);
  const yTop = 0;
  const yBottom = topHeight + rowGap;

  const renderWidth = Math.min(centerWidth + colGap + rightWidth, maxWidth - xCenter);

  return {
    preview: { x: xLeft, y: yTop, width: leftWidth, height: topHeight, z: 21 },
    render: { x: xCenter, y: yTop, width: renderWidth, height: topHeight, z: 22 },
    controls: { x: xLeft, y: yBottom, width: leftWidth, height: bottomHeight, z: 23 },
    panels: { x: xCenter, y: yBottom, width: centerWidth, height: bottomHeight, z: 24 },
    install: { x: xRight, y: yBottom, width: rightWidth, height: bottomHeight, z: 25 },
  };
};

const clampWindowRect = (rect) => {
  const bounds = getDesktopBounds();
  const EDGE_BUFFER = 40;  // conservative buffer to prevent overflow
  const maxWidth = bounds.width - EDGE_BUFFER;
  const width = Math.min(Math.max(260, rect.width), maxWidth);
  const height = Math.min(Math.max(180, rect.height), bounds.height);
  const x = Math.min(Math.max(0, rect.x), Math.max(0, maxWidth - width));
  const y = Math.min(Math.max(0, rect.y), Math.max(0, bounds.height - height));
  return { x, y, width, height };
};

const persistLayout = () => {
  const payload = {};
  Object.entries(state.windows).forEach(([id, win]) => {
    payload[id] = {
      x: win.x,
      y: win.y,
      width: win.width,
      height: win.height,
      z: win.z,
      minimized: !!win.minimized,
      closed: !!win.closed,
      maximized: !!win.maximized,
      snap: win.snap || '',
      prevRect: win.prevRect || null,
    };
  });
  localStorage.setItem(WINDOW_LAYOUT_KEY, JSON.stringify(payload));
};

const loadPersistedLayout = () => {
  const defaults = buildDefaultWindowLayout();
  try {
    const raw = localStorage.getItem(WINDOW_LAYOUT_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw);
    const merged = { ...defaults };
    Object.keys(defaults).forEach((id) => {
      const fromStorage = parsed[id] || {};
      const base = defaults[id];
      merged[id] = {
        x: toFiniteNumber(fromStorage.x) ?? base.x,
        y: toFiniteNumber(fromStorage.y) ?? base.y,
        width: toFiniteNumber(fromStorage.width) ?? base.width,
        height: toFiniteNumber(fromStorage.height) ?? base.height,
        z: toFiniteNumber(fromStorage.z) ?? base.z,
        minimized: !!fromStorage.minimized,
        closed: !!fromStorage.closed,
        maximized: !!fromStorage.maximized,
        snap: typeof fromStorage.snap === 'string' ? fromStorage.snap : '',
        prevRect: fromStorage.prevRect && typeof fromStorage.prevRect === 'object' ? fromStorage.prevRect : null,
      };
    });
    return merged;
  } catch {
    return defaults;
  }
};

const setWindowRect = (element, rect) => {
  element.style.width = `${rect.width}px`;
  element.style.height = `${rect.height}px`;
  element.dataset.x = String(rect.x);
  element.dataset.y = String(rect.y);
  element.style.transform = `translate(${rect.x}px, ${rect.y}px)`;
};

const renderDock = () => {
  windowDock.innerHTML = '';
  Object.keys(WINDOW_TITLES).forEach((id) => {
    const win = state.windows[id];
    const button = document.createElement('button');
    button.className = 'dock-item';
    if (win.closed) button.classList.add('is-closed');
    if (win.minimized) button.classList.add('is-minimized');
    button.textContent = WINDOW_TITLES[id];
    button.addEventListener('click', () => {
      if (win.closed) {
        openWindow(id);
      } else if (win.minimized) {
        restoreWindow(id);
      } else {
        focusWindow(id);
      }
    });
    windowDock.appendChild(button);
  });
};

const applyWindowVisualState = (id) => {
  const element = desktop.querySelector(`[data-window-id="${id}"]`);
  const win = state.windows[id];
  if (!element || !win) return;

  element.style.zIndex = String(win.z);
  element.classList.toggle('is-closed', !!win.closed);
  element.classList.toggle('is-minimized', !!win.minimized);
  element.classList.toggle('is-maximized', !!win.maximized);

  if (isMobileMode()) {
    element.style.width = '';
    element.style.height = '';
    element.style.transform = '';
    element.style.zIndex = '';
    // Force all windows visible on mobile
    element.classList.remove('is-closed', 'is-minimized', 'is-maximized');
    return;
  }

  if (isTiledDesktopMode()) {
    element.style.width = '';
    element.style.height = '';
    element.style.transform = '';
    element.style.zIndex = '';
    element.classList.toggle('is-maximized', false);
    return;
  }

  const rect = clampWindowRect(win);
  win.x = rect.x;
  win.y = rect.y;
  win.width = rect.width;
  win.height = rect.height;
  setWindowRect(element, rect);
};

const focusWindow = (id) => {
  const win = state.windows[id];
  if (!win || win.closed || win.minimized) return;

  if (isTiledDesktopMode()) {
    windowElements.forEach((el) => el.classList.toggle('is-active', el.dataset.windowId === id));
    applyWindowVisualState(id);
    renderDock();
    persistLayout();
    return;
  }

  state.zSeed += 1;
  win.z = state.zSeed;
  windowElements.forEach((el) => el.classList.toggle('is-active', el.dataset.windowId === id));
  applyWindowVisualState(id);
  renderDock();
  persistLayout();
};

const restoreWindow = (id) => {
  const win = state.windows[id];
  if (!win) return;
  win.closed = false;
  win.minimized = false;
  focusWindow(id);
  applyWindowVisualState(id);
  renderDock();
  persistLayout();
};

const closeWindow = (id) => {
  if (isMobileMode()) return;
  const win = state.windows[id];
  if (!win) return;
  win.closed = true;
  win.minimized = false;
  const element = desktop.querySelector(`[data-window-id="${id}"]`);
  if (element?.classList.contains('is-expanded')) {
    element.classList.remove('is-expanded');
    desktop.classList.remove('has-expanded');
  }
  applyWindowVisualState(id);
  renderDock();
  persistLayout();
};

const minimizeWindow = (id) => {
  if (isMobileMode()) return;
  const win = state.windows[id];
  if (!win || win.closed) return;
  win.minimized = true;
  win.maximized = false;
  win.snap = '';
  const element = desktop.querySelector(`[data-window-id="${id}"]`);
  if (element?.classList.contains('is-expanded')) {
    element.classList.remove('is-expanded');
    desktop.classList.remove('has-expanded');
  }
  applyWindowVisualState(id);
  renderDock();
  persistLayout();
};

const maximizeWindow = (id) => {
  if (isMobileMode()) return;
  const win = state.windows[id];
  if (!win || win.closed) return;

  if (isTiledDesktopMode()) {
    const element = desktop.querySelector(`[data-window-id="${id}"]`);
    if (!element) return;
    const alreadyExpanded = element.classList.contains('is-expanded');
    windowElements.forEach((el) => el.classList.remove('is-expanded'));
    if (alreadyExpanded) {
      desktop.classList.remove('has-expanded');
    } else {
      element.classList.add('is-expanded');
      desktop.classList.add('has-expanded');
      focusWindow(id);
    }
    renderDock();
    persistLayout();
    return;
  }

  const bounds = getDesktopBounds();

  if (!win.maximized) {
    win.prevRect = { x: win.x, y: win.y, width: win.width, height: win.height };
    win.x = 0;
    win.y = 0;
    win.width = bounds.width;
    win.height = bounds.height;
    win.maximized = true;
    win.snap = 'fullscreen';
  } else {
    const fallback = buildDefaultWindowLayout()[id];
    const prev = win.prevRect || fallback;
    win.x = prev.x;
    win.y = prev.y;
    win.width = prev.width;
    win.height = prev.height;
    win.maximized = false;
    win.snap = '';
  }

  restoreWindow(id);
  applyWindowVisualState(id);
  persistLayout();
};

const openWindow = (id) => {
  const win = state.windows[id];
  if (!win) return;
  win.closed = false;
  win.minimized = false;
  if (!win.width || !win.height) {
    const defaults = buildDefaultWindowLayout();
    Object.assign(win, defaults[id]);
  }
  focusWindow(id);
  applyWindowVisualState(id);
  persistLayout();
};

const snapWindow = (id, side) => {
  const win = state.windows[id];
  if (!win || isMobileMode()) return;
  const bounds = getDesktopBounds();

  if (!win.prevRect || win.maximized) {
    win.prevRect = { x: win.x, y: win.y, width: win.width, height: win.height };
  }

  const rect = getSnapRect(side, bounds);
  win.x = rect.x;
  win.y = rect.y;
  win.width = rect.width;
  win.height = rect.height;
  win.maximized = side === 'fullscreen';
  win.snap = side;

  restoreWindow(id);
  applyWindowVisualState(id);
  persistLayout();
};

const normalizeWindowStates = () => {
  const ids = Object.keys(state.windows);
  ids.forEach((id) => applyWindowVisualState(id));

  if (isTiledDesktopMode()) {
    const open = ids.filter((id) => !state.windows[id].closed && !state.windows[id].minimized);
    const active = windowElements.find((el) => el.classList.contains('is-active'))?.dataset.windowId;
    if (!active || !open.includes(active)) {
      const nextId = open[0];
      windowElements.forEach((el) => el.classList.toggle('is-active', el.dataset.windowId === nextId));
    }
    renderDock();
    persistLayout();
    return;
  }

  const open = ids.filter((id) => !state.windows[id].closed && !state.windows[id].minimized);
  if (open.length) {
    const topId = open.reduce((best, current) => {
      if (!best) return current;
      return state.windows[current].z > state.windows[best].z ? current : best;
    }, open[0]);
    focusWindow(topId);
  }
  renderDock();
  persistLayout();
};

const setupWindowControls = () => {
  windowElements.forEach((element) => {
    const id = element.dataset.windowId;
    element.addEventListener('pointerdown', () => focusWindow(id));

    element.querySelectorAll('[data-window-action]').forEach((button) => {
      button.addEventListener('click', () => {
        const action = button.dataset.windowAction;
        if (action === 'close') closeWindow(id);
        if (action === 'minimize') minimizeWindow(id);
        if (action === 'maximize') maximizeWindow(id);
      });
    });
  });
};

const setupInteract = () => {
  if (isTiledDesktopMode()) {
    return;
  }

  if (typeof window.interact !== 'function') {
    return;
  }

  window.interact('.os-window').draggable({
    allowFrom: '.window-titlebar',
    ignoreFrom: '.window-controls, .window-controls *, select, button, input, textarea, a',
    listeners: {
      start(event) {
        const id = event.target.dataset.windowId;
        focusWindow(id);
        hideSnapOverlay();
      },
      move(event) {
        if (isMobileMode()) return;
        const id = event.target.dataset.windowId;
        const win = state.windows[id];
        if (!win || win.closed || win.minimized || win.maximized) return;

        win.x += event.dx;
        win.y += event.dy;
        const clamped = clampWindowRect(win);
        win.x = clamped.x;
        win.y = clamped.y;
        setWindowRect(event.target, clamped);

        const bounds = getDesktopBounds();
        const target = detectSnapTarget(win, bounds);
        if (target) {
          showSnapOverlay(getSnapRect(target, bounds));
        } else {
          hideSnapOverlay();
        }
      },
      end(event) {
        if (isMobileMode()) return;
        const id = event.target.dataset.windowId;
        const win = state.windows[id];
        if (!win || win.closed || win.minimized) return;

        const bounds = getDesktopBounds();
        const target = detectSnapTarget(win, bounds);
        hideSnapOverlay();

        if (target) {
          snapWindow(id, target);
          return;
        }

        win.snap = '';
        persistLayout();
      },
    },
  });

  window.interact('.os-window').resizable({
    edges: { left: true, right: true, bottom: true, top: true },
    listeners: {
      move(event) {
        if (isMobileMode()) return;
        const id = event.target.dataset.windowId;
        const win = state.windows[id];
        if (!win || win.closed || win.minimized || win.maximized) return;

        win.width = event.rect.width;
        win.height = event.rect.height;
        win.x += event.deltaRect.left;
        win.y += event.deltaRect.top;

        const clamped = clampWindowRect(win);
        win.x = clamped.x;
        win.y = clamped.y;
        win.width = clamped.width;
        win.height = clamped.height;
        win.snap = '';
        setWindowRect(event.target, clamped);
        hideSnapOverlay();
      },
      end() {
        hideSnapOverlay();
        persistLayout();
      },
    },
    modifiers: [
      window.interact.modifiers.restrictSize({ min: { width: 260, height: 180 } }),
    ],
    inertia: false,
  });
};

const resetWindowLayout = () => {
  localStorage.removeItem(WINDOW_LAYOUT_KEY);
  state.windows = loadPersistedLayout();
  state.zSeed = 30;
  desktop.classList.remove('has-expanded');
  windowElements.forEach((el) => el.classList.remove('is-expanded'));
  if (isTiledDesktopMode()) {
    Object.values(state.windows).forEach((win) => {
      win.closed = false;
      win.minimized = false;
      win.maximized = false;
      win.snap = '';
    });
  }
  normalizeWindowStates();
};

const initWindowManager = () => {
  state.windows = loadPersistedLayout();
  state.zSeed = 30;
  setupWindowControls();
  setupInteract();
  normalizeWindowStates();
};

const main = async () => {
  initTheme();
  initWindowManager();

  const params = new URLSearchParams(window.location.search);
  const viserPort = params.get('viser_port') || '8765';
  state.metricsFile = normalizeWebPath(params.get('metrics_file'), '/artifacts/metrics/latest_preview.json');
  state.artifactsRoot = normalizeWebPath(params.get('artifacts_root'), '/artifacts');
  const host = window.location.hostname || '127.0.0.1';
  viserFrame.src = `http://${host}:${viserPort}`;

  const [metricsResult, catalogResult, scenesResult, experimentsResult] = await Promise.allSettled([
    loadMetrics(),
    loadCatalog(),
    loadScenes(),
    loadExperiments(state.experimentsFilters),
  ]);
  const metrics = metricsResult.status === 'fulfilled' ? metricsResult.value : {};
  const catalog = catalogResult.status === 'fulfilled' ? catalogResult.value : { datasets: [], methods: [], notes: [] };
  const scenes = scenesResult.status === 'fulfilled' ? scenesResult.value : [];
  const experiments = experimentsResult.status === 'fulfilled' ? experimentsResult.value : [];

  if (metricsResult.status === 'rejected' && !state.metricsError) {
    state.metricsError = 'Nao foi possivel carregar metricas.';
  }

  state.metrics = Object.fromEntries(
    Object.entries(metrics || {}).map(([method, payload]) => [method, normalizeMetricEntry(method, payload)])
  );
  state.catalog = catalog;
  state.methodLabels = buildMethodLabelMap(catalog);
  state.experiments = experiments;

  const installedMethods = (catalog.methods || [])
    .filter((item) => item.installed)
    .map((item) => item.item_id);
  state.installedMethods = installedMethods;

  const metricMethods = Object.keys(metrics || {});
  const selectableMethods = Array.from(new Set([...metricMethods, ...installedMethods]));

  metricMethods.forEach((methodId) => {
    if (!state.methodLabels.has(methodId)) {
      state.methodLabels.set(methodId, methodId);
    }
  });

  populateSelect(primarySelect, selectableMethods, 'Nenhum modelo instalado');
  populateSelect(secondarySelect, selectableMethods, 'Nenhum modelo instalado');
  populateSelect(renderMethodSelect, selectableMethods, 'Nenhum modelo instalado');
  populateSelect(panelMethodSelect, selectableMethods, 'Nenhum modelo instalado');

  if (selectableMethods.length) {
    primarySelect.value = selectableMethods[0];
    if (renderMethodSelect) {
      renderMethodSelect.value = selectableMethods[0];
    }
    secondarySelect.value = selectableMethods[1] || selectableMethods[0];
  }

  const hasMethods = selectableMethods.length > 0;
  if (loadSceneBtn) loadSceneBtn.disabled = false;
  if (liveRenderBtn) liveRenderBtn.disabled = !hasMethods;
  if (generateReportBtn) generateReportBtn.disabled = false;

  const metricEntries = buildPanels(metrics);
  renderMetricsSummary(metricEntries, state.metricsError);

  if (selectableMethods.length) {
    resetPanels(selectableMethods.slice(0, 4));
  } else {
    renderPanelGrid();
  }

  updateRender();
  updateActiveMetrics();
  renderInstallCards(catalog);
  populateSceneControls(scenes);
  renderExperimentHistory();
  populateHistoryMethodFilter();
  if (compareSelectedBtn) {
    compareSelectedBtn.disabled = state.selectedRunIds.size < 2;
  }
};

panelCollapse?.addEventListener('click', () => {
  const isHidden = panelBody.classList.toggle('hidden');
  panelCollapse.textContent = isHidden ? '+' : '-';
});

primarySelect.addEventListener('change', () => {
  if (renderMethodSelect) {
    renderMethodSelect.value = primarySelect.value;
  }
  updateRender();
  updateActiveMetrics();
});

renderMethodSelect?.addEventListener('change', () => {
  if (primarySelect.value !== renderMethodSelect.value) {
    primarySelect.value = renderMethodSelect.value;
  }
  updateRender();
  updateActiveMetrics();
});

secondarySelect.addEventListener('change', () => {
  updateActiveMetrics();
});

addPanelBtn.addEventListener('click', () => {
  const methodId = panelMethodSelect.value;
  if (!methodId) return;
  if (!state.panels.includes(methodId)) {
    state.panels.push(methodId);
    renderPanelGrid();
  }
});

resetPanelsBtn.addEventListener('click', () => {
  const defaults = state.installedMethods.length ? state.installedMethods.slice(0, 4) : [];
  resetPanels(defaults);
});

resetLayoutBtn?.addEventListener('click', () => {
  resetWindowLayout();
});

loadSceneBtn?.addEventListener('click', () => {
  loadSelectedSceneFrame();
});

liveRenderBtn?.addEventListener('click', () => {
  if (!viserFrame?.src) return;
  window.open(viserFrame.src, '_blank', 'noopener,noreferrer');
});

generateReportBtn?.addEventListener('click', () => {
  window.open('/artifacts/reports/benchmark_report.html', '_blank', 'noopener,noreferrer');
});

refreshHistoryBtn?.addEventListener('click', async () => {
  const items = await loadExperiments(state.experimentsFilters);
  state.experiments = items;
  renderExperimentHistory();
  populateHistoryMethodFilter();
});

compareSelectedBtn?.addEventListener('click', async () => {
  await requestExperimentsComparison();
});

historyFilterMethod?.addEventListener('change', async () => {
  state.experimentsFilters.method = historyFilterMethod.value || '';
  const items = await loadExperiments(state.experimentsFilters);
  state.experiments = items;
  state.selectedRunIds.clear();
  renderExperimentHistory();
});

historyFilterStatus?.addEventListener('change', async () => {
  state.experimentsFilters.status = historyFilterStatus.value || '';
  const items = await loadExperiments(state.experimentsFilters);
  state.experiments = items;
  state.selectedRunIds.clear();
  renderExperimentHistory();
});

historyFilterDataset?.addEventListener('change', async () => {
  state.experimentsFilters.dataset = (historyFilterDataset.value || '').trim();
  const items = await loadExperiments(state.experimentsFilters);
  state.experiments = items;
  state.selectedRunIds.clear();
  renderExperimentHistory();
});

themeToggle.addEventListener('change', (event) => {
  const value = event.target.value;
  if (value === 'auto') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  } else {
    setTheme(value);
  }
});

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (event) => {
  if (themeToggle.value === 'auto') {
    setTheme(event.matches ? 'dark' : 'light');
  }
});

window.addEventListener('resize', () => {
  hideSnapOverlay();
  normalizeWindowStates();
});

sceneDatasetSelect?.addEventListener('change', () => {
  state.selectedSceneId = sceneDatasetSelect.value;
  refreshSceneSplitOptions();
});

sceneSplitSelect?.addEventListener('change', () => {
  refreshSceneFrameOptions();
});

main();
