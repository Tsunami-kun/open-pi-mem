const state = {
  report: null,
  activeIndex: 0,
  baseDir: "",
  clipAnimationTimer: null,
  siteRootUrl: new URL("./", window.location.href).toString(),
  importedFromFile: false,
  demos: [],
};

const els = {
  reportFile: document.getElementById("report-file"),
  reportUrl: document.getElementById("report-url"),
  loadReportBtn: document.getElementById("load-report-btn"),
  loadSampleBtn: document.getElementById("load-sample-btn"),
  loadStatus: document.getElementById("load-status"),
  demoList: document.getElementById("demo-list"),
  goal: document.getElementById("goal"),
  modelPath: document.getElementById("model-path"),
  source: document.getElementById("source"),
  plannerHz: document.getElementById("planner-hz"),
  frameList: document.getElementById("frame-list"),
  frameMeta: document.getElementById("frame-meta"),
  clipPreview: document.getElementById("clip-preview"),
  videoMeta: document.getElementById("video-meta"),
  inputFrameStrip: document.getElementById("input-frame-strip"),
  detailGoal: document.getElementById("detail-goal"),
  inputMemory: document.getElementById("input-memory"),
  nextSubtask: document.getElementById("next-subtask"),
  nextMemory: document.getElementById("next-memory"),
  prompt: document.getElementById("prompt"),
  rawOutput: document.getElementById("raw-output"),
  prevBtn: document.getElementById("prev-btn"),
  nextBtn: document.getElementById("next-btn"),
};

function init() {
  els.reportFile.addEventListener("change", handleImportFile);
  els.loadReportBtn.addEventListener("click", () => {
    void loadReportFromInput();
  });
  els.loadSampleBtn.addEventListener("click", () => {
    void loadBundledSample();
  });
  els.prevBtn.addEventListener("click", () => setActiveIndex(state.activeIndex - 1));
  els.nextBtn.addEventListener("click", () => setActiveIndex(state.activeIndex + 1));
  els.reportUrl.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void loadReportFromInput();
    }
  });
  void boot();
}

async function boot() {
  await loadManifest();
  const queryReport = getQueryReport();
  if (queryReport) {
    els.reportUrl.value = queryReport;
    await loadReportFromReference(queryReport, { updateQuery: false });
    return;
  }
  if (state.demos.length) {
    await loadReportFromReference(state.demos[0].path, { updateQuery: true });
    return;
  }
  setStatus("No bundled reports were found. Import a report JSON or provide a report URL.");
}

function getQueryReport() {
  const url = new URL(window.location.href);
  return url.searchParams.get("report") || "";
}

function setStatus(message, tone = "info") {
  els.loadStatus.textContent = message;
  els.loadStatus.dataset.tone = tone;
}

async function handleImportFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const report = JSON.parse(text);
  state.importedFromFile = true;
  applyReport(report, "");
  setStatus(
    "Loaded report JSON from your computer. Text fields work immediately; relative image previews require a hosted report URL.",
    "warn",
  );
}

async function loadReportFromInput() {
  const reportRef = els.reportUrl.value.trim();
  if (!reportRef) {
    setStatus("Enter a report path or URL first.", "warn");
    return;
  }
  await loadReportFromReference(reportRef, { updateQuery: true });
}

async function loadBundledSample() {
  if (!state.demos.length) {
    setStatus("No bundled sample reports are available.", "warn");
    return;
  }
  await loadReportFromReference(state.demos[0].path, { updateQuery: true });
}

async function loadReportFromReference(reportRef, { updateQuery = true } = {}) {
  try {
    const reportUrl = resolveReportUrl(reportRef);
    setStatus(`Loading ${reportRef} ...`);
    const response = await fetch(reportUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const report = await response.json();
    state.importedFromFile = false;
    applyReport(report, reportUrl);
    els.reportUrl.value = reportRef;
    if (updateQuery) {
      const url = new URL(window.location.href);
      url.searchParams.set("report", reportRef);
      window.history.replaceState({}, "", url);
    }
    const count = Array.isArray(report.records) ? report.records.length : 0;
    setStatus(`Loaded ${count} records from ${reportRef}.`, "success");
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load report: ${error.message}`, "error");
  }
}

async function loadManifest() {
  const manifestUrl = new URL("reports/index.json", state.siteRootUrl).toString();
  try {
    const response = await fetch(manifestUrl);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const demos = await response.json();
    state.demos = Array.isArray(demos) ? demos : [];
    renderDemoList();
  } catch (error) {
    state.demos = [];
    renderDemoList();
  }
}

function renderDemoList() {
  if (!state.demos.length) {
    els.demoList.className = "demo-list empty-state";
    els.demoList.textContent = "No bundled reports found yet.";
    return;
  }
  els.demoList.className = "demo-list";
  els.demoList.innerHTML = "";
  state.demos.forEach((demo) => {
    const button = document.createElement("button");
    button.className = "demo-item secondary";
    button.innerHTML = `
      <div class="demo-item-title">${escapeHtml(demo.title || demo.task || demo.path)}</div>
      <div class="demo-item-meta">${escapeHtml(demo.model_label || "Bundled demo")}</div>
      <div class="demo-item-meta">${escapeHtml(demo.goal || "")}</div>
    `;
    button.addEventListener("click", () => {
      void loadReportFromReference(demo.path, { updateQuery: true });
    });
    els.demoList.appendChild(button);
  });
}

function applyReport(report, sourceUrl) {
  state.report = report;
  state.activeIndex = 0;
  state.baseDir = sourceUrl ? new URL("./", sourceUrl).toString() : "";
  renderSummary();
  renderFrameList();
  renderActiveFrame();
}

function renderSummary() {
  const report = state.report;
  if (!report) return;
  els.goal.textContent = report.goal || "-";
  els.modelPath.textContent = report.model_path || "-";
  els.source.textContent = report.source || "-";
  els.plannerHz.textContent = `${report.planner_hz ?? "-"} Hz`;
}

function renderFrameList() {
  const report = state.report;
  if (!report?.records?.length) {
    els.frameList.className = "frame-list empty-state";
    els.frameList.textContent = "No frame records loaded yet.";
    return;
  }
  els.frameList.className = "frame-list";
  els.frameList.innerHTML = "";
  report.records.forEach((record, index) => {
    const btn = document.createElement("button");
    btn.className = `frame-item${index === state.activeIndex ? " active" : ""}`;
    const frameLabel = Number.isFinite(record.frame_index) ? String(record.frame_index).padStart(3, "0") : String(index).padStart(3, "0");
    const timestamp = Number.isFinite(record.timestamp_sec) ? `${record.timestamp_sec.toFixed(2)}s` : "-";
    btn.innerHTML = `
      <div class="frame-item-title">Frame ${frameLabel}</div>
      <div class="frame-item-meta">t=${timestamp}</div>
      <div class="frame-item-meta">${escapeHtml(record.next_subtask || "(missing subtask)")}</div>
    `;
    btn.addEventListener("click", () => setActiveIndex(index));
    els.frameList.appendChild(btn);
  });
}

function setActiveIndex(index) {
  const report = state.report;
  if (!report?.records?.length) return;
  state.activeIndex = Math.max(0, Math.min(index, report.records.length - 1));
  renderFrameList();
  renderActiveFrame();
}

function renderActiveFrame() {
  const record = state.report?.records?.[state.activeIndex];
  if (!record) return;
  const timestamp = Number.isFinite(record.timestamp_sec) ? record.timestamp_sec.toFixed(2) : "-";
  els.frameMeta.textContent = `Frame ${record.frame_index ?? state.activeIndex} | t=${timestamp}s`;
  renderInputFrameStrip(record);
  renderClipPreview(record);
  els.detailGoal.textContent = record.goal || state.report?.goal || "(missing)";
  els.inputMemory.textContent = record.input_memory || "(none)";
  els.nextSubtask.textContent = record.next_subtask || "(missing)";
  els.nextMemory.textContent = record.next_memory || "(missing)";
  els.prompt.textContent = record.prompt || "";
  els.rawOutput.textContent = record.raw_output || "";
}

function isAbsoluteUrl(value) {
  return /^(?:[a-z]+:)?\/\//i.test(value);
}

function resolveReportUrl(reportRef) {
  if (isAbsoluteUrl(reportRef)) {
    return reportRef;
  }
  const normalized = reportRef.startsWith("/") ? reportRef.slice(1) : reportRef;
  return new URL(normalized, state.siteRootUrl).toString();
}

function toViewerSourcePath(path) {
  if (!path) return "";
  if (isAbsoluteUrl(path) || path.startsWith("data:") || path.startsWith("blob:")) {
    return path;
  }
  if (state.importedFromFile) {
    return "";
  }
  const normalized = path.startsWith("/") ? path.slice(1) : path;
  if (path.startsWith("/")) {
    return new URL(normalized, state.siteRootUrl).toString();
  }
  if (!state.baseDir) {
    return "";
  }
  return new URL(normalized, state.baseDir).toString();
}

function renderClipPreview(record) {
  if (state.clipAnimationTimer) {
    clearInterval(state.clipAnimationTimer);
    state.clipAnimationTimer = null;
  }

  const denseSamplingHz = Number(state.report?.dense_sampling_hz || 0);
  const frameTimestamps = Array.isArray(record.input_frame_timestamps_sec)
    ? record.input_frame_timestamps_sec.filter((value) => Number.isFinite(value))
    : [];
  const inputPaths = Array.isArray(record.input_image_paths) && record.input_image_paths.length
    ? record.input_image_paths
    : (record.image_path ? [record.image_path] : []);
  const viewerPaths = inputPaths.map((path) => toViewerSourcePath(path)).filter(Boolean);

  if (!viewerPaths.length) {
    els.clipPreview.removeAttribute("src");
    els.videoMeta.textContent = state.importedFromFile
      ? "Frame previews are unavailable in local file mode. Load a hosted report URL to view images."
      : "This record does not contain preview images.";
    return;
  }

  let currentIndex = 0;
  els.clipPreview.src = viewerPaths[0];

  const frameIndices = inputPaths
    .map((path) => {
      const match = String(path).match(/frame_(\d+)\.(?:png|jpg|jpeg)$/i);
      return match ? Number(match[1]) : null;
    })
    .filter((value) => Number.isFinite(value));

  let startTime = Number(record.input_clip_start_sec);
  let endTime = Number(record.input_clip_end_sec);

  if (!Number.isFinite(startTime) || !Number.isFinite(endTime)) {
    if (frameTimestamps.length) {
      startTime = frameTimestamps[0];
      endTime = frameTimestamps[frameTimestamps.length - 1];
    }
  }

  if ((!Number.isFinite(startTime) || !Number.isFinite(endTime)) && denseSamplingHz > 0 && frameIndices.length) {
    startTime = frameIndices[0] / denseSamplingHz;
    endTime = frameIndices[frameIndices.length - 1] / denseSamplingHz;
  }

  if (!Number.isFinite(startTime)) {
    startTime = 0;
  }
  if (!Number.isFinite(endTime)) {
    endTime = Number.isFinite(record.timestamp_sec) ? Number(record.timestamp_sec) : startTime;
  }

  if (viewerPaths.length > 1) {
    const intervalMs = denseSamplingHz > 0 ? Math.max(120, 1000 / denseSamplingHz) : 300;
    state.clipAnimationTimer = window.setInterval(() => {
      currentIndex = (currentIndex + 1) % viewerPaths.length;
      els.clipPreview.src = viewerPaths[currentIndex];
    }, intervalMs);
  }

  els.videoMeta.textContent = `Input clip preview: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s, ${viewerPaths.length} frame(s)`;
}

function renderInputFrameStrip(record) {
  const inputPaths = record.input_image_paths || (record.image_path ? [record.image_path] : []);
  const viewerPaths = inputPaths
    .map((path) => ({
      path,
      resolved: toViewerSourcePath(path),
    }))
    .filter((item) => Boolean(item.resolved));

  if (!viewerPaths.length) {
    els.inputFrameStrip.className = "input-frame-strip empty-state";
    els.inputFrameStrip.textContent = state.importedFromFile
      ? "Input frame thumbnails are unavailable in local file mode. Load a hosted report URL to view them."
      : "This record does not contain multi-frame input.";
    return;
  }
  els.inputFrameStrip.className = "input-frame-strip";
  els.inputFrameStrip.innerHTML = "";
  viewerPaths.forEach((item, index) => {
    const frameName = item.path.split("/").pop() || `frame-${index + 1}`;
    const thumb = document.createElement("div");
    thumb.className = "input-frame-thumb";
    thumb.innerHTML = `
      <img src="${item.resolved}" alt="input frame ${index}" />
      <div class="input-frame-caption">Input ${index + 1} · ${escapeHtml(frameName)}</div>
    `;
    els.inputFrameStrip.appendChild(thumb);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

init();
