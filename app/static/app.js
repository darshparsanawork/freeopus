const $ = (sel) => document.querySelector(sel);

const dashboardSection = $("#dashboardSection");
const detailSection = $("#detailSection");
const detailProgress = $("#detailProgress");
const detailError = $("#detailError");
const detailSelection = $("#detailSelection");
const detailResults = $("#detailResults");

const LAST_JOB_KEY = "openshorts_last_job_id";
let selectedJobId = localStorage.getItem(LAST_JOB_KEY) || null;
let listPollTimer = null;
let detailPollTimer = null;
let previewTimeUpdateHandler = null;

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || "Request failed");
  }
  return res.json();
}

// ---- URL submission ----
$("#urlForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const url = $("#urlInput").value.trim();
  $("#urlError").classList.add("hidden");
  try {
    const job = await api("/api/jobs", { method: "POST", body: JSON.stringify({ url }) });
    $("#urlInput").value = "";
    refreshJobsList();
    openDetail(job.id);
  } catch (err) {
    $("#urlError").textContent = err.message;
    $("#urlError").classList.remove("hidden");
  }
});

// ---- Persistent job list (the "dashboard") ----
const STAGE_BADGES = {
  queued: { label: "Queued", cls: "badge-queued" },
  downloading: { label: "Downloading", cls: "badge-working" },
  transcribing: { label: "Transcribing", cls: "badge-working" },
  detecting_scenes: { label: "Analyzing scenes", cls: "badge-working" },
  analyzing: { label: "Picking moments", cls: "badge-working" },
  awaiting_selection: { label: "Needs your input", cls: "badge-attention" },
  processing: { label: "Rendering clips", cls: "badge-working" },
  done: { label: "Ready to download", cls: "badge-done" },
  error: { label: "Error", cls: "badge-error" },
};

function formatExpiry(expiresInSeconds) {
  if (expiresInSeconds == null) return "";
  const hours = Math.floor(expiresInSeconds / 3600);
  const minutes = Math.floor((expiresInSeconds % 3600) / 60);
  return `deletes in ${hours}h ${minutes}m`;
}

async function refreshJobsList() {
  try {
    const jobList = await api("/api/jobs");
    renderJobsList(jobList);
  } catch (err) {
    // Transient network hiccup - keep the existing list visible, retry on next tick.
  } finally {
    clearTimeout(listPollTimer);
    listPollTimer = setTimeout(refreshJobsList, 4000);
  }
}

function renderJobsList(jobList) {
  const container = $("#jobsList");
  $("#emptyState").classList.toggle("hidden", jobList.length > 0);
  container.innerHTML = "";

  for (const job of jobList) {
    const badge = STAGE_BADGES[job.stage] || { label: job.stage, cls: "badge-working" };
    const card = document.createElement("div");
    card.className = "job-card" + (job.id === selectedJobId ? " job-card-active" : "");
    const doneClips = (job.clips || []).filter((c) => c.status === "done").length;
    const totalClips = (job.clips || []).length;
    let subtext = job.message || "";
    if (job.stage === "processing" && totalClips > 0) {
      subtext = `Rendering clip ${Math.min(doneClips + 1, totalClips)} of ${totalClips}...`;
    } else if (job.stage === "done") {
      subtext = `${totalClips} clip${totalClips === 1 ? "" : "s"} ready`;
    }
    card.innerHTML = `
      <div class="job-card-main">
        <div class="job-card-title">${escapeHtml(job.title || job.url)}</div>
        <div class="job-card-sub muted">${escapeHtml(subtext)}</div>
      </div>
      <div class="job-card-meta">
        <span class="badge ${badge.cls}">${badge.label}</span>
        ${job.expires_in_seconds != null ? `<span class="job-card-expiry muted">${formatExpiry(job.expires_in_seconds)}</span>` : ""}
      </div>
    `;
    if (["downloading", "transcribing", "detecting_scenes", "analyzing", "processing"].includes(job.stage)) {
      const bar = document.createElement("div");
      bar.className = "job-card-progress";
      bar.innerHTML = `<div class="job-card-progress-fill" style="width:${job.progress}%"></div>`;
      card.querySelector(".job-card-main").appendChild(bar);
    }
    card.addEventListener("click", () => openDetail(job.id));
    container.appendChild(card);
  }
}

// ---- Detail view (one job at a time) ----
function openDetail(jobId) {
  selectedJobId = jobId;
  localStorage.setItem(LAST_JOB_KEY, jobId);
  detailSection.classList.remove("hidden");
  detailSection.scrollIntoView({ behavior: "smooth", block: "start" });
  pollDetail();
}

$("#closeDetailBtn").addEventListener("click", () => {
  clearTimeout(detailPollTimer);
  selectedJobId = null;
  localStorage.removeItem(LAST_JOB_KEY);
  detailSection.classList.add("hidden");
  stopPreview();
});

async function pollDetail() {
  clearTimeout(detailPollTimer);
  if (!selectedJobId) return;
  try {
    const job = await api(`/api/jobs/${selectedJobId}`);
    renderDetail(job);
    renderJobsListHighlight();
    const stillWorking =
      job.stage !== "error" &&
      (job.stage !== "done" || (job.clips || []).some((c) => ["pending", "rendering", "captioning"].includes(c.status)));
    if (job.stage === "awaiting_selection") return; // waiting on the user, not the server
    if (stillWorking) {
      detailPollTimer = setTimeout(pollDetail, 1500);
    }
  } catch (err) {
    // Job may have expired/been cleaned up.
    $("#detailErrorMessage").textContent = err.message;
    showDetailBlock(detailError);
  }
}

function renderJobsListHighlight() {
  document.querySelectorAll(".job-card").forEach((el) => el.classList.remove("job-card-active"));
}

function showDetailBlock(block) {
  for (const b of [detailProgress, detailError, detailSelection, detailResults]) {
    b.classList.toggle("hidden", b !== block);
  }
}

const STAGE_LABELS = {
  queued: "Queued — waiting for a free worker slot...",
  downloading: "Downloading video",
  transcribing: "Transcribing audio",
  detecting_scenes: "Detecting scene cuts",
  analyzing: "Picking the best moments",
  processing: "Rendering clips",
};

function renderDetail(job) {
  if (job.stage === "error") {
    $("#detailErrorMessage").textContent = job.error || job.message;
    showDetailBlock(detailError);
    return;
  }

  if (job.stage === "awaiting_selection") {
    showDetailBlock(detailSelection);
    renderMoments(job);
    return;
  }

  if (job.stage === "processing" || job.stage === "done") {
    showDetailBlock(detailResults);
    renderClips(job);
    renderExpiryNotice(job.expires_in_seconds);
    return;
  }

  showDetailBlock(detailProgress);
  $("#progressStage").textContent = STAGE_LABELS[job.stage] || job.stage;
  $("#progressFill").style.width = `${job.progress}%`;
  $("#progressMessage").textContent = job.message;
}

function renderExpiryNotice(expiresInSeconds) {
  const el = $("#expiryNotice");
  if (expiresInSeconds == null) {
    el.textContent = "";
    return;
  }
  const hours = Math.floor(expiresInSeconds / 3600);
  const minutes = Math.floor((expiresInSeconds % 3600) / 60);
  el.textContent = `⏳ These clips and their files are deleted automatically in ${hours}h ${minutes}m — download what you want before then.`;
}

// ---- Moment selection + video preview ----
function renderMoments(job) {
  const list = $("#momentsList");
  list.innerHTML = "";
  job.moments.forEach((m, i) => {
    const div = document.createElement("div");
    div.className = "moment-item";
    div.innerHTML = `
      <input type="checkbox" class="moment-check" value="${m.id}" ${i < 5 ? "checked" : ""} />
      <div class="moment-body">
        <div class="moment-title">${escapeHtml(m.title)}</div>
        <div class="moment-meta">${formatTime(m.start)} – ${formatTime(m.end)} · ${escapeHtml(m.reason)}</div>
      </div>
      <button type="button" class="btn ghost small moment-preview-btn">▶ Preview</button>
      <div class="moment-score">${Math.round(m.score)}</div>
    `;
    div.querySelector(".moment-preview-btn").addEventListener("click", () => previewMoment(job.id, m));
    list.appendChild(div);
  });
}

function previewMoment(jobId, moment) {
  const wrap = $("#previewPlayerWrap");
  const player = $("#previewPlayer");
  wrap.classList.remove("hidden");
  wrap.scrollIntoView({ behavior: "smooth", block: "nearest" });
  $("#previewLabel").textContent = `Previewing "${moment.title}" (${formatTime(moment.start)}–${formatTime(moment.end)})`;

  const src = `/api/jobs/${jobId}/source`;
  const needsNewSrc = !player.src.includes(src);
  if (needsNewSrc) {
    player.src = src;
  }

  if (previewTimeUpdateHandler) {
    player.removeEventListener("timeupdate", previewTimeUpdateHandler);
  }
  previewTimeUpdateHandler = () => {
    if (player.currentTime >= moment.end || player.currentTime < moment.start - 0.5) {
      player.pause();
      player.currentTime = moment.start;
    }
  };
  player.addEventListener("timeupdate", previewTimeUpdateHandler);

  const seekAndPlay = () => {
    player.currentTime = moment.start;
    player.play().catch(() => {});
  };
  if (needsNewSrc) {
    player.addEventListener("loadedmetadata", seekAndPlay, { once: true });
  } else {
    seekAndPlay();
  }
}

function stopPreview() {
  const player = $("#previewPlayer");
  player.pause();
  player.removeAttribute("src");
  player.load();
  $("#previewPlayerWrap").classList.add("hidden");
}

$("#closePreviewBtn").addEventListener("click", stopPreview);

// ---- Caption position picker (live CSS preview, no backend call needed) ----
document.querySelectorAll('input[name="captionPosition"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const bar = $("#reelCaptionBar");
    bar.className = "reel-caption-bar reel-caption-" + radio.value;
  });
});

// ---- Generate clips ----
$("#generateBtn").addEventListener("click", async () => {
  const momentIds = Array.from(document.querySelectorAll(".moment-check:checked")).map((el) => el.value);
  if (momentIds.length === 0) {
    alert("Select at least one moment.");
    return;
  }
  const captionFormats = Array.from(document.querySelectorAll(".capfmt:checked")).map((el) => el.value);
  const captionPosition = document.querySelector('input[name="captionPosition"]:checked').value;
  const payload = {
    moment_ids: momentIds,
    crop_mode: $("#cropMode").value,
    subtitles_enabled: $("#subtitlesEnabled").checked,
    burn_in: $("#burnIn").checked,
    caption_formats: captionFormats,
    caption_position: captionPosition,
  };
  $("#generateBtn").disabled = true;
  try {
    stopPreview();
    await api(`/api/jobs/${selectedJobId}/process`, { method: "POST", body: JSON.stringify(payload) });
    pollDetail();
  } catch (err) {
    alert(err.message);
  } finally {
    $("#generateBtn").disabled = false;
  }
});

// ---- Results ----
const CLIP_STATUS_LABELS = {
  pending: "Queued",
  rendering: "Rendering (smart crop)...",
  captioning: "Adding captions...",
  done: "Ready",
  error: "Failed",
};

function renderClips(job) {
  const grid = $("#clipsGrid");
  grid.innerHTML = "";
  job.clips.forEach((c) => {
    const div = document.createElement("div");
    div.className = "clip-card";
    const videoTag =
      c.status === "done"
        ? `<video controls preload="metadata" src="/api/jobs/${job.id}/clips/${c.id}/preview"></video>`
        : `<div class="clip-placeholder">${CLIP_STATUS_LABELS[c.status] || c.status}</div>`;
    const captionLinks = (c.caption_formats || [])
      .map((f) => `<a class="btn ghost small" href="/api/jobs/${job.id}/clips/${c.id}/captions/${f}" download>${f.toUpperCase()}</a>`)
      .join("");
    div.innerHTML = `
      ${videoTag}
      <div class="title">${escapeHtml(c.title)}</div>
      <div class="status">${CLIP_STATUS_LABELS[c.status] || c.status}${c.mode_used ? " · " + c.mode_used + " mode" : ""}${c.error ? " · " + escapeHtml(c.error) : ""}</div>
      <div class="actions">
        ${c.status === "done" ? `<a class="btn primary small" href="/api/jobs/${job.id}/clips/${c.id}/download" download>Download MP4</a>` : ""}
        ${captionLinks}
      </div>
    `;
    grid.appendChild(div);
  });
}

function formatTime(t) {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

// ---- Settings ----
const settingsModal = $("#settingsModal");

$("#settingsBtn").addEventListener("click", async () => {
  settingsModal.classList.remove("hidden");
  try {
    const s = await api("/api/settings");
    $("#llmProvider").value = s.llm_provider;
    $("#openrouterModel").innerHTML = `<option value="${s.openrouter_model}">${s.openrouter_model}</option>`;
    $("#geminiModel").value = s.gemini_model;
    $("#openrouterKey").placeholder = s.has_openrouter_key ? "•••••••• (saved, leave blank to keep)" : "sk-or-...";
    $("#geminiKey").placeholder = s.has_gemini_key ? "•••••••• (saved, leave blank to keep)" : "AIza...";
    toggleProviderFields();
  } catch (err) {
    $("#settingsStatus").textContent = err.message;
  }
});

$("#closeSettingsBtn").addEventListener("click", () => settingsModal.classList.add("hidden"));

$("#saveCookiesBtn").addEventListener("click", async () => {
  const text = $("#cookiesTxt").value.trim();
  if (!text) return;
  $("#cookiesStatus").textContent = "Saving and checking against YouTube...";
  $("#cookiesStatus").className = "muted";
  try {
    const result = await api("/api/youtube-cookies", { method: "POST", body: JSON.stringify({ cookies_txt: text }) });
    setCookiesStatus(result.valid, result.message);
    $("#cookiesTxt").value = "";
  } catch (err) {
    setCookiesStatus(false, err.message);
  }
});

$("#revalidateCookiesBtn").addEventListener("click", async () => {
  $("#cookiesStatus").textContent = "Checking against YouTube...";
  $("#cookiesStatus").className = "muted";
  try {
    const result = await api("/api/youtube-cookies/validate", { method: "POST" });
    setCookiesStatus(result.valid, result.message);
  } catch (err) {
    setCookiesStatus(false, err.message);
  }
});

function setCookiesStatus(valid, message) {
  const el = $("#cookiesStatus");
  el.textContent = (valid ? "✅ " : "❌ ") + message;
  el.className = valid ? "success" : "error";
}
$("#llmProvider").addEventListener("change", toggleProviderFields);

function toggleProviderFields() {
  const provider = $("#llmProvider").value;
  $("#openrouterFields").classList.toggle("hidden", provider !== "openrouter");
  $("#geminiFields").classList.toggle("hidden", provider !== "gemini");
}

$("#loadModelsBtn").addEventListener("click", async () => {
  const key = $("#openrouterKey").value.trim();
  if (key) {
    await api("/api/settings", { method: "POST", body: JSON.stringify({ openrouter_api_key: key }) });
  }
  try {
    $("#settingsStatus").textContent = "Loading models...";
    const models = await api("/api/openrouter/models");
    const select = $("#openrouterModel");
    select.innerHTML = models.map((m) => `<option value="${m.id}">${escapeHtml(m.name)}</option>`).join("");
    $("#settingsStatus").textContent = `Loaded ${models.length} models.`;
  } catch (err) {
    $("#settingsStatus").textContent = err.message;
  }
});

$("#saveSettingsBtn").addEventListener("click", async () => {
  const payload = {
    llm_provider: $("#llmProvider").value,
    openrouter_model: $("#openrouterModel").value,
    gemini_model: $("#geminiModel").value,
  };
  const orKey = $("#openrouterKey").value.trim();
  const gKey = $("#geminiKey").value.trim();
  if (orKey) payload.openrouter_api_key = orKey;
  if (gKey) payload.gemini_api_key = gKey;

  try {
    await api("/api/settings", { method: "POST", body: JSON.stringify(payload) });
    $("#settingsStatus").textContent = "Saved.";
    $("#openrouterKey").value = "";
    $("#geminiKey").value = "";
  } catch (err) {
    $("#settingsStatus").textContent = err.message;
  }
});

// ---- Boot ----
refreshJobsList();
if (selectedJobId) {
  detailSection.classList.remove("hidden");
  pollDetail();
}
