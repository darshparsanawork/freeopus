const $ = (sel) => document.querySelector(sel);

const urlSection = $("#urlSection");
const progressSection = $("#progressSection");
const selectionSection = $("#selectionSection");
const resultsSection = $("#resultsSection");

let currentJobId = null;
let pollTimer = null;

function showOnly(section) {
  for (const s of [urlSection, progressSection, selectionSection, resultsSection]) {
    s.classList.add("hidden");
  }
  section.classList.remove("hidden");
}

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
    currentJobId = job.id;
    showOnly(progressSection);
    pollJob();
  } catch (err) {
    $("#urlError").textContent = err.message;
    $("#urlError").classList.remove("hidden");
  }
});

const STAGE_LABELS = {
  queued: "Queued...",
  downloading: "Downloading video",
  transcribing: "Transcribing audio",
  detecting_scenes: "Detecting scene cuts",
  analyzing: "Picking the best moments",
  awaiting_selection: "Ready for your review",
  processing: "Rendering clips",
  done: "Done",
  error: "Something went wrong",
};

async function pollJob() {
  clearTimeout(pollTimer);
  try {
    const job = await api(`/api/jobs/${currentJobId}`);
    renderJob(job);
    if (job.stage === "error") return;
    if (job.stage !== "done") {
      pollTimer = setTimeout(pollJob, 1500);
    } else if (job.clips.some((c) => c.status === "pending" || c.status === "rendering" || c.status === "captioning")) {
      pollTimer = setTimeout(pollJob, 1500);
    }
  } catch (err) {
    $("#progressMessage").textContent = err.message;
  }
}

function renderJob(job) {
  if (job.stage === "error") {
    showOnly(progressSection);
    $("#progressStage").textContent = "Something went wrong";
    $("#progressMessage").textContent = job.error || job.message;
    return;
  }

  if (job.stage === "awaiting_selection") {
    showOnly(selectionSection);
    renderMoments(job.moments);
    return;
  }

  if (job.stage === "processing" || job.stage === "done") {
    showOnly(resultsSection);
    renderClips(job.clips);
    renderExpiryNotice(job.expires_in_seconds);
    return;
  }

  showOnly(progressSection);
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

function renderMoments(moments) {
  const list = $("#momentsList");
  list.innerHTML = "";
  moments.forEach((m, i) => {
    const div = document.createElement("div");
    div.className = "moment-item";
    div.innerHTML = `
      <input type="checkbox" class="moment-check" value="${m.id}" ${i < 5 ? "checked" : ""} />
      <div>
        <div class="moment-title">${escapeHtml(m.title)}</div>
        <div class="moment-meta">${formatTime(m.start)} – ${formatTime(m.end)} · ${escapeHtml(m.reason)}</div>
      </div>
      <div class="moment-score">${Math.round(m.score)}</div>
    `;
    list.appendChild(div);
  });
}

$("#generateBtn").addEventListener("click", async () => {
  const momentIds = Array.from(document.querySelectorAll(".moment-check:checked")).map((el) => el.value);
  if (momentIds.length === 0) {
    alert("Select at least one moment.");
    return;
  }
  const captionFormats = Array.from(document.querySelectorAll(".capfmt:checked")).map((el) => el.value);
  const payload = {
    moment_ids: momentIds,
    crop_mode: $("#cropMode").value,
    subtitles_enabled: $("#subtitlesEnabled").checked,
    burn_in: $("#burnIn").checked,
    caption_formats: captionFormats,
  };
  $("#generateBtn").disabled = true;
  try {
    await api(`/api/jobs/${currentJobId}/process`, { method: "POST", body: JSON.stringify(payload) });
    showOnly(resultsSection);
    pollJob();
  } catch (err) {
    alert(err.message);
  } finally {
    $("#generateBtn").disabled = false;
  }
});

const CLIP_STATUS_LABELS = {
  pending: "Queued",
  rendering: "Rendering (face-tracking crop)...",
  captioning: "Adding captions...",
  done: "Ready",
  error: "Failed",
};

function renderClips(clips) {
  const grid = $("#clipsGrid");
  grid.innerHTML = "";
  clips.forEach((c) => {
    const div = document.createElement("div");
    div.className = "clip-card";
    const videoTag =
      c.status === "done"
        ? `<video controls src="/api/jobs/${currentJobId}/clips/${c.id}/download"></video>`
        : `<div style="aspect-ratio:9/16;background:#000;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#666;">${CLIP_STATUS_LABELS[c.status] || c.status}</div>`;
    const captionLinks = (c.caption_formats || [])
      .map((f) => `<a class="btn ghost small" href="/api/jobs/${currentJobId}/clips/${c.id}/captions/${f}" download>${f.toUpperCase()}</a>`)
      .join("");
    div.innerHTML = `
      ${videoTag}
      <div class="title">${escapeHtml(c.title)}</div>
      <div class="status">${CLIP_STATUS_LABELS[c.status] || c.status}${c.mode_used ? " · " + c.mode_used + " mode" : ""}${c.error ? " · " + escapeHtml(c.error) : ""}</div>
      <div class="actions">
        ${c.status === "done" ? `<a class="btn primary small" href="/api/jobs/${currentJobId}/clips/${c.id}/download" download>Download MP4</a>` : ""}
        ${captionLinks}
      </div>
    `;
    grid.appendChild(div);
  });
}

$("#startOverBtn").addEventListener("click", () => {
  clearTimeout(pollTimer);
  currentJobId = null;
  $("#urlInput").value = "";
  showOnly(urlSection);
});

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
    $("#whisperSize").value = s.whisper_model_size;
    $("#device").value = s.device;
    $("#transcriptionProvider").value = s.transcription_provider;
    $("#openrouterKey").placeholder = s.has_openrouter_key ? "•••••••• (saved, leave blank to keep)" : "sk-or-...";
    $("#geminiKey").placeholder = s.has_gemini_key ? "•••••••• (saved, leave blank to keep)" : "AIza...";
    toggleProviderFields();
    toggleTranscriptionFields();
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

$("#transcriptionProvider").addEventListener("change", toggleTranscriptionFields);

function toggleTranscriptionFields() {
  const isLocal = $("#transcriptionProvider").value === "local";
  $("#whisperSizeLabel").classList.toggle("hidden", !isLocal);
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
    whisper_model_size: $("#whisperSize").value,
    device: $("#device").value,
    transcription_provider: $("#transcriptionProvider").value,
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

showOnly(urlSection);
