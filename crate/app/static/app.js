// Crate frontend — hum, drop/upload, or describe. One embedding space, three ways in.
const $ = (id) => document.getElementById(id);
let sessionId = null;
let pickedFile = null;

const HEART = '<svg class="i" viewBox="0 0 24 24"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.6l-1-1a5.5 5.5 0 1 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8z"/></svg>';
const CROSS = '<svg class="i" viewBox="0 0 24 24"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';

const busy = (on) => document.body.classList.toggle("busy", on);
function setStatus(m) { $("status").textContent = m || ""; }

async function post(url, form) {
  if (sessionId) form.append("session_id", sessionId);
  const r = await fetch(url, { method: "POST", body: form });
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch {}
    setStatus("error: " + msg);
    throw new Error(msg);
  }
  const data = await r.json();
  if (data.session_id) sessionId = data.session_id;
  return data;
}

function render(data) {
  const box = $("results");
  box.innerHTML = "";
  if (data.kit) {
    setStatus("crate assembled — matched samples per stem");
    for (const role of Object.keys(data.kit)) {
      const h = document.createElement("div");
      h.className = "rolehdr " + role;
      h.textContent = role;
      box.appendChild(h);
      appendResults(box, data.kit[role]);
    }
  } else {
    setStatus(`${data.modality} query · ${data.results.length} matches`);
    appendResults(box, data.results);
  }
  // animate match bars from 0 once laid out
  requestAnimationFrame(() =>
    box.querySelectorAll(".bar i").forEach((i) => { i.style.width = i.dataset.w + "%"; }));
}

function appendResults(box, results) {
  if (!results.length) return;
  const max = Math.max(...results.map((r) => r.final ?? r.score ?? 0), 1e-6);
  results.forEach((r, idx) => {
    const el = resultRow(r, max);
    el.style.setProperty("--i", idx);
    box.appendChild(el);
  });
}

function resultRow(r, max) {
  const el = document.createElement("div");
  el.className = "result";
  const raw = r.final ?? r.score ?? 0;
  const pct = Math.max(0, Math.round((raw / max) * 100)); // match relative to the top hit
  el.innerHTML = `
    <div class="match">
      <div class="pct">${pct}%</div>
      <div class="bar"><i data-w="${pct}"></i></div>
    </div>
    <div class="meta">
      <div class="t">${escapeHtml(r.text || r.id)}</div>
      <div class="sub">${escapeHtml(r.attribution || "")}${r.license ? " · " + licenseShort(r.license) : ""}</div>
      <audio controls preload="none" src="/audio/${encodeURIComponent(r.id)}"></audio>
    </div>
    <div class="fb"></div>`;
  const fb = el.querySelector(".fb");
  fb.appendChild(mkBtn("accent", HEART + "Save", () => feedback("/save", r.id, el, "saved")));
  fb.appendChild(mkBtn("", CROSS + "Skip", () => feedback("/skip", r.id, el, "skipped")));
  return el;
}

function mkBtn(cls, html, fn) {
  const b = document.createElement("button");
  if (cls) b.className = cls;
  b.innerHTML = html;
  b.onclick = fn;
  return b;
}

async function feedback(url, id, el, klass) {
  const f = new FormData();
  f.append("result_id", id);
  el.classList.remove("saved", "skipped");
  el.classList.add(klass);
  setStatus(url.endsWith("save")
    ? "saved — trains your taste; future searches rank things you like higher"
    : "skipped — the taste ranker learns to push results like this down");
  await post(url, f);
}

async function run(fn) {
  busy(true);
  try { await fn(); } finally { busy(false); }
}

function searchText() {
  const q = $("q").value.trim();
  if (!q) return;
  setStatus("searching…");
  return run(async () => {
    const f = new FormData();
    f.append("text", q);
    render(await post("/search", f));
  });
}

function sendAudio(file, endpoint = "/search") {
  setStatus(endpoint === "/crate" ? "splitting into stems (Demucs)…" : "analyzing audio…");
  return run(async () => {
    const f = new FormData();
    f.append("audio", file, file.name || "clip.webm");
    render(await post(endpoint, f));
  });
}

function usePickedFile(file) {
  pickedFile = file;
  if (!file) return;
  $("buildCrate").disabled = false;
  $("drop").querySelector("b").textContent = "dropped: " + file.name;
  sendAudio(file);
}

function escapeHtml(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function licenseShort(u) { return /zero/.test(u) ? "CC0" : /by\//.test(u) ? "CC-BY" : u; }

// --- text ---
$("go").onclick = searchText;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") searchText(); });

// --- upload (click) + drop-zone ---
const drop = $("drop");
drop.onclick = () => $("file").click();
$("file").addEventListener("change", (e) => usePickedFile(e.target.files[0]));
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, () => drop.classList.remove("hot")));
drop.addEventListener("drop", (e) => { e.preventDefault(); usePickedFile(e.dataTransfer.files[0]); });
$("buildCrate").onclick = () => pickedFile && sendAudio(pickedFile, "/crate");

// --- mic hum ---
let recorder, chunks = [];
$("mic").onclick = async () => {
  const btn = $("mic");
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      sendAudio(new File(chunks, "hum.webm", { type: chunks[0]?.type || "audio/webm" }));
      btn.classList.remove("on");
      btn.querySelector("span").textContent = "Rec hum";
    };
    recorder.start();
    btn.classList.add("on");
    btn.querySelector("span").textContent = "Stop";
    setStatus("recording — hum the sound, then Stop");
  } catch (e) { setStatus("mic blocked: " + e.message); }
};

// --- shortcut chips: the three ways in ---
$("chip-hum").onclick = () => $("mic").click();
$("chip-drop").onclick = () => $("file").click();
$("chip-desc").onclick = () => $("q").focus();

// --- clickable example queries ---
document.querySelectorAll("#examples .ex").forEach((el) => {
  el.onclick = () => { $("q").value = el.textContent; searchText(); };
});

// --- animated sound-wave background ---
(function buildWave() {
  const el = $("wavebg");
  if (!el) return;
  const n = Math.min(72, Math.max(20, Math.floor(window.innerWidth / 20)));
  for (let i = 0; i < n; i++) {
    const s = document.createElement("span");
    s.style.animationDelay = (-Math.random() * 3.4).toFixed(2) + "s";
    s.style.animationDuration = (2.8 + Math.random() * 2.4).toFixed(2) + "s";
    el.appendChild(s);
  }
})();

// --- latency dashboard (only shown when real numbers exist) ---
fetch("/latency").then((r) => r.json()).then((d) => {
  if (!d || !d.base_torch_cpu_ms) return;
  $("lat-base").textContent = d.base_torch_cpu_ms;
  $("lat-int8").textContent = d.int8_onnx_cpu_ms;
  $("lat-speed").textContent = d.speedup;
  $("dash").hidden = false;
}).catch(() => {});
