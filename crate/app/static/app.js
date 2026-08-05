// Crate frontend — mic hum, drop-zone, text. One embedding space, three ways in.
const $ = (id) => document.getElementById(id);
let sessionId = null;
let droppedFile = null;

function setStatus(msg) { $("status").textContent = msg || ""; }

async function post(url, form) {
  if (sessionId) form.append("session_id", sessionId);
  const r = await fetch(url, { method: "POST", body: form });
  if (!r.ok) { setStatus("error: " + (await r.text())); throw new Error(r.status); }
  const data = await r.json();
  if (data.session_id) sessionId = data.session_id;
  return data;
}

function render(data) {
  const box = $("results");
  box.innerHTML = "";
  if (data.kit) { renderKit(data.kit, box); return; }
  setStatus(`${data.modality} query · ${data.results.length} results`);
  data.results.forEach((r) => box.appendChild(resultRow(r)));
}

function renderKit(kit, box) {
  setStatus("crate assembled from stems");
  for (const role of Object.keys(kit)) {
    const h = document.createElement("div");
    h.className = "kit-role"; h.textContent = role;
    box.appendChild(h);
    kit[role].forEach((r) => box.appendChild(resultRow(r)));
  }
}

function resultRow(r) {
  const el = document.createElement("div");
  el.className = "result";
  const score = (r.final ?? r.score ?? 0).toFixed(3);
  el.innerHTML = `
    <span class="score">${score}</span>
    <div class="meta">
      <div class="t">${r.text || r.id}</div>
      <div class="sub">${r.attribution || ""} ${r.license ? "· " + r.license : ""}</div>
    </div>`;
  el.appendChild(feedbackBtn("♥", () => feedback("/save", r.id, el)));
  el.appendChild(feedbackBtn("✕", () => feedback("/skip", r.id, el)));
  return el;
}

function feedbackBtn(label, fn) {
  const b = document.createElement("button");
  b.className = "iconbtn"; b.textContent = label; b.onclick = fn;
  return b;
}

async function feedback(url, id, el) {
  const f = new FormData();
  f.append("result_id", id);
  await post(url, f);
  el.style.opacity = url.endsWith("save") ? "1" : "0.4";
}

async function searchText() {
  const q = $("q").value.trim();
  if (!q) return;
  setStatus("searching…");
  const f = new FormData(); f.append("text", q);
  render(await post("/search", f));
}

async function searchAudio(file, endpoint = "/search") {
  setStatus("analyzing audio…");
  const f = new FormData(); f.append("audio", file, file.name || "clip.webm");
  render(await post(endpoint, f));
}

// --- text ---
$("go").onclick = searchText;
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") searchText(); });

// --- drop-zone ---
const drop = $("drop");
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hot"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, () => drop.classList.remove("hot")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  droppedFile = e.dataTransfer.files[0];
  if (!droppedFile) return;
  $("buildCrate").disabled = false;
  drop.firstChild.textContent = "dropped: " + droppedFile.name;
  searchAudio(droppedFile);
});
$("buildCrate").onclick = () => droppedFile && searchAudio(droppedFile, "/crate");

// --- mic hum ---
let recorder, chunks = [];
$("mic").onclick = async () => {
  if (recorder && recorder.state === "recording") { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      searchAudio(new File(chunks, "hum.webm", { type: chunks[0]?.type || "audio/webm" }));
      $("mic").textContent = "● Rec hum"; $("mic").classList.remove("rec");
    };
    recorder.start();
    $("mic").textContent = "■ Stop"; $("mic").classList.add("rec");
    setStatus("recording — hum the sound, then Stop");
  } catch (e) { setStatus("mic blocked: " + e.message); }
};

// --- latency dashboard ---
fetch("/latency").then((r) => r.json()).then((d) => {
  if (d.base_torch_cpu_ms) $("lat-base").textContent = d.base_torch_cpu_ms;
  if (d.int8_onnx_cpu_ms) $("lat-int8").textContent = d.int8_onnx_cpu_ms;
  if (d.speedup) $("lat-speed").textContent = d.speedup;
}).catch(() => {});
