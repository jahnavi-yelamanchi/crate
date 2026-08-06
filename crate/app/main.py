"""FastAPI backend — one search box: mic + drop-zone + text.

Endpoints:
  GET  /            → the app
  POST /search      → text field OR uploaded audio (hum/track) → ranked results
  POST /crate       → uploaded reference track → assembled per-stem kit
  POST /save /skip  → taste feedback, retrains the reco head live
  GET  /latency     → quantized-encoder benchmark for the dashboard

The heavy agent (encoder + FAISS) loads lazily so the process starts instantly and
missing artifacts return a clear 503 instead of crashing at import.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from crate import config

app = FastAPI(title="Crate", description="Sound-native search for producers")
STATIC = Path(__file__).parent / "static"

_AGENT = None
_META = None


def _meta() -> dict[str, str]:
    """clip id → source file path, read once. Used to stream audio without the model."""
    global _META
    if _META is None:
        _META = {}
        if config.META_PATH.exists():
            for line in config.META_PATH.read_text().splitlines():
                rec = json.loads(line)
                _META[rec["id"]] = rec.get("path", "")
    return _META


def agent():
    """Lazy singleton. 503 if the index/model artifacts aren't built yet."""
    global _AGENT
    if _AGENT is None:
        try:
            from crate.agent.graph import CrateAgent

            _AGENT = CrateAgent()
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(503, f"not ready: {e}. Install extras and build the index.") from e
    return _AGENT


def _decode_upload(file: UploadFile) -> np.ndarray:
    """Bytes from mic/drop → 48k mono fixed-length. Needs ffmpeg for webm/opus."""
    from crate.data.preprocess import preprocess_one

    suffix = Path(file.filename or "clip").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        return preprocess_one(path)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}") from e
    finally:
        Path(path).unlink(missing_ok=True)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/search")
async def search(
    session_id: str = Form(None),
    text: str = Form(None),
    audio: UploadFile = File(None),
):
    sid = session_id or uuid.uuid4().hex
    if text:
        query = text
    elif audio is not None:
        query = _decode_upload(audio)
    else:
        raise HTTPException(400, "provide `text` or an `audio` file")
    out = agent().query(sid, query)
    return JSONResponse({"session_id": sid, **out})


@app.post("/crate")
async def crate(session_id: str = Form(None), audio: UploadFile = File(...)):
    sid = session_id or uuid.uuid4().hex
    wav = _decode_upload(audio)
    try:
        from crate.agent.crate_builder import build_kit

        kit = build_kit(wav)
    except ImportError as e:
        raise HTTPException(503, f"stem separation needs Demucs — `pip install demucs`. ({e})") from e
    except FileNotFoundError as e:
        raise HTTPException(503, f"not ready: {e}. Build the index first.") from e
    return JSONResponse({"session_id": sid, "kit": kit})


@app.get("/audio/{clip_id}")
def audio_clip(clip_id: str):
    """Stream a result's source clip so the browser can play it."""
    path = _meta().get(clip_id)
    if not path or not Path(path).exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(path)


@app.post("/save")
async def save(session_id: str = Form(...), result_id: str = Form(...)):
    agent().feedback(session_id, result_id, saved=True)
    return {"ok": True}


@app.post("/skip")
async def skip(session_id: str = Form(...), result_id: str = Form(...)):
    agent().feedback(session_id, result_id, saved=False)
    return {"ok": True}


@app.get("/latency")
def latency():
    path = config.EVAL_DIR / "latency.json"
    if not path.exists():
        return {"note": "run `python -m crate.model.quantize` to generate latency numbers"}
    return json.loads(path.read_text())


# static assets (css/js) — mounted last so it doesn't shadow the routes above
app.mount("/static", StaticFiles(directory=STATIC), name="static")
