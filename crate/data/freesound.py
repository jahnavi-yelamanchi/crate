"""Freesound API client — search + download licensing-clean audio.

Only pulls CC0 / CC-BY (attribution tracked). Writes one JSONL line of metadata
per clip so the rest of the pipeline never re-hits the API.

Get a key: https://freesound.org/apiv2/apply  → put FREESOUND_KEY in .env
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from crate import config

API = "https://freesound.org/apiv2"

# Freesound license strings we accept (licensing-clean, redistributable).
CLEAN_LICENSES = (
    "Creative Commons 0",
    "Attribution",
    "Attribution 4.0",
    "Attribution 3.0",
)


def _clean(license_str: str) -> bool:
    return any(license_str.startswith(ok) for ok in CLEAN_LICENSES)


def _get(url: str, params: dict, retries: int = 5) -> dict:
    """GET with backoff on 429/5xx. Freesound free tier is ~60 req/min."""
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("Retry-After", 2 ** attempt))
            print(f"  rate-limited ({r.status_code}), backing off {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()  # out of retries → surface the last error
    return {}


def search(query: str, page_size: int = 50, max_results: int = 200,
           max_pages: int = 4) -> list[dict]:
    """Search Freesound, filtered to clean licenses. Returns raw sound dicts.

    Caps pages so a query where most results fail the license filter can't page
    dozens deep and burn the rate limit.
    """
    if not config.FREESOUND_KEY:
        raise RuntimeError("FREESOUND_KEY missing — add it to .env (see .env.example)")
    out: list[dict] = []
    url = f"{API}/search/text/"
    params = {
        "query": query,
        "token": config.FREESOUND_KEY,
        "page_size": page_size,
        "fields": "id,name,tags,license,previews,username,duration",
        "filter": "duration:[0.5 TO 30]",
    }
    pages = 0
    while url and len(out) < max_results and pages < max_pages:
        data = _get(url, params)
        for s in data.get("results", []):
            if _clean(s.get("license", "")):
                out.append(s)
        url = data.get("next")
        params = {"token": config.FREESOUND_KEY}  # `next` already carries query params
        pages += 1
        time.sleep(1.0)  # stay under ~60 req/min
    return out[:max_results]


def download_preview(sound: dict, dest_dir: Path) -> Path | None:
    """Grab the hq mp3 preview (no OAuth needed, unlike the original)."""
    preview_url = sound.get("previews", {}).get("preview-hq-mp3")
    if not preview_url:
        return None
    dest = dest_dir / f"fs_{sound['id']}.mp3"
    if dest.exists():
        return dest
    r = requests.get(preview_url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def ingest(queries: list[str], per_query: int = 100) -> int:
    """Search each query, download previews, append metadata. Returns clip count."""
    config.ensure_dirs()
    seen = _existing_ids()
    written = 0
    with open(config.META_PATH, "a") as meta:
        for q in queries:
            try:
                hits = search(q, max_results=per_query)
            except requests.HTTPError as e:
                print(f"  skip query '{q}': {e}")
                continue
            for s in hits:
                if s["id"] in seen:
                    continue
                try:
                    path = download_preview(s, config.RAW_DIR)
                except requests.HTTPError as e:
                    print(f"  skip download {s['id']}: {e}")
                    continue
                if path is None:
                    continue
                text = _caption(s, q)
                meta.write(json.dumps({
                    "id": f"fs_{s['id']}",
                    "path": str(path),
                    "text": text,
                    "tags": s.get("tags", []),
                    "license": s["license"],
                    "attribution": s.get("username", ""),
                    "source": "freesound",
                    "query": q,
                }) + "\n")
                seen.add(s["id"])
                written += 1
    return written


def _caption(sound: dict, query: str) -> str:
    """Cheap caption: query term + top tags + name. The fine-tune improves on this."""
    tags = " ".join(sound.get("tags", [])[:6])
    name = sound.get("name", "").rsplit(".", 1)[0]
    return f"{query}, {tags}, {name}".strip(", ")


def _existing_ids() -> set[int]:
    if not config.META_PATH.exists():
        return set()
    ids = set()
    for line in config.META_PATH.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("source") == "freesound":
            ids.add(int(rec["id"].removeprefix("fs_")))
    return ids


if __name__ == "__main__":
    # Seed queries from the producer vocab so we pull sounds we actually care about.
    from crate.data.pairs import load_vocab

    terms = load_vocab()
    n = ingest(terms, per_query=60)
    print(f"ingested {n} new clips → {config.META_PATH}")
