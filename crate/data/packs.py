"""Ingest local free sample packs — folder walk, tag from path + filename.

Point it at a directory of downloaded (licensing-clean) packs. Filenames and
parent folders are the only metadata sample packs ship, so we mine those.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from crate import config

AUDIO_EXT = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}
_SPLIT = re.compile(r"[_\-/\s]+")


def _caption_from_path(p: Path, root: Path) -> str:
    """Turn `Boom Bap Kit/drums/dusty_kick_01.wav` into readable text."""
    rel = p.relative_to(root).with_suffix("")
    words = [w for w in _SPLIT.split(str(rel)) if w and not w.isdigit()]
    # de-dup while preserving order
    seen, kept = set(), []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            kept.append(w.lower())
    return " ".join(kept)


def ingest(pack_root: str | Path, license_name: str = "unknown-local") -> int:
    """Register every audio file under pack_root. Returns clip count."""
    config.ensure_dirs()
    root = Path(pack_root)
    if not root.exists():
        raise FileNotFoundError(f"pack dir not found: {root}")
    seen = _existing_paths()
    written = 0
    with open(config.META_PATH, "a") as meta:
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXT or str(p) in seen:
                continue
            text = _caption_from_path(p, root)
            meta.write(json.dumps({
                "id": f"pack_{abs(hash(str(p))) % (10**10)}",
                "path": str(p),
                "text": text,
                "tags": text.split(),
                "license": license_name,
                "attribution": root.name,
                "source": "pack",
            }) + "\n")
            written += 1
    return written


def _existing_paths() -> set[str]:
    if not config.META_PATH.exists():
        return set()
    return {json.loads(line)["path"] for line in config.META_PATH.read_text().splitlines()}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m crate.data.packs <pack_dir> [license]")
        raise SystemExit(1)
    lic = sys.argv[2] if len(sys.argv) > 2 else "unknown-local"
    print(f"ingested {ingest(sys.argv[1], lic)} clips from {sys.argv[1]}")
