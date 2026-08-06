"""Serve the Crate app on a free public URL via a cloudflared quick tunnel.

For Colab (or any Linux box) where the index is already built and deps installed:

    python scripts/serve_colab.py

Prints a https://<random>.trycloudflare.com URL and stays running to keep the demo
live. No account, no auth, no HF Pro. The URL dies when you stop the cell.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CF = Path("/usr/local/bin/cloudflared")
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
PORT = 8000
ROOT = Path(__file__).resolve().parent.parent


def _ensure_cloudflared() -> None:
    if CF.exists():
        return
    print("downloading cloudflared…", flush=True)
    urllib.request.urlretrieve(CF_URL, CF)
    CF.chmod(0o755)


def _wait_healthy(timeout: int = 120) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/latency", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def main() -> None:
    # The index was built with the fine-tuned encoder — queries must use it too, or
    # query/corpus embeddings mismatch and results silently degrade.
    if not os.getenv("HF_MODEL_REPO"):
        print("WARNING: HF_MODEL_REPO unset — serving with BASE CLAP, but the index was "
              "built with the fine-tuned adapter. Set it so queries match the index.", flush=True)

    _ensure_cloudflared()
    app = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "crate.app.main:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=str(ROOT),
    )
    if not _wait_healthy():
        app.terminate()
        sys.exit("app failed to start — check deps and that the index is built")

    tun = subprocess.Popen([str(CF), "tunnel", "--url", f"http://localhost:{PORT}"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in tun.stdout:
            m = re.search(r"https://[-\w]+\.trycloudflare\.com", line)
            if m:
                print(f"\n🎚️  LIVE → {m.group()}\n(keep this cell running to keep it up)\n",
                      flush=True)
        tun.wait()
    finally:
        tun.terminate()
        app.terminate()


if __name__ == "__main__":
    main()
