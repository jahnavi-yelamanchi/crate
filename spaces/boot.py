"""Space entrypoint: fetch prebuilt index artifacts, then serve the app on 7860.

Keeps big files out of git — the FAISS index, id map, and metadata live in an HF
dataset repo (CRATE_ARTIFACTS_REPO) and are downloaded once at container start.
"""

import os

from huggingface_hub import hf_hub_download

from crate import config

ARTIFACTS_REPO = os.environ.get("CRATE_ARTIFACTS_REPO", "")
TOKEN = os.environ.get("HF_TOKEN") or None


def fetch():
    if not ARTIFACTS_REPO:
        print("WARNING: CRATE_ARTIFACTS_REPO unset — app will 503 until an index exists")
        return
    config.ensure_dirs()
    targets = {
        "index.faiss": config.INDEX_PATH,
        "ids.json": config.IDS_PATH,
        "metadata.jsonl": config.META_PATH,
    }
    for name, dest in targets.items():
        path = hf_hub_download(ARTIFACTS_REPO, name, repo_type="dataset", token=TOKEN)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(open(path, "rb").read())
        print(f"fetched {name} → {dest}")


if __name__ == "__main__":
    import uvicorn

    fetch()
    uvicorn.run("crate.app.main:app", host="0.0.0.0", port=7860)
