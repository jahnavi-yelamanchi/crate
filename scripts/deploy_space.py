"""One-shot HF Space deploy — no clicking.

    export HF_TOKEN=...        # a write token
    python scripts/deploy_space.py [space-id] [model-repo] [index-repo]

Creates the Docker Space, pushes spaces/{Dockerfile,boot.py,README.md}, and sets the
two runtime variables it needs. Idempotent — re-run to update. Standalone: the only
dependency is huggingface_hub (so it runs from Colab, CI, or your laptop).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACES = Path(__file__).resolve().parent.parent / "spaces"
FILES = ("Dockerfile", "boot.py", "README.md")


def main(space: str, model_repo: str, index_repo: str) -> None:
    token = os.getenv("HF_TOKEN") or True
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True, token=token)
    for name in FILES:
        api.upload_file(path_or_fileobj=str(SPACES / name), path_in_repo=name,
                        repo_id=space, repo_type="space", token=token)
    api.add_space_variable(space, "HF_MODEL_REPO", model_repo, token=token)
    api.add_space_variable(space, "CRATE_ARTIFACTS_REPO", index_repo, token=token)
    print(f"deployed → https://huggingface.co/spaces/{space}")
    print("building now — watch the log there; first build installs torch/demucs (~5-10 min)")


if __name__ == "__main__":
    a = sys.argv
    main(
        a[1] if len(a) > 1 else "jahnaviym/crate",
        a[2] if len(a) > 2 else "jahnaviym/crate-clap-lora",
        a[3] if len(a) > 3 else "jahnaviym/crate-index",
    )
