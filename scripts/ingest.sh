#!/usr/bin/env bash
# Full data pipeline: pull licensing-clean audio → preprocess → build pairs.
# Prereqs: FREESOUND_KEY in .env. Optional: pass a local sample-pack dir as $1.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] Freesound ingest (vocab-seeded queries)"
python -m crate.data.freesound

if [ "${1:-}" != "" ]; then
  echo "==> [1b] local pack ingest: $1"
  python -m crate.data.packs "$1" "${2:-unknown-local}"
fi

echo "==> [2/4] preprocess → 48k mono .npy"
python -m crate.data.preprocess

echo "==> [3/4] build train/val/heldout pairs"
python -m crate.data.pairs

echo "==> [4/4] done. metadata: data/metadata.jsonl  pairs: data/pairs/"
