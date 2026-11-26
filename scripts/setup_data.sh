#!/usr/bin/env bash
# Download AIST++ SMPL motions (~80 MB) and build data/library.npz (~40 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
if [ ! -f data/motions.zip ]; then
  curl -L -o data/motions.zip https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip
fi
rm -rf data/motions && unzip -q -o data/motions.zip -d data/ -x '__MACOSX/*'
python server/build_library.py data/motions data/library.npz
