#!/usr/bin/env bash
# Download the model artifacts produced by the three Kaggle notebooks into
# local folders, using the official Kaggle CLI (pip install kaggle; token from
# kaggle.com/settings → "Create New Token" → ~/.kaggle/access_token).
#
# Usage: bash tools/fetch_kaggle_outputs.sh [kaggle_username]
set -euo pipefail
cd "$(dirname "$0")/.."

USER=${1:-aliomarsaleh}
OUT=${OUT:-kaggle-outputs}

for slug in digitlab-01-classical-ml-rbf-svm \
            digitlab-02-neural-network-mlp \
            digitlab-03-deep-learning-cnn; do
  echo "==> $USER/$slug"
  kaggle kernels output "$USER/$slug" -p "$OUT/$slug"
done

echo "done — copy the model files into models/ with:"
echo "  cp $OUT/digitlab-*/models/* models/"
