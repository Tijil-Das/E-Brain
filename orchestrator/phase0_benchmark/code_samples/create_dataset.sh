#!/usr/bin/env bash
# create_dataset.sh — Upload ebrain-phase0-code-samples to Kaggle once.
# Run from: orchestrator/phase0_benchmark/code_samples/
# Requires: kaggle<2.0.0, ~/.kaggle/kaggle.json with training account credentials.
#
# Replace TRAINING_KAGGLE_USERNAME in dataset-metadata.json with the real username
# before running (or the workflow does it automatically at push time).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== E-Brain Phase 0 Code Samples Dataset Upload ==="
echo "Working dir: $SCRIPT_DIR"
echo ""
echo "Files to upload:"
for f in sample.py sample.js sample.java sample.cpp sample.go; do
    if [ -f "$f" ]; then
        sha=$(sha256sum "$f" | awk '{print $1}')
        echo "  $f  sha256=$sha"
    else
        echo "  MISSING: $f"
        exit 1
    fi
done
echo ""

# Check metadata has a real username, not the placeholder
if grep -q "TRAINING_KAGGLE_USERNAME" dataset-metadata.json; then
    echo "ERROR: dataset-metadata.json still contains TRAINING_KAGGLE_USERNAME placeholder."
    echo "Edit it to use your Kaggle training account username, then re-run."
    exit 1
fi

echo "Creating dataset (first time)..."
kaggle datasets create -p . || {
    echo ""
    echo "If the dataset already exists, run this instead:"
    echo "  kaggle datasets version -p . -m 'code sample files v1'"
    exit 1
}
echo ""
echo "Done. Verify at: https://www.kaggle.com/datasets/<username>/ebrain-phase0-code-samples"
