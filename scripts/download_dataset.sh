#!/usr/bin/env bash
set -euo pipefail

# Download CIC-IDS-2018 dataset to data/raw/CSE-CIC-IDS2018/
# Uses AWS open bucket with no credentials required.

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw/CSE-CIC-IDS2018"
mkdir -p "$DATA_DIR"

echo "Downloading CIC-IDS-2018 to $DATA_DIR" >&2
echo "This is large (tens of GB). Ensure you have bandwidth and disk space." >&2

aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/" "$DATA_DIR/original" \
	--only-show-errors

aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" "$DATA_DIR/processed_ml" \
	--only-show-errors

echo "Download complete." >&2
