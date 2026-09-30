#!/usr/bin/env bash
# Regenerate results/summary.csv and both figures from scratch.
set -euo pipefail
cd "$(dirname "$0")"

python synthetic.py -m \
  ds@_global_=block_outliers,heteroskedastic,multimodal \
  n=50,100,200

python aggregate_results.py
python combine_grid_summary_columns.py
python multimodal_overlay.py
