#!/usr/bin/env bash
# Regenerate figures/example_grid_and_summary_columns.png from scratch.
set -euo pipefail
cd "$(dirname "$0")"

python synthetic.py -m \
  ds@_global_=block_outliers,heteroskedastic,multimodal,well_specified \
  algorithm=standard_gp,pro_gp \
  n=50,100,200

python aggregate_results.py
python combine_grid_summary_columns.py
