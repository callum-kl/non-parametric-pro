#!/usr/bin/env bash
# Finish the protein benchmark: run both PRO-GP stages on the saved vgp_noncollapsed
# and ppgpr fits, then rebuild results/summary.csv.
#
# Usage: bash experiments/uci/scripts/run_protein_pro.sh [splits] [n_jobs]
#   defaults: splits=1,2,3,4,5,6,7,8,9,10  n_jobs=2 (protein peaks at ~12.6GB per job)
set -euo pipefail

SPLITS="${1:-1,2,3,4,5,6,7,8,9,10}"
N_JOBS="${2:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

common=(-m hydra/launcher=joblib "hydra.launcher.n_jobs=$N_JOBS"
        "ds@_global_=protein" "split=$SPLITS" seed=0)

uv run python "$SCRIPT_DIR/fit_pro.py" "${common[@]}" vgp_variant=noncollapsed name=gibbs
uv run python "$SCRIPT_DIR/fit_pro.py" "${common[@]}" vgp_variant=ppgpr name=gibbs_ppgpr
uv run python "$SCRIPT_DIR/aggregate_results.py"
