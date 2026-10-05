#!/usr/bin/env bash
# OMGP baselines: K-sensitivity on the existing regimes (E1), unknown number of
# branches (E2) and particles-vs-components (E3), then the mixture figures.
set -euo pipefail
cd "$(dirname "$0")"

python synthetic.py -m \
  ds@_global_=block_outliers,heteroskedastic,multimodal,well_specified \
  n=50,100,200 \
  'algorithms=[omgp]'

python synthetic.py -m \
  ds@_global_=multibranch \
  num_branches=1,2,3,4 \
  'algorithms=[standard_gp,pro_gp,omgp]'

for num_particles in 5 10 20 100; do
  python synthetic.py -m \
    ds@_global_=multimodal,well_specified \
    n=100 \
    'algorithms=[pro_gp]' \
    pro.num_particles=$num_particles \
    pro_label=pro_gp_n$num_particles
done
python synthetic.py -m \
  ds@_global_=multimodal,well_specified \
  n=100 \
  'algorithms=[omgp]' \
  'mixture.ks=[20,50]' \
  mixture.sparse_k=null \
  mixture.select_k=false

python aggregate_results.py
python plot_mixture.py
python multibranch_overlay.py
