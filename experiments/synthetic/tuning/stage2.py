"""Isolates what makes kernel adaptation robust to its initial lengthscale.

Schedules are chosen so the gap between consecutive kernel updates varies both at
fixed update count (150/300/500 total, 30 updates) and at fixed total (300 total,
10/30/60 updates) -- separating "how many updates" from "how far apart".
Rows are flushed as they complete, so a partial run is still usable.
"""

import csv
import itertools
import time
from pathlib import Path

import jax.random as jr
from harness import run

from non_parametric_pro.data.synthetic.illustrative import make_illustrative_instance

REGIMES = {"block_outliers": 16, "heteroskedastic": 1, "multimodal": 10, "well_specified": 1}
ELL_STAR = {"block_outliers": 0.0532, "heteroskedastic": 0.1787,
            "multimodal": 0.1227, "well_specified": 0.1909}
INIT_ELL = [0.05, 0.3, 1.0, 3.0]
LRS = [0.2, 0.4]
SCHEDULES = [(150, 30), (300, 30), (500, 30), (300, 10), (300, 60)]
SEEDS = [0]

OUT = Path(__file__).parent / "stage2.csv"
FIELDS = ["regime", "seed", "lr", "total", "kernel_adapt_steps", "gap",
          "init_ell", "ell_star", "ell_final", "nlpd", "sigma"]

keys = jr.split(jr.PRNGKey(2421), 20)
combos = list(itertools.product(REGIMES, SEEDS, LRS, SCHEDULES, INIT_ELL))
t0 = time.time()

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    f.flush()
    for i, (regime, seed, lr, (total, kas), ell0) in enumerate(combos):
        data = make_illustrative_instance(keys[REGIMES[regime]], regime=regime)
        r = run(data, jr.PRNGKey(seed), kernel_lengthscale=ell0, kernel_lr=lr,
                kernel_adapt_steps=kas, num_adapt_steps=total)
        writer.writerow({
            "regime": regime, "seed": seed, "lr": lr, "total": total,
            "kernel_adapt_steps": kas, "gap": round((total - 20) / kas, 2),
            "init_ell": ell0, "ell_star": ELL_STAR[regime],
            "ell_final": r["ell"], "nlpd": r["nlpd"], "sigma": r["sigma"],
        })
        f.flush()
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{len(combos)}  {time.time()-t0:.0f}s", flush=True)

print(f"done {len(combos)} in {time.time()-t0:.0f}s -> {OUT}")
