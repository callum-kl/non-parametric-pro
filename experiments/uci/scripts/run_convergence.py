"""
Convergence / runtime sweep over the number of inducing points, run strictly sequentially so timings are not contended.

For each (seed, m):
    convergence_vgp.py     (vgp_noncollapsed_conv_m{m}_s{seed}: VGP fit, objective trace, timings)
    fit_pro.py             (inducing_pro_gp_conv_m{m}_s{seed}: sigma adaptation with fit_pro_gibbs.yaml)
    convergence_pro.py     (inducing_pro_gp_conv_m{m}_s{seed}: fixed-sigma Gibbs score trace, timings)

Usage:
    python experiments/uci/scripts/run_convergence.py
    python experiments/uci/scripts/run_convergence.py --ms 100,250 --seeds 0 gp_num_iters=500
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR.parent / "results"


def _run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="parkinsons")
    parser.add_argument("--split", default="1")
    parser.add_argument("--ms", default="100,250,500")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "overrides", nargs="*", help="extra hydra overrides for convergence.yaml"
    )
    args = parser.parse_args()

    split_dir = RESULTS_ROOT / args.dataset / f"split_{args.split}"
    for seed in args.seeds.split(","):
        for m in args.ms.split(","):
            tag = f"conv_m{m}_s{seed}"
            common = [f"ds@_global_={args.dataset}", f"split={args.split}", f"seed={seed}"]
            conv = [*common, f"num_inducing={m}", *args.overrides]

            if not (split_dir / f"vgp_noncollapsed_{tag}" / "convergence.npz").exists():
                _run([sys.executable, str(SCRIPT_DIR / "convergence_vgp.py"), *conv])

            pro_dir = split_dir / f"inducing_pro_gp_{tag}"
            if not (pro_dir / "pro_metrics.json").exists():
                _run(
                    [
                        sys.executable,
                        str(SCRIPT_DIR / "fit_pro.py"),
                        "--config-name",
                        "fit_pro_gibbs",
                        *common,
                        "inducing=true",
                        "vgp_variant=noncollapsed",
                        f"vgp_name={tag}",
                        f"name={tag}",
                    ]
                )

            if not (pro_dir / "convergence.npz").exists():
                _run([sys.executable, str(SCRIPT_DIR / "convergence_pro.py"), *conv])


if __name__ == "__main__":
    main()
