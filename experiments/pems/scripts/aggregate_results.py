"""Aggregate exact-GP vs PRO test NLPD across training sizes and splits into results/summary.csv."""

import csv
import json
from pathlib import Path

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

NUM_TRAINS = (200, 225, 250, 275)
# method dir -> (metrics file, NLPD key)
METHODS = {
    "exact_gp": ("gp_metrics.json", "gp_nlpd"),
    "pro_gp_gibbs_50": ("pro_metrics.json", "pro_nlpd"),
}


def collect() -> list[dict]:
    rows = []
    for num_train in NUM_TRAINS:
        split_dirs = (RESULTS_ROOT / f"num_train_{num_train}").glob("split_*")
        for split_dir in sorted(split_dirs, key=lambda p: int(p.name.removeprefix("split_"))):
            split = int(split_dir.name.removeprefix("split_"))
            for method, (fname, key) in METHODS.items():
                path = split_dir / method / fname
                if path.exists():
                    nlpd = json.loads(path.read_text())[key]
                    rows.append(
                        {"num_train": num_train, "method": method, "split": split, "nlpd": nlpd}
                    )
    return rows


def main() -> None:
    rows = collect()
    path = RESULTS_ROOT / "summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_train", "method", "split", "nlpd"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'num_train':<11}" + "".join(f"{m:>18}" for m in METHODS))
    for num_train in NUM_TRAINS:
        line = f"{num_train:<11}"
        for method in METHODS:
            v = [r["nlpd"] for r in rows if r["num_train"] == num_train and r["method"] == method]
            se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            line += f"{f'{np.mean(v):.3f}±{se:.3f}' if v else '—':>18}"
        print(line)
    print(f"Saved per-split results to {path}")


if __name__ == "__main__":
    main()
