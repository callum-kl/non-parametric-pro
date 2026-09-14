import csv
import math
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path(__file__).with_name("stage2.csv").open()))
for r in rows:
    for k in ("lr", "init_ell", "ell_star", "ell_final", "nlpd", "sigma", "gap"):
        r[k] = float(r[k])
    for k in ("total", "kernel_adapt_steps"):
        r[k] = int(r[k])

cells = defaultdict(list)
for r in rows:
    cells[(r["regime"], r["lr"], r["total"], r["kernel_adapt_steps"])].append(r)

agg = defaultdict(lambda: {"spread": [], "bias": [], "nlpd": []})
for (_regime, lr, total, kas), group in cells.items():
    ells = [g["ell_final"] for g in group]
    if not all(math.isfinite(e) and e > 0 for e in ells):
        continue
    logs = [math.log(e) for e in ells]
    a = agg[(lr, total, kas)]
    a["spread"].append(max(logs) - min(logs))
    a["bias"].append(sum(logs) / len(logs) - math.log(group[0]["ell_star"]))
    a["nlpd"].append(sum(g["nlpd"] for g in group) / len(group))

print("Stage 2 (trimmed): 4 regimes x 1 seed x 4 initial lengthscales {0.05,0.3,1.0,3.0}")
print("gap = sampler iterations between kernel updates = (total-20)/k_adapt")
print("spread = log-range of final ell across inits (0 = fully robust)\n")
print(f"{'lr':>5}{'total':>7}{'k_adapt':>9}{'frac':>7}{'gap':>7}{'spread':>9}{'bias':>8}{'nlpd':>9}")
table = []
for key in sorted(agg, key=lambda k: (k[0], (k[1] - 20) / k[2])):
    lr, total, kas = key
    a = agg[key]
    n = len(a["spread"])
    sp, bi, nl = (sum(a[k]) / n for k in ("spread", "bias", "nlpd"))
    table.append((sp, bi, nl, key))
    print(f"{lr:>5}{total:>7}{kas:>9}{kas/total:>7.2f}{(total-20)/kas:>7.1f}{sp:>9.3f}{bi:>8.3f}{nl:>9.3f}")

print("\ncollapsed by gap (pooling both learning rates):")
by_gap = defaultdict(list)
for sp, _, _, key in table:
    by_gap[round((key[1] - 20) / key[2], 1)].append(sp)
for g in sorted(by_gap):
    v = by_gap[g]
    print(f"  gap={g:>5.1f}  mean spread={sum(v)/len(v):.3f}  (n={len(v)})")

print("\ncollapsed by kernel-update count:")
by_count = defaultdict(list)
for sp, _, _, key in table:
    by_count[key[2]].append(sp)
for c in sorted(by_count):
    v = by_count[c]
    print(f"  k_adapt={c:>4}  mean spread={sum(v)/len(v):.3f}  (n={len(v)})")

print("\ncollapsed by learning rate:")
by_lr = defaultdict(list)
for sp, _, _, key in table:
    by_lr[key[0]].append(sp)
for lr in sorted(by_lr):
    v = by_lr[lr]
    print(f"  lr={lr:<5} mean spread={sum(v)/len(v):.3f}  (n={len(v)})")

best = min(table)
print(f"\nmost robust: lr={best[3][0]}, total={best[3][1]}, k_adapt={best[3][2]}"
      f" -> spread={best[0]:.3f} bias={best[1]:.3f} nlpd={best[2]:.3f}")
print(f"best nlpd  : {min(table, key=lambda t: t[2])[3]}")
