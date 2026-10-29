"""Inter-rater agreement for Phase B round 1 (pre-registered thresholds: Spearman rho >= 0.7
for pointedness, Cohen's kappa >= 0.6 for view). Reads data/phase_b/labels/phase_b_labels_rater{A,B}.csv.
No activations are involved.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data" / "phase_b" / "labels"


def load(r):
    return {row["image_id"]: row for row in csv.DictReader(open(LAB / f"phase_b_labels_rater{r}.csv"))}


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe)


def main():
    A, B = load("A"), load("B")
    both = [i for i in A if i in B and "rate" in A[i]["tasks"]]
    pa = np.array([float(A[i]["pointedness"]) for i in both])
    pb = np.array([float(B[i]["pointedness"]) for i in both])
    rho = spearmanr(pa, pb).statistic
    va, vb = [A[i]["view"] for i in both], [B[i]["view"] for i in both]
    k = kappa(va, vb)
    small = np.mean([A[i]["too_small"] == B[i]["too_small"] for i in both])
    exact = np.mean(pa == pb)
    within1 = np.mean(np.abs(pa - pb) <= 1)
    print(f"images rated by both: {len(both)}")
    print(f"pointedness: Spearman rho = {rho:.3f} (threshold 0.70) -> {'PASS' if rho >= 0.7 else 'FAIL'}; "
          f"exact {exact:.0%}, within 1 point {within1:.0%}")
    print(f"view: Cohen's kappa = {k:.3f} (threshold 0.60) -> {'PASS' if k >= 0.6 else 'FAIL'}; "
          f"raw agreement {np.mean([x == y for x, y in zip(va, vb)]):.0%}")
    print(f"too_small: raw agreement {small:.0%}")


if __name__ == "__main__":
    main()
