"""Bootstrap CIs and the angle curve for results/tighten/scores.npz.

Statistic per condition: mean of the top 10% of restarts (top 13 of 128). This is the
"what the optimizer can reach" level, less noisy than the single best. CIs: 10,000
bootstrap resamples of restarts within each condition.
"""
from pathlib import Path

import numpy as np

import sys
OUT = Path(__file__).resolve().parent.parent / "results" / "tighten" / (sys.argv[1] if len(sys.argv) > 1 else "")
ANGLES = [20, 45, 70, 90, 120, 150, 180]
B, TOP = 10000, 13


def topmean(x):
    return np.sort(x, -1)[..., -TOP:].mean(-1)


def boot(x, rng):
    idx = rng.integers(0, len(x), (B, len(x)))
    return topmean(x[idx])


def main():
    s = {k: v.ravel() for k, v in np.load(OUT / "scores.npz").items()}
    rng = np.random.default_rng(0)
    bs = {k: boot(v, rng) for k, v in s.items()}
    lines = [f"Statistic: mean of top {TOP} of {len(next(iter(s.values())))} restarts; 95% bootstrap CI", ""]
    for k in s:
        lo, hi = np.percentile(bs[k], [2.5, 97.5])
        lines.append(f"  {k:>7}: {topmean(s[k]):6.0f}  [{lo:5.0f}, {hi:5.0f}]   best {s[k].max():5.0f}")
    lines.append("")

    def diff(a, b, label):
        d = bs[a] - bs[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = (d <= 0).mean()
        lines.append(f"  {label:<38} {topmean(s[a]) - topmean(s[b]):+6.0f}  [{lo:+5.0f}, {hi:+5.0f}]  P(diff<=0)={p:.3f}")

    lines.append("Predicted comparisons (positive = first is higher):")
    if "V_free" in s:
        diff("V_free", "bars2", "V (free angle) - two bars")
        diff("V_free", "O", "V (free angle) - O")
    acute = max(("V20", "V45", "V70"), key=lambda k: topmean(s[k]))
    diff(acute, "V180", f"best acute ({acute}) - straight bar (V180)")
    diff("V90", "V150", "V90 - V150")
    lines.append("")
    curve = [topmean(s[f"V{a}"]) for a in ANGLES]
    slope = np.polyfit(ANGLES, curve, 1)[0]
    bslope = np.array([np.polyfit(ANGLES, [bs[f"V{a}"][i] for a in ANGLES], 1)[0] for i in range(0, B, 10)])
    lo, hi = np.percentile(bslope, [2.5, 97.5])
    lines.append("Angle curve (top-10% mean): " + ", ".join(f"{a}deg {c:.0f}" for a, c in zip(ANGLES, curve)))
    lines.append(f"  linear slope {slope:+.2f} per degree  [{lo:+.2f}, {hi:+.2f}]  "
                 f"(V detector predicts negative; peak at {ANGLES[int(np.argmax(curve))]} deg)")
    print("\n".join(lines))
    (OUT / "analysis.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
