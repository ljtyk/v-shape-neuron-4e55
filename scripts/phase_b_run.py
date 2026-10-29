"""Phase B: the single confirmatory run (measurement + pre-registered analysis).

  python phase_b_run.py --dry-run    random fake activations, no model; tests the pipeline only
  python phase_b_run.py              REAL run: only after the manifest is frozen and hashed

Measurement (phase-b-design.md): mixed4e:55 pre-ReLU, spatial max and spatial mean on the model's
224 px square, minus the gray-image baseline. Decision rules are implemented exactly as written in
the design doc; nothing here may change after freezing.
Writes results/phase_b{_DRYRUN}/activations.csv and report.md.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
B_RESAMPLES = 10000
B_REGRESSION = 2000
RNG_SEED = 20261028


# ------------------------------------------------------------------ measurement
def measure(rows, dry):
    if dry:
        rng = np.random.default_rng(0)
        return {r["stim_id"]: (float(rng.normal(300, 200)), float(rng.normal(-100, 80))) for r in rows}
    sys.path.insert(0, str(Path(__file__).parent))
    from neuron import Neuron
    from PIL import Image
    n = Neuron()
    base_max, base_mean, _ = n.measure([Image.new("RGB", (224, 224), (128, 128, 128))])[0]
    out = {}
    for k in range(0, len(rows), 64):
        chunk = rows[k:k + 64]
        res = n.measure([Image.open(r["path"]) for r in chunk])
        for r, (mx, mn, _) in zip(chunk, res):
            out[r["stim_id"]] = (mx - base_max, mn - base_mean)
    return out


# ------------------------------------------------------------------ helpers
def boot_ci(stat, data, rng, n=B_RESAMPLES):
    """95% percentile CI of stat over resamples of each array in data (independently)."""
    vals = []
    for _ in range(n):
        vals.append(stat(*[d[rng.integers(0, len(d), len(d))] for d in data]))
    return np.percentile(vals, [2.5, 97.5])


def design(levels_by_factor, rows, include):
    """One-hot design matrix (treatment coding) for the given terms."""
    cols = [np.ones(len(rows))]
    for term in include:
        if "*" in term:
            f1, f2 = term.split("*")
            for a in levels_by_factor[f1][1:]:
                for b in levels_by_factor[f2][1:]:
                    cols.append(np.array([(r[f1] == a) * (r[f2] == b) for r in rows], float))
        else:
            for a in levels_by_factor[term][1:]:
                cols.append(np.array([r[term] == a for r in rows], float))
    return np.column_stack(cols)


def sse(X, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(((y - X @ beta) ** 2).sum())


def partial_r2(rows, y, factor):
    lv = {f: sorted({r[f] for r in rows}) for f in ("shape", "texture", "size")}
    full = ["shape", "texture", "size", "shape*texture"]
    reduced = [t for t in full if factor not in t]
    s_full, s_red = sse(design(lv, rows, full), y), sse(design(lv, rows, reduced), y)
    return (s_red - s_full) / s_red if s_red > 0 else 0.0


# ------------------------------------------------------------------ analysis
def analyze(rows, act, out_dir):
    rng = np.random.default_rng(RNG_SEED)
    L = []
    say = L.append
    mx = lambda ids: np.array([act[i][0] for i in ids])  # noqa: E731
    mn = lambda ids: np.array([act[i][1] for i in ids])  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(r["group"], []).append(r)
    verdict = {}

    # Test 1
    P, F = mx([r["stim_id"] for r in by["T1_pointed"]]), mx([r["stim_id"] for r in by["T1_roundface"]])
    d = P.mean() - F.mean()
    lo, hi = boot_ci(lambda a, b: a.mean() - b.mean(), [P, F], rng)
    v1 = "H_V" if d >= 150 and lo > 0 else "routes/category" if d <= -150 and hi < 0 else "inconclusive"
    verdict["T1"] = v1
    say(f"## Test 1: pointed objects vs. round faces\n\nn = {len(P)} pointed, {len(F)} round faces. "
        f"Mean max: pointed {P.mean():.0f}, round faces {F.mean():.0f}.\n\n"
        f"Difference (pointed − round faces) = **{d:+.0f}**, 95% CI [{lo:+.0f}, {hi:+.0f}]. Verdict: **{v1}**.\n")

    # Test 2
    comp = []
    for r in by["T2_composite"]:
        shape, size, tex, src = r["extra"].split("|")
        comp.append(dict(id=r["stim_id"], shape=shape, size=size, texture=tex, src=src))
    y = mx([c["id"] for c in comp])
    pr = {f: partial_r2(comp, y, f) for f in ("shape", "texture")}
    bs = {"shape": [], "texture": []}
    idx = np.arange(len(comp))
    for _ in range(B_REGRESSION):
        s = rng.integers(0, len(comp), len(comp))
        sub, ys = [comp[i] for i in s], y[s]
        for f in bs:
            bs[f].append(partial_r2(sub, ys, f))
    ci = {f: np.percentile(bs[f], [2.5, 97.5]) for f in bs}
    shp = np.array([c["shape"] for c in comp])
    tex = np.array([c["texture"] for c in comp])
    pointy, round_ = y[np.isin(shp, ["V", "triangle"])], y[np.isin(shp, ["disc", "square"])]
    fur, other = y[np.isin(tex, ["fox_fur", "cat_fur"])], y[~np.isin(tex, ["fox_fur", "cat_fur"])]
    sp_lo, sp_hi = boot_ci(lambda a, b: a.mean() - b.mean(), [pointy, round_], rng)
    fu_lo, fu_hi = boot_ci(lambda a, b: a.mean() - b.mean(), [fur, other], rng)
    shape_wins = ci["shape"][0] > ci["texture"][1] and sp_lo > 0
    tex_wins = ci["texture"][0] > ci["shape"][1] and fu_lo > 0
    v2 = "H_V" if shape_wins else "routes/category" if tex_wins else "inconclusive"
    verdict["T2"] = v2
    cell = {}
    for c, val in zip(comp, y):
        cell.setdefault((c["texture"], c["shape"]), []).append(val)
    say(f"## Test 2: shape vs. texture\n\nn = {len(comp)} composites. Partial R²: shape **{pr['shape']:.3f}** "
        f"[{ci['shape'][0]:.3f}, {ci['shape'][1]:.3f}], texture **{pr['texture']:.3f}** "
        f"[{ci['texture'][0]:.3f}, {ci['texture'][1]:.3f}].\n\n"
        f"V+triangle − disc+square = {pointy.mean() - round_.mean():+.0f} [{sp_lo:+.0f}, {sp_hi:+.0f}]. "
        f"Fur − other textures = {fur.mean() - other.mean():+.0f} [{fu_lo:+.0f}, {fu_hi:+.0f}]. Verdict: **{v2}**.\n\n"
        "Mean max by texture × shape (reported, not part of the verdict):\n\n"
        "| texture | V | triangle | disc | square |\n| --- | --- | --- | --- | --- |\n" +
        "\n".join(f"| {t} | " + " | ".join(f"{np.mean(cell.get((t, s), [np.nan])):.0f}" for s in ("V", "triangle", "disc", "square")) + " |"
                  for t in sorted({c['texture'] for c in comp})) + "\n")

    # Test 3
    R, Bx = mn([r["stim_id"] for r in by["T3_round"]]), mn([r["stim_id"] for r in by["T3_boxy"]])
    d3 = R.mean() - Bx.mean()
    lo3, hi3 = boot_ci(lambda a, b: a.mean() - b.mean(), [R, Bx], rng)
    pass3a = d3 <= -50 and hi3 < 0
    pairs = {}
    for c in comp:
        if c["shape"] in ("disc", "square"):
            pairs.setdefault((c["texture"], c["src"], c["size"]), {})[c["shape"]] = act[c["id"]][1]
    pd = np.array([p["disc"] - p["square"] for p in pairs.values() if len(p) == 2])
    lo3b, hi3b = boot_ci(lambda a: a.mean(), [pd], rng)
    pass3b = hi3b < 0
    verdict["T3"] = "confirmed" if pass3a and pass3b else "not replicated" if not pass3a and not pass3b else "partial"
    say(f"## Test 3: round suppression\n\n3a (natural, spatial mean): round {R.mean():.0f} (n={len(R)}) vs. boxy "
        f"{Bx.mean():.0f} (n={len(Bx)}); difference **{d3:+.0f}** [{lo3:+.0f}, {hi3:+.0f}] → {'pass' if pass3a else 'fail'}.\n\n"
        f"3b (synthetic, disc − square, matched pairs n={len(pd)}): **{pd.mean():+.1f}** [{lo3b:+.1f}, {hi3b:+.1f}] → "
        f"{'pass' if pass3b else 'fail'}. Negative side: **{verdict['T3']}**.\n")

    # Test 4
    def effects(group):
        eff = []
        for r in by.get(group, []):
            if r["kind"] != "occlusion_original":
                continue
            sid = r["extra"]
            o, e, c = act[f"{sid}_original"][0], act[f"{sid}_ear"][0], act[f"{sid}_control"][0]
            if o <= 0:
                continue  # drop fraction undefined when the original doesn't drive the neuron
            eff.append((o - e) / o - (o - c) / o)
        return np.array(eff)
    up, fl = effects("T4_upright"), effects("T4_floppy")
    med = float(np.median(up)) if len(up) else np.nan
    lo4, hi4 = boot_ci(lambda a: np.median(a), [up], rng) if len(up) else (np.nan, np.nan)
    v4 = "H_V" if med >= 0.5 else "H_routes" if (0.1 <= med < 0.5 and lo4 > 0) else "H_category"
    verdict["T4"] = v4
    say(f"## Test 4: ear occlusion\n\nUpright-eared (verdict group, n={len(up)}): median effect **{med:+.3f}** "
        f"[{lo4:+.3f}, {hi4:+.3f}]. Verdict: **{v4}**.\n\nFloppy-eared (comparison, n={len(fl)}): median effect "
        f"{np.median(fl) if len(fl) else float('nan'):+.3f}.\n")

    # Test 5
    cars = by["T5_car"]
    q = mx([r["stim_id"] for r in cars if r["view"] == "three-quarter"])
    f5 = mx([r["stim_id"] for r in cars if r["view"] == "frontal"])
    d5 = q.mean() - f5.mean()
    lo5, hi5 = boot_ci(lambda a, b: a.mean() - b.mean(), [q, f5], rng)
    v5 = ("underpowered" if len(f5) < 15 else "replicates" if d5 >= 150 and lo5 > 0
          else "fails to replicate" if d5 <= 0 else "inconclusive")
    verdict["T5"] = v5
    say(f"## Test 5: car views\n\nThree-quarter {q.mean():.0f} (n={len(q)}) vs. head-on {f5.mean():.0f} (n={len(f5)}); "
        f"difference **{d5:+.0f}** [{lo5:+.0f}, {hi5:+.0f}]. Verdict: **{v5}**.\n")

    # Overall
    t1, t2, t4 = verdict["T1"], verdict["T2"], verdict["T4"]
    if t1 == t2 == t4 == "H_V":
        overall = "Cammarata's claim is supported (H_V)"
    elif t1 == "routes/category" and t2 == "routes/category" and t4 == "H_routes":
        overall = "H_routes: two category routes with a within-category shape preference"
    elif t1 == "routes/category" and t2 == "routes/category" and t4 == "H_category":
        overall = "H_category: the original cat + car reading"
    else:
        overall = "Mixed: see the individual tests"
    head = (f"# Phase B results{' (DRY RUN, FAKE DATA)' if 'DRYRUN' in str(out_dir) else ''}\n\n"
            f"**Overall: {overall}.**\n\nTest 1: {t1} · Test 2: {t2} · Test 4: {t4} · "
            f"Test 3 (negative side): {verdict['T3']} · Test 5 (replication): {verdict['T5']}\n\n")
    (out_dir / "report.md").write_text(head + "\n".join(L), encoding="utf-8")
    print(head + "\n".join(L))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dry = "--dry-run" in sys.argv
    out_dir = ROOT / "results" / ("phase_b_DRYRUN" if dry else "phase_b")
    if not dry and (out_dir / "activations.csv").exists():
        sys.exit("results/phase_b/activations.csv already exists: the confirmatory run happens once.")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "data" / "phase_b" / "final_stimuli.csv")))
    act = measure(rows, dry)
    with open(out_dir / "activations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stim_id", "group", "max_minus_gray", "mean_minus_gray"])
        for r in rows:
            w.writerow([r["stim_id"], r["group"], f"{act[r['stim_id']][0]:.4f}", f"{act[r['stim_id']][1]:.4f}"])
    analyze(rows, act, out_dir)


if __name__ == "__main__":
    main()
