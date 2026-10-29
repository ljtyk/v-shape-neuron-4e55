"""Phase A analyses on results/explore/acts.npz (non-pool images only).

1. Identify 4e:55's top weighted inputs (4d) and readers (5a) by their top classes.
2. Bottom of 4e:55's ranking (by max and by mean): is the negative end round things?
3. Per-class profile: mean of 4e:55's max over each class's images.
4. Spatial peaks: top 30 images with the 14x14 map overlaid.
Writes text report results/explore/report.txt and sheets in results/explore/.
"""
import collections
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import load  # noqa: E402
from neuron import Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "explore"
ESSAY_IN = {493: "red fox", 477: "white wolf", 201: "van, head-on"}
ESSAY_OUT = {94: "candle flame", 352: "coyote face", 317: "martini glass"}


def name(ds, cls_idx):
    return ds.classes[cls_idx][0]


def top_classes(ds, cls, scores, k=20, n=5):
    top = np.argsort(-scores)[:k]
    return collections.Counter(name(ds, cls[j]) for j in top).most_common(n)


def sheet(ds, a, rows, path, overlay=False, cols=6, size=160):
    img = Image.new("RGB", (cols * size, ((len(rows) + cols - 1) // cols) * size), "white")
    for k, j in enumerate(rows):
        im = Image.open(ds.samples[a["sample_idx"][j]][0])
        x = Neuron.preprocess(im).permute(1, 2, 0).numpy()
        tile = Image.fromarray((x * 255).astype("uint8"))
        if overlay:
            m = a["e55_map"][j].astype(np.float32)
            m = np.clip((m - 0) / max(m.max(), 1e-6), 0, 1)
            heat = Image.fromarray((m * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
            red = Image.new("RGB", (224, 224), (255, 40, 0))
            tile = Image.composite(red, tile, heat.point(lambda v: int(v * 0.55)))
            iy, ix = np.unravel_index(np.argmax(a["e55_map"][j]), m.shape)
            cy, cx = (iy + 0.5) * 224 / m.shape[0], (ix + 0.5) * 224 / m.shape[1]
            ImageDraw.Draw(tile).ellipse([cx - 6, cy - 6, cx + 6, cy + 6], outline=(255, 255, 0), width=3)
        img.paste(tile.resize((size, size)), ((k % cols) * size, (k // cols) * size))
    img.save(path, quality=90)


def main():
    ds = load()
    a = dict(np.load(OUT / "acts.npz"))
    cls = a["cls"]
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731

    # 1. Inputs and readers
    for fname, key, essay, label in (("inputs.csv", "d_max", ESSAY_IN, "mixed4d"),
                                     ("readers.csv", "a5_max", ESSAY_OUT, "mixed5a")):
        ranked = list(csv.DictReader(open(OUT / fname)))
        say(f"== Top 15 {label} by weight with 4e:55 (what each one fires on) ==")
        for r in ranked[:15]:
            c = int(r["channel"])
            tag = f"  <- essay: {essay[c]}" if c in essay else ""
            extra = f" [{r['branch']}]" if "branch" in r else ""
            say(f"  #{r['rank']:>3} {label}:{c:<4}{extra} w={float(r['weight']):+.3f}  "
                f"{top_classes(ds, cls, a[key][:, c].astype(np.float32))}{tag}")
        for c, desc in essay.items():
            r = next(x for x in ranked if int(x["channel"]) == c)
            if int(r["rank"]) > 15:
                say(f"  #{r['rank']:>3} {label}:{c:<4} w={float(r['weight']):+.3f}  "
                    f"{top_classes(ds, cls, a[key][:, c].astype(np.float32))}  <- essay: {desc}")
        say()

    # 2. Negative end
    for key in ("e55_max", "e55_mean"):
        s = a[key]
        say(f"== Bottom 40 by {key} (most negative) ==")
        say(f"  {top_classes(ds, cls, -s, k=40, n=15)}")
        say(f"  top 40 by {key}: {top_classes(ds, cls, s, k=40, n=15)}")
    sheet(ds, a, np.argsort(a["e55_max"])[:30], OUT / "bottom30_by_max.jpg")
    sheet(ds, a, np.argsort(a["e55_mean"])[:30], OUT / "bottom30_by_mean.jpg")
    say()

    # 3. Per-class profile
    prof = collections.defaultdict(list)
    for j, c in enumerate(cls):
        prof[c].append(a["e55_max"][j])
    rows = sorted(((np.mean(v), name(ds, c)) for c, v in prof.items()), reverse=True)
    with open(OUT / "class_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "class", "mean_max_act"])
        for r, (m, n) in enumerate(rows, 1):
            w.writerow([r, n, f"{m:.2f}"])
    say("== Per-class mean of max activation ==")
    say("  top 40:    " + ", ".join(f"{n} {m:.0f}" for m, n in rows[:40]))
    say("  bottom 40: " + ", ".join(f"{n} {m:.0f}" for m, n in rows[-40:]))
    allm = np.array([m for m, _ in rows])
    say(f"  class means: median {np.median(allm):.0f}, 90th pct {np.percentile(allm, 90):.0f}, "
        f"max {allm.max():.0f}, min {allm.min():.0f}")
    probe = ["tabby", "Persian cat", "red fox", "Arctic fox", "lesser panda", "papillon", "pug",
             "brown bear", "ice bear", "otter", "badger", "meerkat", "hare", "wood rabbit",
             "fruit bat", "gazelle", "impala", "sports car", "beach wagon", "minivan", "cab",
             "Model T", "grille", "bullet train", "church", "obelisk", "candle", "volcano",
             "spindle", "golf ball", "tennis ball", "wall clock", "analog clock", "orange", "plate"]
    rank_of = {n: (r, m) for r, (m, n) in enumerate(rows, 1)}
    say("  probes (rank of 1000, mean max): " + ", ".join(
        f"{p} #{rank_of[p][0]} ({rank_of[p][1]:.0f})" for p in probe if p in rank_of))
    say()

    # 4. Spatial peaks
    top30 = np.argsort(-a["e55_max"])[:30]
    sheet(ds, a, top30, OUT / "top30_peaks.jpg", overlay=True)
    say("Sheets: top30_peaks.jpg (peak circled), bottom30_by_max.jpg, bottom30_by_mean.jpg")
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
