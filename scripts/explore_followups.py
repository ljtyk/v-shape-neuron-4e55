"""Phase A follow-ups (non-pool images only).

1. Most negative 4d inputs to 4e:55: what do they fire on?
2. Persian cats: sheet of the highest and lowest Persian images with peaks.
3. Crop-scale test: for pointed-object classes and control classes, measure 4e:55 on the
   full image vs. a tight square crop around the largest object box.
"""
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_labeling import square_crop  # noqa: E402
from explore_analyze import sheet, top_classes  # noqa: E402
from imagenet_val import class_index, load, object_boxes, val_id  # noqa: E402
from neuron import Neuron  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "explore"
POINTED = ["obelisk", "church", "spindle", "volcano", "candle", "bell cote", "mosque",
           "mountain tent", "bullet train", "airliner", "warplane", "space shuttle"]
CONTROL = ["Persian cat", "tabby", "red fox", "beach wagon", "sports car",
           "golf ball", "wall clock", "orange", "brown bear", "hare"]


def main():
    ds = load()
    a = dict(np.load(OUT / "acts.npz"))
    cls, lines = a["cls"], []
    say = lambda s="": (print(s, flush=True), lines.append(s))  # noqa: E731

    # 1. Negative inputs
    ranked = list(csv.DictReader(open(OUT / "inputs.csv")))
    say("== Most negative 15 mixed4d inputs (what each fires on) ==")
    for r in ranked[::-1][:15]:
        c = int(r["channel"])
        say(f"  #{r['rank']:>3} mixed4d:{c:<4} w={float(r['weight']):+.3f}  "
            f"{top_classes(ds, cls, a['d_max'][:, c].astype(np.float32))}")
    say()

    # 2. Persian
    idx = class_index(ds)
    per = np.where(cls == idx["Persian cat"])[0]
    per = per[np.argsort(-a["e55_max"][per])]
    sheet(ds, a, list(per[:12]) + list(per[-12:]), OUT / "persian_top12_bottom12.jpg", overlay=True)
    say(f"== Persian cat ({len(per)} non-pool images): max act top {a['e55_max'][per[0]]:.0f}, "
        f"median {np.median(a['e55_max'][per]):.0f}, bottom {a['e55_max'][per[-1]]:.0f} ==")
    say("  sheet: persian_top12_bottom12.jpg (rows 1-2 highest, rows 3-4 lowest)")
    say()

    # 3. Crop-scale test
    n = Neuron()
    boxes = object_boxes()
    pos = {i: j for j, i in enumerate(a["sample_idx"])}
    say("== Crop-scale test: 4e:55 max, full image -> tight object crop (median over class) ==")
    say(f"  {'class':<16}{'n':>4}{'full':>8}{'crop':>8}{'ratio':>7}  {'obj/img area':>12}")
    rows = []
    for group, names in (("pointed", POINTED), ("control", CONTROL)):
        for name in names:
            if name not in idx:
                say(f"  (class not found: {name})")
                continue
            js = np.where(cls == idx[name])[0]
            full, crops, areas = [], [], []
            for j in js:
                path = ds.samples[a["sample_idx"][j]][0]
                img = Image.open(path).convert("RGB")
                b = max(boxes[val_id(path)], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                areas.append((b[2] - b[0]) * (b[3] - b[1]) / (img.size[0] * img.size[1]))
                crops.append(img.crop(square_crop(b, *img.size)))
                full.append(a["e55_max"][j])
            cm = np.array([m[0] for m in n.measure(crops)])
            f_med, c_med = np.median(full), np.median(cm)
            rows.append(dict(group=group, cls=name, n=len(js), full=f_med, crop=c_med,
                             area=np.median(areas)))
            say(f"  {name:<16}{len(js):>4}{f_med:>8.0f}{c_med:>8.0f}{c_med / f_med:>7.2f}  "
                f"{np.median(areas):>12.2f}   [{group}]")
    with open(OUT / "crop_scale.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    # Where would the cropped pointed classes land in the full-image class ranking?
    prof = list(csv.DictReader(open(OUT / "class_profile.csv")))
    means = np.array([float(r["mean_max_act"]) for r in prof])
    say()
    say("  cropped median placed against full-image class means (rank of 1000):")
    for r in rows:
        say(f"    {r['cls']:<16} full-image class rank -> cropped would rank #{int((means > r['crop']).sum()) + 1}")
    (OUT / "followups.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
