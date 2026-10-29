"""Does the fox detector (4d:493) respond to cars because of a projecting, snout-like bonnet?

Prediction (written before running): three-quarter front views drive 4d:493 more than head-on,
side or rear views, and on cars its peak sits on the bonnet / front corner.

Uses the 120 randomly sampled car images and the view labels in results/car_view/views.txt
(made before measurement). Measures 4d:493 spatial max, its contribution to 4e:55 at the
4e:55 peak, and 4e:55 max. Bootstrap 95% CIs for differences in group means.
Also overlays the 4d:493 map on the top 3q cars and on the top foxes.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import class_index, load, stimulus_pool  # noqa: E402
from neuron import CHANNEL, Neuron  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "car_view"
FOX = 493


def overlay(img, m):
    x = Neuron.preprocess(img).permute(1, 2, 0).numpy()
    tile = Image.fromarray((x * 255).astype("uint8"))
    mm = np.clip(m / max(m.max(), 1e-6), 0, 1)
    heat = Image.fromarray((mm * 255).astype("uint8")).resize((224, 224), Image.BILINEAR)
    tile = Image.composite(Image.new("RGB", (224, 224), (0, 200, 255)), tile, heat.point(lambda v: int(v * 0.6)))
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    cy, cx = (iy + 0.5) * 224 / m.shape[0], (ix + 0.5) * 224 / m.shape[1]
    ImageDraw.Draw(tile).ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=(255, 255, 0), width=3)
    return tile


def main():
    ds = load()
    n = Neuron()
    w = n.model.mixed4e_1x1_pre_relu_conv.weight[CHANNEL, :, 0, 0].detach().cpu().numpy()
    store = {}
    n.model.mixed4d.register_forward_hook(lambda m, i, o: store.__setitem__("d", o.detach()))

    def run(imgs):
        x = torch.stack([Neuron.preprocess(im) for im in imgs])
        e = n.maps(x)
        fox = store["d"][:, FOX].cpu().numpy()
        flat = e.flatten(1).argmax(1)
        yy, xx = flat // e.shape[2], flat % e.shape[2]
        fox_at_peak = fox[np.arange(len(x)), yy.numpy(), xx.numpy()] * w[FOX]
        return fox.reshape(len(x), -1).max(1), fox_at_peak, e.flatten(1).max(1).values.numpy(), fox

    sample = list(csv.DictReader(open(OUT / "sample.csv")))
    views = {int(l.split()[0]): l.split()[1] for l in open(OUT / "views.txt") if not l.startswith("#")}
    imgs = [Image.open(ds.samples[int(r["sample_idx"])][0]).convert("RGB") for r in sample]
    fmax, fpeak, emax, fmaps = run(imgs)
    v = np.array([views[k] for k in range(len(sample))])

    lines = ["Fox detector 4d:493 on 120 random cars, by view (labels made before measurement)", ""]
    lines.append(f"  {'view':<8}{'n':>4}{'4d:493 max':>12}{'493 -> 4e:55':>14}{'4e:55 max':>11}")
    for g in ("front", "3q", "side", "rear", "unclear"):
        s = v == g
        lines.append(f"  {g:<8}{s.sum():>4}{fmax[s].mean():>12.1f}{fpeak[s].mean():>14.0f}{emax[s].mean():>11.0f}")
    rng = np.random.default_rng(0)

    def diff(a, b, arr, label):
        A, B = arr[v == a], arr[v == b]
        bs = [rng.choice(A, len(A)).mean() - rng.choice(B, len(B)).mean() for _ in range(10000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"  {label:<34}{A.mean() - B.mean():+8.1f}  [{lo:+7.1f}, {hi:+7.1f}]")

    lines += ["", "Differences in means, 95% bootstrap CI (prediction: 3q higher than each)"]
    for other in ("front", "side", "rear"):
        diff("3q", other, fmax, f"4d:493 max: 3q - {other}")
    for other in ("front", "side", "rear"):
        diff("3q", other, emax, f"4e:55 max: 3q - {other}")
    print("\n".join(lines))
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    # where does 4d:493 peak? top 3q cars and, for comparison, top foxes
    top3q = [k for k in np.argsort(-fmax) if v[k] == "3q"][:12]
    tiles = [overlay(imgs[k], fmaps[k]) for k in top3q]
    idx = class_index(ds)
    held = {i for *_, i in stimulus_pool(ds)}
    foxes = [i for i, (_, t) in enumerate(ds.samples) if t in {idx["red fox"], idx["kit fox"]} and i not in held]
    fimgs = [Image.open(ds.samples[i][0]).convert("RGB") for i in foxes]
    ffmax, _, _, ffmaps = run(fimgs)
    tiles += [overlay(fimgs[k], ffmaps[k]) for k in np.argsort(-ffmax)[:6]]
    sheet = Image.new("RGB", (6 * 224, 3 * 224), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % 6) * 224, (k // 6) * 224))
    sheet.save(OUT / "fox_detector_peaks_cars_then_foxes.jpg", quality=90)
    lines2 = f"fox images: 4d:493 max mean {ffmax.mean():.1f} (cars overall {fmax.mean():.1f})"
    print(lines2)
    with open(OUT / "report.txt", "a", encoding="utf-8") as fh:
        fh.write("\n" + lines2 + "\n")


if __name__ == "__main__":
    main()
