"""Why do cars drive 4e:55? Split its peak activation into per-input contributions.

For every non-pool image in the car classes and the cat/fox classes:
  4e:55 pre-ReLU at its peak location = bias + sum_c w[c] * mixed4d[c, peak]
Contributions are grouped by what each 4d input is selective for (z-scores from Phase A):
  catfox  = the A3 ablation set (cat/fox routes, incl. the fox input 4d:493)
  car     = positive-weight inputs with car z > 1 and cat/fox z < 1
  other   = everything else (positive and negative)
Also saves a sheet of the top car images with the peak circled.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ablation import CAR, CATFOX, ablation_sets  # noqa: E402
from explore_analyze import sheet  # noqa: E402
from imagenet_val import class_index, load, stimulus_pool  # noqa: E402
from neuron import CHANNEL, DEVICE, Neuron  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "car_route"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load()
    idx = class_index(ds)
    held = {i for *_, i in stimulus_pool(ds)}
    a = np.load(ROOT / "results" / "explore" / "acts.npz")
    cls_all, d = a["cls"], a["d_max"].astype(np.float32)
    z = lambda names: (d[np.isin(cls_all, [idx[n] for n in names])].mean(0) - d.mean(0)) / (d.std(0) + 1e-6)  # noqa: E731
    z_cf, z_car = z(CATFOX), z(CAR)

    n = Neuron()
    conv = n.model.mixed4e_1x1_pre_relu_conv
    w = conv.weight[CHANNEL, :, 0, 0].detach().cpu().numpy()
    b = float(conv.bias[CHANNEL])
    catfox = set(ablation_sets(ds)["A3"])
    car = {c for c in range(528) if w[c] > 0 and z_car[c] > 1 and z_cf[c] < 1}
    groups = {"catfox": sorted(catfox), "car": sorted(car),
              "other": [c for c in range(528) if c not in catfox | car]}
    print(f"input groups: catfox {len(groups['catfox'])}, car {len(groups['car'])}, other {len(groups['other'])}")
    print(f"car-route inputs: {groups['car']}")

    store = {}
    n.model.mixed4d.register_forward_hook(lambda m, i, o: store.__setitem__("d", o.detach()))
    lines = []
    for label, names in (("cars", CAR), ("cats/foxes", CATFOX)):
        js = [i for i, (_, t) in enumerate(ds.samples) if t in {idx[x] for x in names} and i not in held]
        contrib, peaks, acts = [], [], []
        for k in range(0, len(js), 64):
            x = torch.stack([Neuron.preprocess(Image.open(ds.samples[i][0])) for i in js[k:k + 64]])
            m = n.maps(x)                                     # (B, 14, 14)
            flat = m.flatten(1).argmax(1)
            yy, xx = flat // m.shape[2], flat % m.shape[2]
            dd = store["d"][torch.arange(len(x)), :, yy, xx].cpu().numpy()   # (B, 528) at peak
            contrib.append(dd * w)
            acts.append(m.flatten(1).max(1).values.numpy())
        contrib, acts = np.concatenate(contrib), np.concatenate(acts)
        top = np.argsort(-acts)[: max(10, len(acts) // 4)]  # top quarter: when it fires well
        say = lambda s: (print(s), lines.append(s))  # noqa: E731
        say(f"\n== {label}: {len(js)} images, using top {len(top)} by activation (median peak {np.median(acts[top]):.0f}) ==")
        tot = contrib[top].sum(1) + b
        for g, chans in groups.items():
            gsum = contrib[top][:, chans].sum(1)
            pos = np.clip(contrib[top][:, chans], 0, None).sum(1)
            say(f"  {g:>6}: net {np.median(gsum):7.0f}  (positive part {np.median(pos):6.0f})  "
                f"= {np.median(gsum / tot) * 100:5.1f}% of activation")
        mean_c = contrib[top].mean(0)
        say("  top 8 contributing inputs: " + ", ".join(
            f"4d:{c} ({'catfox' if c in catfox else 'car' if c in car else 'other'}) {mean_c[c]:.0f}"
            for c in np.argsort(-mean_c)[:8]))
        say(f"  fox input 4d:493 alone: median {np.median(contrib[top][:, 493]):.0f}")
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")

    # where on the car does it peak?
    acts_npz = dict(a)
    car_rows = np.where(np.isin(cls_all, [idx[x] for x in CAR]))[0]
    best = car_rows[np.argsort(-a["e55_max"][car_rows])[:30]]
    sheet(ds, acts_npz, list(best), OUT / "top30_cars_peaks.jpg", overlay=True)


if __name__ == "__main__":
    main()
