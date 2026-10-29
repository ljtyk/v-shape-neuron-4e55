"""Is the fox input's share on cars driven by colour? Compare colour vs grayscale images.

For the top-quarter car and cat/fox images (same selection as car_route.py), measure at the
4e:55 peak: total activation, the fox input (4d:493) contribution, and the car-route
contribution, in colour and in grayscale.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from ablation import CAR, CATFOX, ablation_sets  # noqa: E402
from imagenet_val import class_index, load, stimulus_pool  # noqa: E402
from neuron import CHANNEL, Neuron  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    ds = load()
    idx = class_index(ds)
    held = {i for *_, i in stimulus_pool(ds)}
    a = np.load(ROOT / "results" / "explore" / "acts.npz")
    cls_all, d = a["cls"], a["d_max"].astype(np.float32)
    z = lambda names: (d[np.isin(cls_all, [idx[n] for n in names])].mean(0) - d.mean(0)) / (d.std(0) + 1e-6)  # noqa: E731
    z_cf, z_car = z(CATFOX), z(CAR)
    n = Neuron()
    w = n.model.mixed4e_1x1_pre_relu_conv.weight[CHANNEL, :, 0, 0].detach().cpu().numpy()
    catfox = set(ablation_sets(ds)["A3"])
    car = sorted(c for c in range(528) if w[c] > 0 and z_car[c] > 1 and z_cf[c] < 1)
    store = {}
    n.model.mixed4d.register_forward_hook(lambda m, i, o: store.__setitem__("d", o.detach()))

    def measure(imgs):
        x = torch.stack([Neuron.preprocess(im) for im in imgs])
        m = n.maps(x)
        flat = m.flatten(1).argmax(1)
        yy, xx = flat // m.shape[2], flat % m.shape[2]
        dd = store["d"][torch.arange(len(x)), :, yy, xx].cpu().numpy() * w
        return m.flatten(1).max(1).values.numpy(), dd[:, 493], dd[:, car].sum(1), dd[:, sorted(catfox)].sum(1)

    lines = []
    for label, names in (("cars", CAR), ("cats/foxes", CATFOX)):
        js = [i for i, (_, t) in enumerate(ds.samples) if t in {idx[x] for x in names} and i not in held]
        col = [Image.open(ds.samples[i][0]).convert("RGB") for i in js]
        act = np.concatenate([measure(col[k:k + 64])[0] for k in range(0, len(col), 64)])
        top = np.argsort(-act)[: max(10, len(act) // 4)]
        sel = [col[i] for i in top]
        gray = [ImageOps.grayscale(im).convert("RGB") for im in sel]
        rc = [np.concatenate(v) for v in zip(*[measure(sel[k:k + 64]) for k in range(0, len(sel), 64)])]
        rg = [np.concatenate(v) for v in zip(*[measure(gray[k:k + 64]) for k in range(0, len(gray), 64)])]
        names_ = ["activation", "fox input 4d:493", "car route", "all cat/fox routes"]
        lines.append(f"== {label} (top {len(top)}): median colour -> grayscale ==")
        for nm, c, g in zip(names_, rc, rg):
            lines.append(f"  {nm:<20} {np.median(c):7.0f} -> {np.median(g):7.0f}   ({np.median(g) / max(np.median(c), 1e-6) * 100:5.1f}%)")
    print("\n".join(lines))
    (ROOT / "results" / "car_route" / "colour.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
