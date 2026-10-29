"""Route ablation for 4e:55: zero the weights from chosen mixed4d channels into channel 55 only.

A1: the fox route, 4d:493 (dominant input).
A2: every positive-weight input selective for cats/foxes but not cars
    (cat/fox z > 1 and car z < 1, z from mean spatial-max activation on non-pool val images).
    4d:493 is NOT in A2: it also responds to cars (car z = 1.27).
A3: A1 + A2, every cat/fox route.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CATFOX = ["tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat", "lynx",
          "red fox", "kit fox", "Arctic fox", "grey fox"]
CAR = ["sports car", "beach wagon", "convertible", "cab", "minivan", "limousine", "pickup",
       "police van", "ambulance", "minibus", "jeep", "racer"]


def ablation_sets(ds):
    from imagenet_val import class_index
    idx = class_index(ds)
    a = np.load(ROOT / "results" / "explore" / "acts.npz")
    cls, d = a["cls"], a["d_max"].astype(np.float32)
    z = lambda names: (d[np.isin(cls, [idx[n] for n in names])].mean(0) - d.mean(0)) / (d.std(0) + 1e-6)  # noqa: E731
    z_cf, z_car = z(CATFOX), z(CAR)
    w = {int(r["channel"]): float(r["weight"]) for r in csv.DictReader(open(ROOT / "results" / "explore" / "inputs.csv"))}
    a2 = sorted(c for c in range(528) if w[c] > 0 and z_cf[c] > 1 and z_car[c] < 1)
    return {"A1": [493], "A2": a2, "A3": sorted(set(a2) | {493})}
