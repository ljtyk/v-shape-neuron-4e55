"""Procedure step 2: confirm channel 55 is the published cat-face / car-front neuron.

Runs every ImageNet val image EXCEPT the held-out stimulus pool, saves the top 100,
and writes a contact sheet of the top 20 for the human pass/fail check
(pass = at least 12 of 20 are cat faces or car fronts).
"""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import load, stimulus_pool  # noqa: E402
from neuron import Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "verification"
CAT = {"tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat"}
CAR = {"sports car", "beach wagon", "convertible", "cab", "racer", "minivan", "limousine",
       "jeep", "Model T", "pickup", "car wheel", "grille", "station wagon", "police van",
       "ambulance", "moving van", "minibus", "go-kart", "tow truck"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load()
    held_out = {i for _, _, i in stimulus_pool(ds)}
    keep = [i for i in range(len(ds)) if i not in held_out]
    ds.transform = Neuron.preprocess
    loader = DataLoader(Subset(ds, keep), batch_size=128, num_workers=4)
    n = Neuron()
    scores = []
    for b, (x, _) in enumerate(loader):
        scores.append(n.maps(x).amax(dim=(1, 2)))
        if b % 50 == 0:
            print(f"batch {b}/{len(loader)}", flush=True)
    scores = torch.cat(scores)
    order = scores.argsort(descending=True)[:100]

    rows = []
    for rank, j in enumerate(order.tolist(), 1):
        path, t = ds.samples[keep[j]]
        name = ds.classes[t][0]
        kind = "cat" if name in CAT else "car" if name in CAR else ""
        rows.append(dict(rank=rank, file=Path(path).name, cls=name, family_by_class=kind,
                         max_act=round(float(scores[j]), 4)))
    with open(OUT / "top100.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    sheet = Image.new("RGB", (5 * 224, 4 * 224), "white")
    for k, j in enumerate(order[:20].tolist()):
        img = Image.open(ds.samples[keep[j]][0])
        sheet.paste(Image.fromarray((Neuron.preprocess(img).permute(1, 2, 0).numpy() * 255).astype("uint8")),
                    ((k % 5) * 224, (k // 5) * 224))
    sheet.save(OUT / "top20_sheet.jpg", quality=92)

    top20 = rows[:20]
    by_class = sum(1 for r in top20 if r["family_by_class"])
    print(f"Ran {len(keep)} images ({len(held_out)} stimulus-pool images held out).")
    print(f"Top 20 by class label: {by_class} cat/car classes. "
          f"Pass/fail needs a human look at top20_sheet.jpg (cat FACES / car FRONTS >= 12).")
    for r in top20:
        print(f"  {r['rank']:>2}  {r['max_act']:8.3f}  {r['cls']}")


if __name__ == "__main__":
    main()
