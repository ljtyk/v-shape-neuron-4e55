"""Checkpoint sanity check: do the essay's named neighbour neurons show their stated content?

If 4d:493 is foxes, 4d:201 vans, 4c:447 cars, the weights and concat channel numbering
match the Distill / Goodfire indexing. Runs every other non-pool val image.
"""
import collections
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import load, stimulus_pool  # noqa: E402
from neuron import Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "verification"
CHECK = {("mixed4c", 447): "car (essay)", ("mixed4d", 493): "red fox (essay)",
         ("mixed4d", 201): "van, head-on (essay)", ("mixed4d", 477): "white wolf (essay)",
         ("mixed4e", 55): "this neuron, via concat"}


def main():
    ds = load()
    held = {i for *_, i in stimulus_pool(ds)}
    keep = [i for i in range(len(ds)) if i not in held][::2]
    ds.transform = Neuron.preprocess
    n = Neuron()
    acts = {}
    for layer in {l for l, _ in CHECK}:
        getattr(n.model, layer).register_forward_hook(
            lambda m, i, o, l=layer: acts.__setitem__(l, o.detach()))
    scores = collections.defaultdict(list)
    for x, _ in DataLoader(Subset(ds, keep), batch_size=128, num_workers=4):
        n.maps(x)
        for l, c in CHECK:
            scores[(l, c)].append(acts[l][:, c].amax(dim=(1, 2)).cpu())
    rows = []
    for (l, c), desc in CHECK.items():
        s = torch.cat(scores[(l, c)])
        top = s.argsort(descending=True)[:20].tolist()
        cls = collections.Counter(ds.classes[ds.samples[keep[j]][1]][0] for j in top)
        print(f"{l}:{c:<4} [{desc}] top20: {cls.most_common(6)}", flush=True)
        rows += [dict(neuron=f"{l}:{c}", rank=r + 1, cls=ds.classes[ds.samples[keep[j]][1]][0],
                      file=Path(ds.samples[keep[j]][0]).name) for r, j in enumerate(top)]
    with open(OUT / "neighbors_top20.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
