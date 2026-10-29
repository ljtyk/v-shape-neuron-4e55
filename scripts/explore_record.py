"""Phase A: one pass over every NON-pool val image, saving what the exploratory analyses need.

Saves results/explore/acts.npz:
  sample_idx, cls        dataset index and class index per image
  e55_max, e55_mean      4e:55 pre-ReLU max / mean over space
  e55_map                4e:55 pre-ReLU 14x14 maps (float16)
  d_max                  mixed4d (post-ReLU concat, 528 ch) spatial max per channel (float16)
  a5_max                 mixed5a (post-ReLU concat, 832 ch) spatial max per channel (float16)
The held-out stimulus pool is never run.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import load, stimulus_pool  # noqa: E402
from neuron import Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "explore"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load()
    held = {i for *_, i in stimulus_pool(ds)}
    keep = [i for i in range(len(ds)) if i not in held]
    ds.transform = Neuron.preprocess
    n = Neuron()
    acts = {}
    for layer in ("mixed4d", "mixed5a"):
        getattr(n.model, layer).register_forward_hook(
            lambda m, i, o, l=layer: acts.__setitem__(l, o.detach()))
    maps, d_max, a5_max = [], [], []
    loader = DataLoader(Subset(ds, keep), batch_size=128, num_workers=4)
    for b, (x, _) in enumerate(loader):
        maps.append(n.maps(x).half())
        d_max.append(acts["mixed4d"].amax(dim=(2, 3)).half().cpu())
        a5_max.append(acts["mixed5a"].amax(dim=(2, 3)).half().cpu())
        if b % 50 == 0:
            print(f"batch {b}/{len(loader)}", flush=True)
    maps = torch.cat(maps).float()
    np.savez_compressed(
        OUT / "acts.npz",
        sample_idx=np.array(keep),
        cls=np.array([ds.samples[i][1] for i in keep]),
        e55_max=maps.amax(dim=(1, 2)).numpy(),
        e55_mean=maps.mean(dim=(1, 2)).numpy(),
        e55_map=maps.half().numpy(),
        d_max=torch.cat(d_max).numpy(),
        a5_max=torch.cat(a5_max).numpy(),
    )
    print(f"saved {len(keep)} images to {OUT / 'acts.npz'}")


if __name__ == "__main__":
    main()
