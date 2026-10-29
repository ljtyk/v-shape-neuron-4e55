"""Re-rank non-pool val images by 4e:55 with cat/fox routes ablated (A1, A2), alongside the intact neuron.

4e:55 pre-ReLU = bias + sum_c w[c] * mixed4d[c], so the ablated versions are computed from the
same mixed4d maps with the chosen weights set to zero. Saves results/ablation/rerank.npz.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from ablation import ablation_sets  # noqa: E402
from imagenet_val import load, stimulus_pool  # noqa: E402
from neuron import CHANNEL, Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "ablation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = load()
    sets = ablation_sets(ds)
    held = {i for *_, i in stimulus_pool(ds)}
    keep = [i for i in range(len(ds)) if i not in held]
    ds.transform = Neuron.preprocess
    n = Neuron()
    conv = n.model.mixed4e_1x1_pre_relu_conv
    w = conv.weight[CHANNEL, :, 0, 0].detach()
    b = float(conv.bias[CHANNEL])
    ws = {"intact": w}
    for k, chans in sets.items():
        wk = w.clone()
        wk[chans] = 0
        ws[k] = wk
    W = torch.stack(list(ws.values()))  # (versions, 528)
    store = {}
    n.model.mixed4d.register_forward_hook(lambda m, i, o: store.__setitem__("d", o.detach()))
    out = {k: [] for k in ws}
    for bi, (x, _) in enumerate(DataLoader(Subset(ds, keep), batch_size=128, num_workers=4)):
        n.maps(x)
        maps = torch.einsum("vc,nchw->vnhw", W, store["d"]) + b
        for k, m in zip(ws, maps):
            out[k].append(m.amax(dim=(1, 2)).cpu())
        if bi % 100 == 0:
            print(f"batch {bi}", flush=True)
    np.savez(OUT / "rerank.npz", sample_idx=np.array(keep),
             **{k: torch.cat(v).numpy() for k, v in out.items()},
             **{f"chans_{k}": np.array(c) for k, c in sets.items()})
    print("saved", {k: len(c) for k, c in sets.items()})


if __name__ == "__main__":
    main()
