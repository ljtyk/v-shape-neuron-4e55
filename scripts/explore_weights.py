"""Phase A, weights only: 4e:55's inputs from mixed4d and its readers in mixed5a.

Inputs: 4e:55 is in mixed4e's 1x1 branch, so its weights are one number per mixed4d channel.
Readers: effective weight from mixed4e channel 55 to each mixed5a concat channel.
  1x1 branch (0-255):    w[c, 55]
  3x3 branch (256-575):  sum over space of W3x3[c] . Wbottleneck[:, 55]   (bottleneck ReLU ignored)
  5x5 branch (576-703):  same through the 5x5 bottleneck
  pool branch (704-831): w_poolreduce[c, 55]  (after a 3x3 max-pool)
Writes results/explore/inputs.csv and readers.csv.
"""
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from neuron import CHANNEL, Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "explore"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = Neuron().model
    with torch.no_grad():
        w_in = m.mixed4e_1x1_pre_relu_conv.weight[CHANNEL, :, 0, 0].cpu()
        bias = float(m.mixed4e_1x1_pre_relu_conv.bias[CHANNEL])

        def through(bottleneck, conv):
            wb = bottleneck.weight[:, CHANNEL, 0, 0]          # (k,)
            return torch.einsum("ckhw,k->c", conv.weight, wb).cpu()

        readers = torch.cat([
            m.mixed5a_1x1_pre_relu_conv.weight[:, CHANNEL, 0, 0].cpu(),
            through(m.mixed5a_3x3_bottleneck_pre_relu_conv, m.mixed5a_3x3_pre_relu_conv),
            through(m.mixed5a_5x5_bottleneck_pre_relu_conv, m.mixed5a_5x5_pre_relu_conv),
            m.mixed5a_pool_reduce_pre_relu_conv.weight[:, CHANNEL, 0, 0].cpu(),
        ])
    branch = ["1x1"] * 256 + ["3x3"] * 320 + ["5x5"] * 128 + ["pool"] * 128

    def dump(name, w, extra=None):
        order = w.argsort(descending=True).tolist()
        with open(OUT / name, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "channel", "weight", "z"] + (["branch"] if extra else []))
            z = (w - w.mean()) / w.std()
            for r, c in enumerate(order, 1):
                wr.writerow([r, c, f"{float(w[c]):.5f}", f"{float(z[c]):.2f}"] + ([extra[c]] if extra else []))
        return order

    ins = dump("inputs.csv", w_in)
    outs = dump("readers.csv", readers, branch)
    print(f"4e:55 bias {bias:.3f}; input weights mean {w_in.mean():.4f} sd {w_in.std():.4f}")
    print("top 12 inputs (4d):   ", [(c, round(float(w_in[c]), 4)) for c in ins[:12]])
    print("most negative inputs: ", [(c, round(float(w_in[c]), 4)) for c in ins[::-1][:8]])
    print("essay's inputs rank:  ", {c: ins.index(c) + 1 for c in (493, 477, 201)}, "of", len(ins))
    print("top 12 readers (5a):  ", [(c, branch[c], round(float(readers[c]), 4)) for c in outs[:12]])
    print("essay's readers rank: ", {c: outs.index(c) + 1 for c in (94, 352, 317)}, "of", len(outs))


if __name__ == "__main__":
    main()
