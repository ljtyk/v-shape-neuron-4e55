"""Tightening the V-vs-O result (maximize 4e:55 centre, pre-ReLU).

Predictions written before running:
  V detector: activation peaks at an acute opening angle and falls toward 180 deg;
              V (free angle) > two bars and V >> O.
  Not a V detector: flat or rising toward 180 deg; two bars >= V.

Conditions (each 128 restarts = 4 independent groups of 32, 400 Adam steps):
  V@theta for theta in 20, 45, 70, 90, 120, 150, 180 (opening fixed; position, orientation,
  arm lengths, widths, gray level free; 180 = straight bar)
  V_free, O, bars2 (as in head_to_head.py)
Score: centre activation averaged over the same 16 jitters. Saves per-restart scores to
results/tighten/scores.npz and best images.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import head_to_head as hh  # noqa: E402
from neuron import DEVICE  # noqa: E402
from simple_shapes import Center, jitter, save_sheet, to_pil  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "tighten"
GROUPS, PER, STEPS = 4, 32, 400
ANGLES = [20, 45, 70, 90, 120, 150, 180]


def fam_V_angle(theta):
    def make():
        g = lambda *s, sd: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
        P = {"v": g(PER, 2, sd=0.3), "phi": g(PER, sd=2.0), "len": g(PER, 2, sd=1),
             "w": g(PER, 2, sd=1), "gray": g(PER, sd=2)}
        half = math.radians(theta) / 2

        def r(P):
            v = torch.tanh(P["v"])
            L = 0.05 + 0.9 * torch.sigmoid(P["len"])
            dirs = torch.stack([P["phi"] + half, P["phi"] - half], 1)
            ends = v.unsqueeze(1) + L.unsqueeze(-1) * torch.stack([torch.cos(dirs), torch.sin(dirs)], -1)
            return hh.paint(hh.seg_sdf(v.unsqueeze(1).expand(-1, 2, -1), ends, hh.width(P["w"])), P["gray"])
        return P, r
    return make


def fam_V_angle_eq(theta):
    """Forced V: both arms the same length (0.3-0.9) and one shared thin width (0.02-0.10),
    so the optimizer can't shrink an arm to a stub or swell it into a blob."""
    def make():
        g = lambda *s, sd: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
        P = {"v": g(PER, 2, sd=0.3), "phi": g(PER, sd=2.0), "len": g(PER, sd=1),
             "w": g(PER, sd=1), "gray": g(PER, sd=2)}
        half = math.radians(theta) / 2

        def r(P):
            v = torch.tanh(P["v"]) * 0.6
            L = (0.3 + 0.6 * torch.sigmoid(P["len"])).view(-1, 1, 1)
            dirs = torch.stack([P["phi"] + half, P["phi"] - half], 1)
            ends = v.unsqueeze(1) + L * torch.stack([torch.cos(dirs), torch.sin(dirs)], -1)
            w = (0.02 + 0.08 * torch.sigmoid(P["w"])).unsqueeze(1).expand(-1, 2)
            return hh.paint(hh.seg_sdf(v.unsqueeze(1).expand(-1, 2, -1), ends, w), P["gray"])
        return P, r
    return make


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hh.N = PER  # head_to_head families build PER restarts per call
    if len(sys.argv) > 1 and sys.argv[1] == "forced":
        conds = {f"V{a}": fam_V_angle_eq(a) for a in ANGLES}
        run(conds, OUT / "forced")
        return
    conds = {f"V{a}": fam_V_angle(a) for a in ANGLES}
    conds.update({"V_free": hh.fam_V, "O": hh.fam_O, "bars2": hh.fam_bars2})
    run(conds, OUT)


def run(conds, out):
    out.mkdir(parents=True, exist_ok=True)
    f = Center()
    scores = {}
    for name, make in conds.items():
        vals, best_img, best_val = [], None, -1e9
        for grp in range(GROUPS):
            torch.manual_seed(1000 + grp)
            P, rend = make()
            opt = torch.optim.Adam(P.values(), lr=0.03)
            for _ in range(STEPS):
                loss = -f(jitter(rend(P))).mean() / 100
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                x = rend(P)
                torch.manual_seed(999)
                c = torch.stack([f(jitter(x)) for _ in range(16)]).mean(0).cpu().numpy()
            vals.append(c)
            k = int(np.argmax(c))
            if c[k] > best_val:
                best_val, best_img = float(c[k]), to_pil(x[k])
        vals = np.stack(vals)  # (GROUPS, PER)
        scores[name] = vals
        best_img.save(out / f"best_{name}.png")
        grp_best = vals.max(1)
        grp_top5 = np.sort(vals, 1)[:, -5:].mean(1)
        print(f"{name:>7}: best {vals.max():6.0f} | per-group best {grp_best.mean():6.0f} +- {grp_best.std():4.0f}"
              f" | per-group top-5 {grp_top5.mean():6.0f} +- {grp_top5.std():4.0f} | median {np.median(vals):6.0f}",
              flush=True)
    np.savez(out / "scores.npz", **scores)
    names = list(conds)
    from PIL import Image
    save_sheet([Image.open(out / f"best_{n}.png") for n in names],
               [f"{n}: {scores[n].max():.0f}" for n in names], out / "best_per_condition.png", cols=5)


if __name__ == "__main__":
    main()
