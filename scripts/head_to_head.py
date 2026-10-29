"""Head-to-head: best possible V vs best possible O vs free shapes, for 4e:55 (pre-ReLU).

Same setup for every family: grayscale figure on mid-gray, one gray level per figure,
soft edges, centre-position objective with jitter, 32 restarts x 400 Adam steps,
maximize and minimize.

Families
  V        two thick segments sharing a vertex (vertex, both arm ends, both widths free)
  O        one ellipse; fill blends ring <-> disc; radii, rotation, thickness free
  bars2    two thick segments with independent endpoints (same budget as V, no shared vertex)
  free5    reference: 3 segments + 2 ellipses with per-primitive gray and opacity
           (the earlier search, rerun here under identical settings)

Scoring (no gradients): centre activation averaged over 16 fixed jitters, and the
plain spatial max on the unjittered image, as used everywhere else in the study.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from neuron import DEVICE, Neuron  # noqa: E402
from simple_shapes import S, SIG, Center, init_prims, jitter, render, save_sheet, to_pil  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "head_to_head"
N, STEPS = 32, 400
lin = torch.linspace(-1 + 1 / S, 1 - 1 / S, S, device=DEVICE)
yy, xx = torch.meshgrid(lin, lin, indexing="ij")
GRID = torch.stack([xx, yy], -1).view(1, 1, S, S, 2)


def seg_sdf(a, b, w):
    """a, b: (n, k, 2) in [-1,1]; w: (n, k). Returns (n, k, S, S)."""
    a, b = a.view(*a.shape[:2], 1, 1, 2), b.view(*b.shape[:2], 1, 1, 2)
    ba, pa = b - a, GRID - a
    h = ((pa * ba).sum(-1) / (ba * ba).sum(-1).clamp_min(1e-6)).clamp(0, 1)
    return (pa - h.unsqueeze(-1) * ba).norm(dim=-1) - w.view(*w.shape, 1, 1)


def paint(d, gray):
    """Union of shapes (min SDF) painted in one gray level on mid-gray."""
    mask = torch.sigmoid(-d.min(1).values / SIG).unsqueeze(1)
    g = torch.sigmoid(gray).view(-1, 1, 1, 1)
    return (0.5 * (1 - mask) + g * mask).expand(-1, 3, -1, -1)


def width(p):
    return 0.02 + 0.25 * torch.sigmoid(p)


def fam_V():
    g = lambda *s, sd: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
    P = {"v": g(N, 2, sd=0.3), "e": g(N, 2, 2, sd=0.8), "w": g(N, 2, sd=1), "gray": g(N, sd=2)}

    def r(P):
        v = torch.tanh(P["v"]).unsqueeze(1).expand(-1, 2, -1)
        return paint(seg_sdf(v, torch.tanh(P["e"]), width(P["w"])), P["gray"])
    return P, r


def fam_bars2():
    g = lambda *s, sd: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
    P = {"a": g(N, 2, 2, sd=0.6), "b": g(N, 2, 2, sd=0.6), "w": g(N, 2, sd=1), "gray": g(N, sd=2)}

    def r(P):
        return paint(seg_sdf(torch.tanh(P["a"]), torch.tanh(P["b"]), width(P["w"])), P["gray"])
    return P, r


def fam_O():
    g = lambda *s, sd: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
    P = {"c": g(N, 2, sd=0.3), "r": g(N, 2, sd=1), "t": g(N, sd=1), "w": g(N, sd=1),
         "f": g(N, sd=1), "gray": g(N, sd=2)}

    def r(P):
        c = (torch.tanh(P["c"]) * 0.8).view(N, 1, 1, 1, 2)
        rad = (0.05 + 0.75 * torch.sigmoid(P["r"])).view(N, 1, 1, 1, 2)
        th = P["t"].view(N, 1, 1, 1)
        q = GRID - c
        qx = q[..., 0] * torch.cos(th) + q[..., 1] * torch.sin(th)
        qy = -q[..., 0] * torch.sin(th) + q[..., 1] * torch.cos(th)
        d_disc = (torch.sqrt((qx / rad[..., 0]) ** 2 + (qy / rad[..., 1]) ** 2 + 1e-8) - 1) * rad.min(-1).values
        d_ring = d_disc.abs() - (0.02 + 0.2 * torch.sigmoid(P["w"])).view(N, 1, 1, 1)
        f = torch.sigmoid(P["f"]).view(N, 1, 1, 1)
        return paint((1 - f) * d_ring + f * d_disc, P["gray"])
    return P, r


def fam_free5():
    return init_prims(N), render


FAMILIES = {"V": fam_V, "O": fam_O, "bars2": fam_bars2, "free5": fam_free5}


@torch.no_grad()
def describe_V(P, k):
    v = torch.tanh(P["v"][k]).cpu().numpy()
    e = torch.tanh(P["e"][k]).cpu().numpy()
    arms = e - v
    ang = np.degrees(np.arccos(np.clip((arms[0] @ arms[1]) / (np.linalg.norm(arms[0]) * np.linalg.norm(arms[1]) + 1e-9), -1, 1)))
    bis = arms[0] / (np.linalg.norm(arms[0]) + 1e-9) + arms[1] / (np.linalg.norm(arms[1]) + 1e-9)
    # direction the vertex points (opposite the bisector), image coords: +y is down
    point_dir = np.degrees(np.arctan2(-bis[1], -bis[0]))
    return {"opening_deg": round(float(ang), 1), "vertex_points_deg": round(float(point_dir), 1),
            "gray": round(float(torch.sigmoid(P["gray"][k])), 2)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, score = Center(), Neuron()
    summary = {}
    for fam, make in FAMILIES.items():
        for sign, tag in ((1, "max"), (-1, "min")):
            torch.manual_seed(7)
            P, rend = make()
            opt = torch.optim.Adam(P.values(), lr=0.03)
            for _ in range(STEPS):
                loss = -sign * f(jitter(rend(P))).mean() / 100
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                x = rend(P)
                torch.manual_seed(999)  # same 16 jitters for every family
                centre = torch.stack([f(jitter(x)) for _ in range(16)]).mean(0).cpu().numpy()
                smax = np.array([score.measure([to_pil(xi)])[0][0] for xi in x])
            order = np.argsort(-sign * centre)
            best = order[:12]
            save_sheet([to_pil(x[k]) for k in best],
                       [f"{fam} {tag} #{i + 1}: ctr {centre[k]:.0f} max {smax[k]:.0f}" for i, k in enumerate(best)],
                       OUT / f"{fam}_{tag}_best12.png", cols=6)
            to_pil(x[order[0]]).save(OUT / f"{fam}_{tag}_best.png")
            top5 = centre[order[:5]]
            summary[f"{fam}_{tag}"] = {
                "best_centre": float(centre[order[0]]), "top5_mean_centre": float(top5.mean()),
                "median_centre": float(np.median(centre)), "best_spatial_max": float(smax[order[0]]),
                **({"best_V_geometry": [describe_V(P, int(k)) for k in order[:5]]} if fam == "V" else {})}
            print(f"{fam:>6} {tag}: best ctr {centre[order[0]]:7.0f} | top-5 mean {top5.mean():7.0f} | "
                  f"median {np.median(centre):7.0f} | best img spatial max {smax[order[0]]:6.0f}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
