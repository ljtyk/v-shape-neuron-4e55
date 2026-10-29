"""Simplest images that drive 4e:55 (pre-ReLU) up or down.

Objective: the centre position (7, 7) of the 14x14 map, so the result is one figure, not a
texture. Every step applies random jitter (shift +-12 px, scale 0.9-1.1) so results can't
rely on pixel-exact tricks.

Part 1, "tv":     grayscale pixels, loss = -act/100 + lambda * TV, swept over lambda.
Part 2, "prims":  no pixels. The image is K soft primitives on mid-gray, each with one gray
                  level and an opacity. Capsules (thick line segments: V, Lambda, triangles)
                  and ellipses (a fill parameter blends ring <-> disc). 48 random restarts.

Usage: python simple_shapes.py tv|prims
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lucent.modelzoo import inceptionv1
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from neuron import CHANNEL, DEVICE, Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "simple_shapes"
S = 224
torch.backends.cudnn.benchmark = True


class Center:
    """Pre-ReLU 4e:55 at the map centre, from a model with redirected ReLU (better gradients)."""

    def __init__(self):
        self.m = inceptionv1(pretrained=True).to(DEVICE).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.a = None
        self.m.mixed4e_1x1_pre_relu_conv.register_forward_hook(lambda m, i, o: setattr(self, "a", o))

    def __call__(self, x):  # x in [0,1], (N,3,S,S)
        self.m(x * 255 - 117)
        h, w = self.a.shape[2:]
        return self.a[:, CHANNEL, h // 2, w // 2]


def jitter(x):
    n = x.shape[0]
    s = torch.empty(n, device=x.device).uniform_(0.9, 1.1)
    t = torch.empty(n, 2, device=x.device).uniform_(-12 / S * 2, 12 / S * 2)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / s
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


def tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def save_sheet(imgs, labels, path, cols):
    from PIL import ImageDraw
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * S, rows * (S + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (im, lab) in enumerate(zip(imgs, labels)):
        x, y = (k % cols) * S, (k // cols) * (S + 18)
        sheet.paste(im, (x, y))
        d.text((x + 4, y + S + 3), lab, fill=(0, 0, 0))
    sheet.save(path)


def to_pil(x):
    return Image.fromarray((x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


# ---------------------------------------------------------------- Part 1: TV sweep
def run_tv(f, score, lams=(0, 30, 100, 300, 1000), name="tv_sweep"):
    imgs, labels = [], []
    for sign, tag in ((1, "max"), (-1, "min")):
        for lam in lams:
            torch.manual_seed(0)
            p = torch.zeros(2, 1, S, S, device=DEVICE).normal_(0, 0.1).requires_grad_()  # 2 seeds
            opt = torch.optim.Adam([p], lr=0.05)
            for _ in range(512):
                x = torch.sigmoid(p).expand(-1, 3, -1, -1)
                loss = -sign * f(jitter(x)).mean() / 100 + lam * tv(x)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                for k, xi in enumerate(torch.sigmoid(p).expand(-1, 3, -1, -1)):
                    im = to_pil(xi)
                    mx, mn, _ = score.measure([im])[0]
                    imgs.append(im)
                    labels.append(f"{tag} TV={lam} s{k}: max {mx:.0f}")
                    im.save(OUT / f"tv_{tag}_lam{lam}_s{k}.png")
    save_sheet(imgs, labels, OUT / f"{name}.png", cols=4)
    print("\n".join(labels))


# ---------------------------------------------------------------- Part 2: primitives
K_CAP, K_ELL, N, SIG = 3, 2, 48, 1.5 / S * 2


def render(P):
    """P: dict of (N, K, ...) params -> (N,3,S,S) image."""
    n = P["cap_a"].shape[0]
    lin = torch.linspace(-1 + 1 / S, 1 - 1 / S, S, device=DEVICE)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    p = torch.stack([xx, yy], -1).view(1, 1, S, S, 2)
    img = torch.full((n, 1, S, S), 0.5, device=DEVICE)
    # capsules
    a, b = torch.tanh(P["cap_a"]).view(n, K_CAP, 1, 1, 2), torch.tanh(P["cap_b"]).view(n, K_CAP, 1, 1, 2)
    w = (0.02 + 0.25 * torch.sigmoid(P["cap_w"])).view(n, K_CAP, 1, 1)
    ba, pa = b - a, p - a
    h = ((pa * ba).sum(-1) / (ba * ba).sum(-1).clamp_min(1e-6)).clamp(0, 1)
    d_cap = (pa - h.unsqueeze(-1) * ba).norm(dim=-1) - w
    # ellipses: centre, radii, rotation, ring thickness, fill (0 ring .. 1 disc)
    c = torch.tanh(P["ell_c"]).view(n, K_ELL, 1, 1, 2) * 0.8
    r = (0.05 + 0.75 * torch.sigmoid(P["ell_r"])).view(n, K_ELL, 1, 1, 2)
    th = P["ell_t"].view(n, K_ELL, 1, 1)
    q = p - c
    qx = q[..., 0] * torch.cos(th) + q[..., 1] * torch.sin(th)
    qy = -q[..., 0] * torch.sin(th) + q[..., 1] * torch.cos(th)
    rmin = r.min(-1).values
    d_disc = (torch.sqrt((qx / r[..., 0]) ** 2 + (qy / r[..., 1]) ** 2 + 1e-8) - 1) * rmin
    thick = (0.02 + 0.2 * torch.sigmoid(P["ell_w"])).view(n, K_ELL, 1, 1)
    d_ring = d_disc.abs() - thick
    fill = torch.sigmoid(P["ell_f"]).view(n, K_ELL, 1, 1)
    d_ell = (1 - fill) * d_ring + fill * d_disc
    d = torch.cat([d_cap, d_ell], 1)                                  # (n, K, S, S)
    mask = torch.sigmoid(-d / SIG)
    alpha = torch.sigmoid(P["alpha"]).view(n, -1, 1, 1)
    col = torch.sigmoid(P["col"]).view(n, -1, 1, 1)
    for k in range(d.shape[1]):
        m = (mask[:, k] * alpha[:, k]).unsqueeze(1)
        img = img * (1 - m) + col[:, k].view(n, 1, 1, 1) * m
    return img.expand(-1, 3, -1, -1)


def init_prims(n):
    g = lambda *s, sd=1.0: (torch.randn(*s, device=DEVICE) * sd).requires_grad_()  # noqa: E731
    K = K_CAP + K_ELL
    return {"cap_a": g(n, K_CAP, 2, sd=0.6), "cap_b": g(n, K_CAP, 2, sd=0.6), "cap_w": g(n, K_CAP),
            "ell_c": g(n, K_ELL, 2, sd=0.5), "ell_r": g(n, K_ELL, 2), "ell_t": g(n, K_ELL),
            "ell_w": g(n, K_ELL), "ell_f": g(n, K_ELL), "alpha": g(n, K, sd=0.5), "col": g(n, K, sd=2.0)}


def run_prims(f, score):
    results = []
    for sign, tag in ((1, "max"), (-1, "min")):
        torch.manual_seed(1)
        P = init_prims(N)
        opt = torch.optim.Adam(P.values(), lr=0.03)
        for step in range(600):
            x = render(P)
            loss = -sign * torch.stack([f(jitter(x)) for _ in range(2)]).mean() / 100
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = render(P)
            for k in range(N):
                im = to_pil(x[k])
                mx = score.measure([im])[0][0]
                al = torch.sigmoid(P["alpha"][k]).cpu().numpy()
                results.append((tag, k, mx, im, int((al > 0.3).sum())))
        ranked = sorted([r for r in results if r[0] == tag], key=lambda r: -sign * r[2])
        for r in ranked:
            r[3].save(OUT / f"prims_{tag}_rank{ranked.index(r) + 1:02d}_restart{r[1]:02d}.png")
        save_sheet([r[3] for r in ranked[:12]],
                   [f"{tag} #{i + 1}: max {r[2]:.0f} ({r[4]} prims on)" for i, r in enumerate(ranked[:12])],
                   OUT / f"prims_{tag}_best12.png", cols=6)
        vals = np.array([r[2] for r in ranked])
        print(f"{tag}: best {vals[0]:.0f}, median {np.median(vals):.0f}, worst {vals[-1]:.0f} over {N} restarts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, score = Center(), Neuron()
    gray = score.measure([Image.new("RGB", (S, S), (128,) * 3)])[0][0]
    print(f"gray baseline max {gray:.1f}")
    if sys.argv[1] == "tv_strong":
        run_tv(f, score, lams=(3000, 10000, 30000, 100000), name="tv_sweep_strong")
    else:
        {"tv": run_tv, "prims": run_prims}[sys.argv[1]](f, score)


if __name__ == "__main__":
    main()
