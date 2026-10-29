"""Simplest image that reaches a fixed 4e:55 activation (no shape vocabulary).

Grayscale pixels from gray + tiny noise. Minimize complexity subject to
centre activation (pre-ReLU, under jitter) >= target, with a hinge penalty whose weight
grows while the constraint is violated (simple augmented-penalty schedule).

Complexity variants:
  tv     total variation (edges)
  tv_l1  total variation + L1 deviation from gray ("ink"), equally weighted

Targets: 500, 1000 (about an average Persian photo), 2000 (above any real photo).
3 seeds each. Final images are scored on 16 fixed jitters (centre) and plain spatial max.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from neuron import DEVICE, Neuron  # noqa: E402
from simple_shapes import S, Center, jitter, save_sheet, to_pil, tv  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "minimal_image"
TARGETS, SEEDS, STEPS = [500, 1000, 2000], 3, 1500


def complexity(x, kind):
    c = tv(x)
    if kind == "tv_l1":
        c = c + (x - 0.5).abs().mean()
    return c


def optimize(f, target, kind, seed):
    torch.manual_seed(seed)
    p = (torch.randn(1, 1, S, S, device=DEVICE) * 0.01).requires_grad_()
    opt = torch.optim.Adam([p], lr=0.02)
    mu = 1.0
    for step in range(STEPS):
        x = torch.sigmoid(p).expand(-1, 3, -1, -1)
        act = f(torch.cat([jitter(x) for _ in range(4)])).mean()
        short = torch.relu(target - act) / target
        loss = complexity(x, kind) + mu * short
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 49:
            mu = mu * 2 if short.item() > 0.02 else max(mu * 0.9, 1.0)
    return torch.sigmoid(p).detach().expand(-1, 3, -1, -1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, score = Center(), Neuron()
    imgs, labels, rows = [], [], []
    for kind in ("tv", "tv_l1"):
        for target in TARGETS:
            for seed in range(SEEDS):
                x = optimize(f, target, kind, seed)
                with torch.no_grad():
                    torch.manual_seed(999)
                    ctr = torch.stack([f(jitter(x)) for _ in range(16)]).mean().item()
                    tvv, ink = tv(x).item(), (x - 0.5).abs().mean().item()
                im = to_pil(x[0])
                smax = score.measure([im])[0][0]
                im.save(OUT / f"{kind}_t{target}_s{seed}.png")
                # contrast-stretched copy for viewing
                a = np.asarray(im.convert("L")).astype(float) - 128
                a = a / (np.abs(a).max() + 1e-6) * 127 + 128
                from PIL import Image
                st = Image.fromarray(a.clip(0, 255).astype("uint8")).convert("RGB")
                st.save(OUT / f"{kind}_t{target}_s{seed}_stretched.png")
                imgs += [st]
                labels += [f"{kind} t={target} s{seed}: ctr {ctr:.0f}"]
                rows.append((kind, target, seed, ctr, smax, tvv, ink))
                print(f"{kind:>5} target {target:>4} seed {seed}: centre {ctr:6.0f} | spatial max {smax:6.0f} | "
                      f"TV {tvv:.4f} | ink {ink:.4f}", flush=True)
    save_sheet(imgs, labels, OUT / "sheet_stretched.png", cols=SEEDS)
    with open(OUT / "scores.csv", "w") as fh:
        fh.write("kind,target,seed,centre,spatial_max,tv,ink\n")
        for r in rows:
            fh.write(",".join(str(round(v, 5)) if isinstance(v, float) else str(v) for v in r) + "\n")


if __name__ == "__main__":
    main()
