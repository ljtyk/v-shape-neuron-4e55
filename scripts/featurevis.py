"""Feature visualization for 4e:55 (pre-ReLU), Lucid-style: FFT + decorrelated colour
parameterization, standard jitter/scale/rotate transforms, 512 steps.

Renders, at 224px:
  max      maximize the channel
  min      minimize the channel (the negative pole)
  diverse  4 maximizing images with a diversity penalty on mixed4e
Then scores every render with scripts/neuron.py (same hook as the rest of the study).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, param, render
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from neuron import CHANNEL, DEVICE, Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "featurevis"
LAYER = "mixed4e_1x1_pre_relu_conv"
STEPS = 512


def run(model, obj, batch=1, seed=0):
    torch.manual_seed(seed)
    imgs = render.render_vis(
        model, obj, lambda: param.image(224, batch=batch),
        thresholds=(STEPS,), show_image=False, progress=False)[-1]
    return [Image.fromarray((im * 255).clip(0, 255).astype(np.uint8)) for im in imgs]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = inceptionv1(pretrained=True).to(DEVICE).eval()  # redirected ReLU on, as in Lucid
    ch = objectives.channel(LAYER, CHANNEL)
    renders = {}
    for seed in range(2):
        renders[f"max_seed{seed}"] = run(model, ch, seed=seed)[0]
        renders[f"min_seed{seed}"] = run(model, -ch, seed=seed)[0]
    div = ch - 1e2 * objectives.diversity("mixed4e")
    for k, im in enumerate(run(model, div, batch=4)):
        renders[f"diverse_{k}"] = im
    for name, im in renders.items():
        im.save(OUT / f"{name}.png")

    n = Neuron()
    scores = {name: n.measure([im])[0] for name, im in renders.items()}
    lines = ["render          max     mean   (natural images: top class mean ~976, top image ~1590, median class ~193)"]
    for name, (mx, mn, _) in scores.items():
        lines.append(f"{name:<14}{mx:8.0f}{mn:9.0f}")
    print("\n".join(lines))
    (OUT / "scores.txt").write_text("\n".join(lines), encoding="utf-8")

    names = list(renders)
    sheet = Image.new("RGB", (4 * 224, ((len(names) + 3) // 4) * 224), "white")
    for k, name in enumerate(names):
        sheet.paste(renders[name], ((k % 4) * 224, (k // 4) * 224))
    sheet.save(OUT / "sheet.png")


if __name__ == "__main__":
    main()
