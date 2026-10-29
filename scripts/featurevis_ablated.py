"""Feature visualization of 4e:55 with cat/fox input routes ablated (see ablation.py).

Same recipe and seeds as featurevis.py. For each version (intact, A1, A2): 2 maximizing seeds
and 4 diverse renders. Each render is scored on the intact neuron and on its own ablated version.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ablation import ablation_sets  # noqa: E402
from featurevis import LAYER, run  # noqa: E402
from imagenet_val import load  # noqa: E402
from neuron import CHANNEL, DEVICE, Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "ablation"


def ablate(model, chans):
    with torch.no_grad():
        model.mixed4e_1x1_pre_relu_conv.weight[CHANNEL, chans] = 0
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"intact": [], **ablation_sets(load())}
    print({k: v for k, v in sets.items()})
    ch = objectives.channel(LAYER, CHANNEL)
    div = ch - 1e2 * objectives.diversity("mixed4e")
    renders = {}
    for version, chans in sets.items():
        model = ablate(inceptionv1(pretrained=True).to(DEVICE).eval(), chans)
        for seed in range(2):
            renders[(version, f"max{seed}")] = run(model, ch, seed=seed)[0]
        for k, im in enumerate(run(model, div, batch=4)):
            renders[(version, f"div{k}")] = im
        for (v, tag), im in renders.items():
            if v == version:
                im.save(OUT / f"{v}_{tag}.png")

    scorers = {v: Neuron() for v in sets}
    for v, chans in sets.items():
        ablate(scorers[v].model, chans)
    lines = ["render            intact-neuron max   own-version max"]
    for (v, tag), im in renders.items():
        lines.append(f"{v + '_' + tag:<18}{scorers['intact'].measure([im])[0][0]:>12.0f}"
                     f"{scorers[v].measure([im])[0][0]:>17.0f}")
    print("\n".join(lines))
    (OUT / "render_scores.txt").write_text("\n".join(lines), encoding="utf-8")

    versions, tags = list(sets), ["max0", "max1", "div0", "div1", "div2", "div3"]
    sheet = Image.new("RGB", (len(tags) * 224, len(versions) * 224), "white")
    for r, v in enumerate(versions):
        for c, t in enumerate(tags):
            sheet.paste(renders[(v, t)], (c * 224, r * 224))
    sheet.save(OUT / "sheet_rows_intact_A1_A2.png")


if __name__ == "__main__":
    main()
