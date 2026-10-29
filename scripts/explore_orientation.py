"""Phase A orientation / shape sweep on a SEPARATE synthetic set (sizes 0.4 and 0.6),
so the pre-registered set in stimuli/synthetic stays unused until Phase B.

Reports 4e:55 max (minus gray baseline) per shape x orientation, filled + textured only.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import make_synthetic as ms  # noqa: E402
from neuron import Neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "explore"


def main():
    n = Neuron()
    rng = np.random.default_rng(1)
    base = n.measure([ms.Image.new("RGB", (224, 224), (ms.GRAY,) * 3)])[0][0]
    shapes = ["V", "roundedV", "U", "O", "disc"]
    res = {}
    for shape in shapes:
        angles = [0] if shape in ("O", "disc") else ms.ORIENTATIONS
        for angle in angles:
            imgs = [ms.render(shape, scale, angle, contrast, version, rng)
                    for scale in (0.4, 0.6) for contrast in ms.CONTRASTS
                    for version in ("filled", "textured")]
            res[(shape, angle)] = np.array([m[0] for m in n.measure(imgs)]) - base
    lines = [f"gray baseline (max) = {base:.1f}; values are max - baseline, mean over 8 renders (sd)",
             "angle: 0 = point down (chin/V), 180 = point up (ear/spire)"]
    for shape in shapes:
        cells = [f"{a:>3}: {v.mean():7.1f} ({v.std():5.1f})" for (s, a), v in res.items() if s == shape]
        lines.append(f"{shape:>9} | " + " | ".join(cells))
    print("\n".join(lines))
    (OUT / "orientation.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
