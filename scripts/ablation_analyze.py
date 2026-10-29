"""What does 4e:55 respond to once cat/fox routes are removed? (results/ablation/rerank.npz)"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from explore_analyze import top_classes  # noqa: E402
from imagenet_val import load  # noqa: E402
from neuron import Neuron  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "ablation"
PROBES = ["Persian cat", "tabby", "red fox", "lesser panda", "papillon", "Pomeranian",
          "beach wagon", "sports car", "minivan", "cab", "grille", "convertible",
          "bullet train", "airliner", "warplane", "obelisk", "church", "spindle", "candle",
          "volcano", "hare", "gazelle", "brown bear", "pug", "golf ball", "wall clock", "barometer"]


def main():
    ds = load()
    r = dict(np.load(OUT / "rerank.npz"))
    keep = r["sample_idx"]
    cls = np.array([ds.samples[i][1] for i in keep])
    versions = [k for k in ("intact", "A1", "A2", "A3") if k in r]
    lines = []
    say = lambda s="": (print(s), lines.append(s))  # noqa: E731
    for v in versions:
        s = r[v]
        say(f"== {v}" + (f" (ablated {len(r['chans_' + v])} channels)" if v != "intact" else "") +
            f": top image {s.max():.0f}, median image {np.median(s):.0f}")
        say(f"  top 50 images: {top_classes(ds, cls, s, k=50, n=10)}")
        prof = collections.defaultdict(list)
        for j, c in enumerate(cls):
            prof[c].append(s[j])
        means = sorted(((np.mean(x), ds.classes[c][0]) for c, x in prof.items()), reverse=True)
        say("  top 20 classes: " + ", ".join(f"{n} {m:.0f}" for m, n in means[:20]))
        rank = {n: k for k, (_, n) in enumerate(means, 1)}
        say("  probe ranks: " + ", ".join(f"{p} #{rank[p]}" for p in PROBES if p in rank))
        say()
        top = np.argsort(-s)[:24]
        sheet = Image.new("RGB", (6 * 150, 4 * 150), "white")
        for k, j in enumerate(top):
            x = Neuron.preprocess(Image.open(ds.samples[keep[j]][0])).permute(1, 2, 0).numpy()
            sheet.paste(Image.fromarray((x * 255).astype("uint8")).resize((150, 150)),
                        ((k % 6) * 150, (k // 6) * 150))
        sheet.save(OUT / f"top24_{v}.jpg", quality=90)
    (OUT / "rerank_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
