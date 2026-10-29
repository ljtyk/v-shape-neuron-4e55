"""Phase B, round 1: draw candidate images and set up the labeling page. NO activations.

Sources: the held-out pool (ImageNet val, never run through the model) and ImageNet-V2
matched-frequency. Images are saved as the exact square the model sees (short side -> 224,
centre crop), upscaled to 448 for raters, under anonymous IDs.

Writes:
  labeling_b/images/<id>.jpg, labeling_b/images.js     the rater set (shuffled)
  data/phase_b/candidates.csv                          id -> group, class, source, file (private)
"""
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import class_index, load, stimulus_pool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "data" / "imagenetv2" / "imagenetv2-matched-frequency-format-val"
LAB = ROOT / "labeling_b"
SEED = 20261027

GROUPS = {  # group: (classes, tasks)
    "T1_pointed": (["obelisk", "bell cote", "church", "candle", "volcano", "mountain tent"], ["rate"]),
    "T1_roundface": (["Persian cat", "Pekinese", "Japanese spaniel", "Shih-Tzu"], ["rate"]),
    "T3_round": (["golf ball", "tennis ball", "ping-pong ball", "wall clock", "analog clock",
                  "plate", "orange"], ["rate"]),
    "T3_boxy": (["carton", "crate", "chest", "safe", "washer", "monitor", "refrigerator",
                 "microwave", "dishwasher", "television", "wardrobe", "file"], ["rate"]),
    "T4_upright": (["red fox", "Arctic fox", "Pomeranian"], ["rate", "ears"]),
    "T4_floppy": (["basset", "clumber", "Saint Bernard"], ["rate", "ears"]),
    "T5_car": (["sports car", "beach wagon", "convertible", "cab", "racer", "minivan", "limousine",
                "jeep", "Model T", "pickup", "police van", "ambulance", "minibus"], ["rate"]),
}
TEXTURES = {"fox_fur": "red fox", "cat_fur": "tabby", "car_paint": "sports car", "stone": "stone wall"}
TEX_PER = 4


def model_square(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return img.crop((l, t, l + 224, t + 224))


def main():
    ds = load()
    idx = class_index(ds)
    rng = random.Random(SEED)
    v2 = {c: sorted(p for p in (V2 / str(c)).iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
          for c in range(1000)}
    pool_by_cls = {}
    for _, cname, i in stimulus_pool(ds):
        pool_by_cls.setdefault(idx[cname], []).append(i)

    items = []  # (group, class, source, path, tasks)
    used = set()
    # texture sources first (ImageNet-V2 only), excluded from every other group
    for tex, cname in TEXTURES.items():
        for p in rng.sample(v2[idx[cname]], TEX_PER):
            items.append((f"T2_tex_{tex}", cname, "v2", str(p), ["object"]))
            used.add(str(p))
    for group, (classes, tasks) in GROUPS.items():
        for cname in classes:
            if cname not in idx:
                raise KeyError(cname)
            c = idx[cname]
            for i in pool_by_cls.get(c, []):
                items.append((group, cname, "pool", ds.samples[i][0], tasks))
            for p in v2[c]:
                if str(p) not in used:
                    items.append((group, cname, "v2", str(p), tasks))

    (LAB / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "phase_b").mkdir(parents=True, exist_ok=True)
    ids = rng.sample(range(100000, 999999), len(items))
    key = []
    for (group, cname, src, path, tasks), anon in zip(items, ids):
        model_square(path).resize((448, 448), Image.BICUBIC).save(LAB / "images" / f"{anon}.jpg", quality=92)
        key.append(dict(image_id=anon, group=group, imagenet_class=cname, source=src,
                        file=path, tasks="+".join(tasks)))
    order = key[:]
    rng.shuffle(order)
    js = [{"id": str(r["image_id"]), "path": f"images/{r['image_id']}.jpg", "tasks": r["tasks"].split("+")}
          for r in order]
    (LAB / "images.js").write_text("const IMAGES = " + json.dumps(js) + ";\n")
    with open(ROOT / "data" / "phase_b" / "candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key[0].keys())
        w.writeheader()
        w.writerows(key)
    counts = {}
    for r in key:
        counts[r["group"]] = counts.get(r["group"], 0) + 1
    print(f"{len(key)} candidates:", counts)


if __name__ == "__main__":
    main()
