"""Crop the held-out stimulus pool to its main object and set up the rater page.

Writes:
  labeling/images/<anon_id>.jpg   object crops, anonymous names (no class in filename)
  labeling/images.js              shuffled image list for label.html
  data/pool_key.csv               anon_id -> family, ImageNet class, source file, crop box
"""
import csv
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from imagenet_val import SEED, load, object_boxes, stimulus_pool, val_id  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT / "labeling" / "images"
PAD = 0.10


def square_crop(box, w, h):
    """Square crop around the box, padded by PAD, clipped to the image."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * PAD)
    side = min(side, w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    l = min(max(0, cx - side / 2), w - side)
    t = min(max(0, cy - side / 2), h - side)
    return round(l), round(t), round(l + side), round(t + side)


def main():
    ds = load()
    pool = stimulus_pool(ds)
    boxes = object_boxes()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 1)
    ids = rng.sample(range(10000, 99999), len(pool))
    key = []
    for (family, cls, i), anon in zip(pool, ids):
        path = ds.samples[i][0]
        stem = val_id(path)
        obj = max(boxes[stem], key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        img = Image.open(path).convert("RGB")
        crop = square_crop(obj[:4], *img.size)
        img.crop(crop).resize((448, 448), Image.BICUBIC).save(IMG_DIR / f"{anon}.jpg", quality=92)
        key.append(dict(image_id=anon, family=family, imagenet_class=cls,
                        source=Path(path).name, crop=" ".join(map(str, crop))))

    order = key[:]
    rng.shuffle(order)
    items = [{"id": str(r["image_id"]), "path": f"images/{r['image_id']}.jpg"} for r in order]
    (PROJECT / "labeling" / "images.js").write_text("const IMAGES = " + json.dumps(items) + ";\n")
    with open(PROJECT / "data" / "pool_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=key[0].keys())
        w.writeheader()
        w.writerows(key)
    print(f"{len(key)} images ready for labeling in {IMG_DIR}")


if __name__ == "__main__":
    main()
