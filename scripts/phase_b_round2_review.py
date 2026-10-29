"""Build the round-2 rating items (no activations).

  earzoom  for each Test 4 source: a zoomed square crop around the head (the region spanned by both
           raters' round-1 ear boxes, doubled in size, at least 40% of the frame), gridded.
           Raters draw a tight box per ear. Finer than round 1 because the grid covers only the crop.
  body     for each Test 4 source: the full gridded image; raters box the animal's torso.
  texbox   for each texture source: the full gridded image; raters box a texture-only area.

Every item gets a new anonymous 6-digit ID. Writes <raters>/images_r2/, <raters>/batches_r2/ and
data/phase_b/round2_review_key.csv (review_id, task, source_id, imagenet_class, crop box for earzoom).
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import phase_b_round2 as r2  # noqa: E402
from phase_b_build import model_square  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SURFACE = {"T2_tex_fox_fur": "fur", "T2_tex_cat_fur": "fur", "T2_tex_car_paint": "car_paint", "T2_tex_stone": "stone"}
COLS = "ABCDEFGHIJ"


def gridded(img):
    canvas = Image.new("RGB", (600, 600), "white")
    canvas.paste(img.resize((560, 560), Image.BICUBIC), (40, 40))
    d = ImageDraw.Draw(canvas, "RGBA")
    for k in range(11):
        x = 40 + k * 56
        d.line([(x, 40), (x, 600)], fill=(255, 255, 0, 110))
        d.line([(40, x), (600, x)], fill=(255, 255, 0, 110))
    for k in range(10):
        d.text((40 + k * 56 + 24, 14), COLS[k], fill="black")
        d.text((14, 40 + k * 56 + 22), str(k), fill="black")
    return canvas


def head_crop(ears):
    x0 = min(e[0] for e in ears); y0 = min(e[1] for e in ears)
    x1 = max(e[2] for e in ears); y1 = max(e[3] for e in ears)
    side = max(x1 - x0, y1 - y0) * 2.0
    side = min(1.0, max(side, 0.40))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + side * 0.15  # shift down a little to include the face below the ears
    l = min(max(0, cx - side / 2), 1 - side)
    t = min(max(0, cy - side / 2), 1 - side)
    return l, t, l + side, t + side


def main():
    R = Path(sys.argv[1])
    (R / "images_r2").mkdir(exist_ok=True)
    (R / "batches_r2").mkdir(exist_ok=True)
    key, A, B = r2.labels()
    items = []
    for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "round2_manifest_edits.csv")):  # included Test 4 sources
        iid = r["image_id"]
        sq = model_square(key[iid]["file"])
        ears = r2.boxes(A[iid], ["ear1", "ear2"]) + r2.boxes(B[iid], ["ear1", "ear2"])
        c = head_crop(ears)
        crop = sq.crop(tuple(round(v * 224) for v in c))
        items.append((dict(task="earzoom", source_id=iid, imagenet_class=r["animal"],
                           crop=" ".join(f"{v:.4f}" for v in c)), gridded(crop)))
        items.append((dict(task="body", source_id=iid, imagenet_class=r["animal"], crop=""), gridded(sq)))
    for iid, k in key.items():
        if k["group"] in SURFACE:
            items.append((dict(task=f"texbox {SURFACE[k['group']]}", source_id=iid,
                               imagenet_class=k["imagenet_class"], crop=""), gridded(model_square(k["file"]))))
    rng = random.Random(20261028)
    rng.shuffle(items)
    ids = rng.sample(range(500000, 599999), len(items))
    rows = []
    for (row, img), rid in zip(items, ids):
        img.save(R / "images_r2" / f"{rid}.png")
        rows.append(dict(review_id=rid, **row))
    with open(ROOT / "data" / "phase_b" / "round2_review_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for old in (R / "batches_r2").glob("*.txt"):
        old.unlink()
    half = (len(rows) + 1) // 2
    for b, chunk in enumerate((rows[:half], rows[half:])):
        (R / "batches_r2" / f"batch_{b:02d}.txt").write_text(
            "\n".join(f"{x['review_id']}\t{x['task']}" for x in chunk) + "\n")
    print(f"{len(rows)} items: {sum(x['task'] == 'earzoom' for x in rows)} earzoom, "
          f"{sum(x['task'] == 'body' for x in rows)} body, {sum(x['task'].startswith('texbox') for x in rows)} texbox; 2 batches")


if __name__ == "__main__":
    main()
