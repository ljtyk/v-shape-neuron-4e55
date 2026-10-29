"""Phase B round 2: build the edited and synthetic stimuli from round-1 labels. NO activations.

  python phase_b_round2.py edits        ear edits for Test 4 (shape edit + control edit per source)
  python phase_b_round2.py composites   Test 2 shape x texture composites
Outputs go to stimuli/phase_b/ and data/phase_b/round2_manifest_*.csv.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
from phase_b_build import model_square  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data" / "phase_b" / "labels"
OUT = ROOT / "stimuli" / "phase_b"
GRAY = 128
PLAIN = 70  # untextured control fill: flat dark gray, visible against the mid-gray background


def labels():
    key = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "candidates.csv"))}
    A = {r["image_id"]: r for r in csv.DictReader(open(LAB / "phase_b_labels_raterA.csv"))}
    B = {r["image_id"]: r for r in csv.DictReader(open(LAB / "phase_b_labels_raterB.csv"))}
    return key, A, B


def boxes(r, prefix_list):
    out = []
    for p in prefix_list:
        v = [r[f"{p}_{k}"] for k in ("x0", "y0", "x1", "y1")]
        if all(v):
            out.append(tuple(float(x) for x in v))
    return out


def enlarge(b, f=1.10):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return max(0, cx - w / 2), max(0, cy - h / 2), min(1, cx + w / 2), min(1, cy + h / 2)


# --------------------------------------------------------------------- ear edits
PROMPTS = {  # source group -> (shape-edit prompt, control prompt, negative for shape edit)
    "T4_upright": ("a {a} with soft rounded floppy ears folded down", "a {a} with pointed upright ears",
                   "pointed ears, upright ears"),
    "T4_floppy": ("a {a} with tall pointed upright ears", "a {a} with long floppy ears",
                  "floppy ears, drooping ears"),
}


def edits():
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    key, A, B = labels()
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16", torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    (OUT / "edits").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for iid, k in key.items():
        if k["group"] not in PROMPTS:
            continue
        a, b = A[iid], B[iid]
        reasons = []
        if a["ears_not_visible"] == "1" or b["ears_not_visible"] == "1":
            reasons.append("ears not visible (a rater)")
        if a["view"] not in ("frontal", "three-quarter") or b["view"] not in ("frontal", "three-quarter"):
            reasons.append(f"view {a['view']}/{b['view']}")
        ears = boxes(a, ["ear1", "ear2"]) + boxes(b, ["ear1", "ear2"])
        if not ears:
            reasons.append("no ear boxes")
        if reasons:
            skipped.append((iid, k["group"], "; ".join(reasons)))
            continue
        src = model_square(k["file"]).resize((512, 512), Image.BICUBIC)
        mask = Image.new("L", (512, 512), 0)
        d = ImageDraw.Draw(mask)
        for e in ears:
            x0, y0, x1, y1 = enlarge(e)
            d.rectangle([x0 * 512, y0 * 512, x1 * 512, y1 * 512], fill=255)
        soft = mask.filter(ImageFilter.GaussianBlur(6))
        animal = k["imagenet_class"]
        shape_p, ctrl_p, neg = PROMPTS[k["group"]]
        seed = int(iid) % (2 ** 31)
        outs = {}
        for kind, prompt, negp in (("shape", shape_p, neg), ("control", ctrl_p, "")):
            g = torch.Generator("cuda").manual_seed(seed)
            res = pipe(prompt=prompt.format(a=animal), negative_prompt=negp or None, image=src, mask_image=mask,
                       num_inference_steps=30, guidance_scale=7.5, generator=g).images[0].resize((512, 512))
            final = Image.composite(res, src, soft).resize((224, 224), Image.BICUBIC)
            final.save(OUT / "edits" / f"{iid}_{kind}.png")
            outs[kind] = f"{iid}_{kind}.png"
        model_square(k["file"]).save(OUT / "edits" / f"{iid}_original.png")
        rows.append(dict(image_id=iid, group=k["group"], animal=animal, seed=seed, n_ear_boxes=len(ears),
                         shape_file=outs["shape"], control_file=outs["control"], original_file=f"{iid}_original.png"))
        print(f"edited {iid} ({animal})", flush=True)
    with open(ROOT / "data" / "phase_b" / "round2_manifest_edits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "data" / "phase_b" / "round2_skipped_ear_sources.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image_id", "group", "reason")] + skipped)
    print(f"{len(rows)} sources edited ({sum(r['group'] == 'T4_upright' for r in rows)} upright, "
          f"{sum(r['group'] == 'T4_floppy' for r in rows)} floppy); {len(skipped)} skipped")


# --------------------------------------------------------------------- composites
TEX_NAMES = {"T2_tex_fox_fur": "fox_fur", "T2_tex_cat_fur": "cat_fur", "T2_tex_car_paint": "car_paint",
             "T2_tex_stone": "stone"}
SIZES = {"s40": 0.40, "s60": 0.60, "s80": 0.80}


def shape_mask(shape, scale, ss=4):
    """Binary mask (224x224) of the shape at a linear scale factor, centred, point facing down."""
    big = 224 * ss
    m = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(m)
    c, r = big / 2, scale * big / 2
    if shape == "disc":
        d.ellipse([c - r, c - r, c + r, c + r], fill=255)
    elif shape == "square":
        d.rectangle([c - r, c - r, c + r, c + r], fill=255)
    elif shape == "triangle":
        d.polygon([(c - r, c - r * 0.8), (c + r, c - r * 0.8), (c, c + r)], fill=255)
    elif shape == "V":
        t = r * 0.45  # arm thickness
        d.polygon([(c - r, c - r * 0.8), (c - r + t, c - r * 0.8), (c, c + r - t * 1.6),
                   (c + r - t, c - r * 0.8), (c + r, c - r * 0.8), (c, c + r)], fill=255)
    return np.asarray(m.resize((224, 224), Image.LANCZOS), dtype=np.float32) / 255


def area_matched(shape, target):
    lo, hi = 0.05, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if shape_mask(shape, mid).sum() < target:
            lo = mid
        else:
            hi = mid
    return shape_mask(shape, (lo + hi) / 2)


def composites():
    key, A, B = labels()
    (OUT / "composites").mkdir(parents=True, exist_ok=True)
    (OUT / "patches").mkdir(parents=True, exist_ok=True)
    patches = []  # (texture, source_id, 224x224 array)
    for iid, k in key.items():
        if k["group"] not in TEX_NAMES:
            continue
        ba, bb = boxes(A[iid], ["obj"]), boxes(B[iid], ["obj"])
        bx = np.mean([ba[0], bb[0]], axis=0)
        cx, cy = (bx[0] + bx[2]) / 2 * 224, (bx[1] + bx[3]) / 2 * 224
        sq = model_square(k["file"])
        l, t = int(round(min(max(cx - 48, 0), 224 - 96))), int(round(min(max(cy - 48, 0), 224 - 96)))
        patch = sq.crop((l, t, l + 96, t + 96))
        patch.save(OUT / "patches" / f"{TEX_NAMES[k['group']]}_{iid}.png")
        patches.append((TEX_NAMES[k["group"]], iid, np.asarray(patch.resize((224, 224), Image.BICUBIC), dtype=np.float32)))
    patches.append(("plain_flat", "none", np.full((224, 224, 3), PLAIN, dtype=np.float32)))
    rows = []
    for size, frac in SIZES.items():
        target = shape_mask("square", frac).sum() * 0.6  # area every shape is matched to
        masks = {s: area_matched(s, target) for s in ("V", "triangle", "disc", "square")}
        for shape, m in masks.items():
            for tex, src, arr in patches:
                img = np.full((224, 224, 3), GRAY, dtype=np.float32) * (1 - m[..., None]) + arr * m[..., None]
                name = f"{shape}_{size}_{tex}_{src}.png"
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(OUT / "composites" / name)
                rows.append(dict(file=name, shape=shape, size=size, texture=tex, texture_source=src,
                                 mask_area_px=round(float(m.sum()), 1)))
    with open(ROOT / "data" / "phase_b" / "round2_manifest_composites.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} composites from {len(patches) - 1} texture patches + plain gray")


if __name__ == "__main__" and sys.argv[1] in ("edits", "composites"):
    {"edits": edits, "composites": composites}[sys.argv[1]]()


# --------------------------------------------------------------------- occlusions (Test 4)
def occlusions():
    key, A, B = labels()
    (OUT / "occlusions").mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "round2_manifest_edits.csv")):  # included sources
        iid = r["image_id"]
        ears = [enlarge(e) for e in boxes(A[iid], ["ear1", "ear2"]) + boxes(B[iid], ["ear1", "ear2"])]
        ctrl = []
        for x0, y0, x1, y1 in ears:
            dy = 1.5 * (y1 - y0)
            ctrl.append((x0, min(1, y0 + dy), x1, min(1, y1 + dy)))

        def cover(bxs):
            m = Image.new("L", (224, 224), 0)
            d = ImageDraw.Draw(m)
            for x0, y0, x1, y1 in bxs:
                d.rectangle([x0 * 224, y0 * 224, x1 * 224, y1 * 224], fill=255)
            return m
        me, mc = cover(ears), cover(ctrl)
        ae, ac = np.asarray(me).astype(bool).sum(), np.asarray(mc).astype(bool).sum()
        if ac < 0.8 * ae:
            excluded.append((iid, r["group"], f"control area {ac} < 80% of ear area {ae}"))
            continue
        src = model_square(key[iid]["file"])
        gray = Image.new("RGB", (224, 224), (GRAY,) * 3)
        src.save(OUT / "occlusions" / f"{iid}_original.png")
        Image.composite(gray, src, me).save(OUT / "occlusions" / f"{iid}_ear.png")
        Image.composite(gray, src, mc).save(OUT / "occlusions" / f"{iid}_control.png")
        rows.append(dict(image_id=iid, group=r["group"], animal=r["animal"], ear_area_px=int(ae), control_area_px=int(ac)))
    with open(ROOT / "data" / "phase_b" / "round2_manifest_occlusions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "data" / "phase_b" / "round2_excluded_occlusions.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image_id", "group", "reason")] + excluded)
    print(f"{len(rows)} occlusion sets ({sum(x['group'] == 'T4_upright' for x in rows)} upright, "
          f"{sum(x['group'] == 'T4_floppy' for x in rows)} floppy); {len(excluded)} excluded")


if __name__ == "__main__" and sys.argv[1] == "occlusions":
    occlusions()


# --------------------------------------------------------------------- final builds from round-2 labels
def r2_labels():
    """{rater: {review_id: record}} from verified round-2 batches, plus the review key."""
    import json
    R = Path(sys.argv[2])
    out = {}
    for rater in ("A", "B"):
        out[rater] = {}
        for f in sorted((R / "accepted_r2").glob(f"rater{rater}_batch*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[rater][str(r["id"])] = r
    rkey = list(csv.DictReader(open(ROOT / "data" / "phase_b" / "round2_review_key.csv")))
    return out, rkey


def cellbox(s):
    a, b = s.split(":")
    c0, r0, c1, r1 = ord(a[0].upper()) - 65, int(a[1:]), ord(b[0].upper()) - 65, int(b[1:])
    c0, c1 = sorted((c0, c1)); r0, r1 = sorted((r0, r1))
    return c0 / 10, r0 / 10, (c1 + 1) / 10, (r1 + 1) / 10


def occlusions_final():
    key = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "candidates.csv"))}
    lab, rkey = r2_labels()
    src_rows = {}
    for r in rkey:
        if r["task"] in ("earzoom", "body"):
            src_rows.setdefault(r["source_id"], {})[r["task"]] = r
    d = OUT / "occlusions"
    if d.exists():
        for f in d.glob("*.png"):
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)
    rows, excluded = [], []
    for sid, t in src_rows.items():
        ez, bd = t["earzoom"], t["body"]
        recA, recB = (lab[x].get(ez["review_id"], {}) for x in ("A", "B"))
        bA, bB = (lab[x].get(bd["review_id"], {}) for x in ("A", "B"))
        reasons = []
        if recA.get("ears_not_visible") or recB.get("ears_not_visible") or not recA.get("ear_boxes") or not recB.get("ear_boxes"):
            reasons.append("ears not visible (round 2)")
        if bA.get("no_body") or bB.get("no_body") or not bA.get("body_box") or not bB.get("body_box"):
            reasons.append("no torso")
        if reasons:
            excluded.append((sid, "; ".join(reasons)))
            continue
        cx0, cy0, cx1, cy1 = (float(v) for v in ez["crop"].split())
        cw, ch = cx1 - cx0, cy1 - cy0
        ears = []
        for rec in (recA, recB):
            for eb in rec["ear_boxes"]:
                x0, y0, x1, y1 = cellbox(eb)
                ears.append((cx0 + x0 * cw, cy0 + y0 * ch, cx0 + x1 * cw, cy0 + y1 * ch))
        body = np.mean([cellbox(bA["body_box"]), cellbox(bB["body_box"])], axis=0)
        bcx, bcy = (body[0] + body[2]) / 2, (body[1] + body[3]) / 2

        def mask(bxs):
            m = Image.new("L", (224, 224), 0)
            dr = ImageDraw.Draw(m)
            for x0, y0, x1, y1 in bxs:
                dr.rectangle([x0 * 224, y0 * 224, x1 * 224, y1 * 224], fill=255)
            return m
        me = mask(ears)
        ea = np.asarray(me) > 0
        ys, xs = np.nonzero(ea)
        ecx, ecy = xs.mean() / 224, ys.mean() / 224
        dx, dy = bcx - ecx, bcy - ecy
        mc = mask([(x0 + dx, y0 + dy, x1 + dx, y1 + dy) for x0, y0, x1, y1 in ears])
        ca = np.asarray(mc) > 0
        if ca.sum() < 0.8 * ea.sum():
            excluded.append((sid, f"control area {ca.sum()} < 80% of ear area {ea.sum()}"))
            continue
        if (ca & ea).any():
            excluded.append((sid, "control overlaps ears"))
            continue
        src = model_square(key[sid]["file"])
        gray = Image.new("RGB", (224, 224), (GRAY,) * 3)
        src.save(d / f"{sid}_original.png")
        Image.composite(gray, src, me).save(d / f"{sid}_ear.png")
        Image.composite(gray, src, mc).save(d / f"{sid}_control.png")
        rows.append(dict(image_id=sid, group=key[sid]["group"], animal=key[sid]["imagenet_class"],
                         ear_area_px=int(ea.sum()), control_area_px=int(ca.sum())))
    with open(ROOT / "data" / "phase_b" / "round2_manifest_occlusions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "data" / "phase_b" / "round2_excluded_occlusions.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image_id", "reason")] + excluded)
    print(f"{len(rows)} occlusion sets ({sum(x['group'] == 'T4_upright' for x in rows)} upright, "
          f"{sum(x['group'] == 'T4_floppy' for x in rows)} floppy); {len(excluded)} excluded")


def composites_final():
    key = {r["image_id"]: r for r in csv.DictReader(open(ROOT / "data" / "phase_b" / "candidates.csv"))}
    lab, rkey = r2_labels()
    for sub in ("composites", "patches"):
        if (OUT / sub).exists():
            for f in (OUT / sub).glob("*.png"):
                f.unlink()
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    patches, dropped = [], []
    for r in rkey:
        if not r["task"].startswith("texbox"):
            continue
        sid = r["source_id"]
        a, b = lab["A"].get(r["review_id"], {}), lab["B"].get(r["review_id"], {})
        if a.get("no_clean_area") or b.get("no_clean_area") or not a.get("texture_box") or not b.get("texture_box"):
            dropped.append((sid, "no clean area (a rater)"))
            continue
        ba, bb = cellbox(a["texture_box"]), cellbox(b["texture_box"])
        ix0, iy0, ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1]), min(ba[2], bb[2]), min(ba[3], bb[3])
        if ix1 - ix0 < 0.2 - 1e-9 or iy1 - iy0 < 0.2 - 1e-9:
            dropped.append((sid, "rater boxes overlap less than 2x2 cells"))
            continue
        side = min(ix1 - ix0, iy1 - iy0)
        cx, cy = (ix0 + ix1) / 2, (iy0 + iy1) / 2
        box = tuple(round(v * 224) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        tex = TEX_NAMES[key[sid]["group"]]
        patch = model_square(key[sid]["file"]).crop(box)
        patch.save(OUT / "patches" / f"{tex}_{sid}.png")
        patches.append((tex, sid, np.asarray(patch.resize((224, 224), Image.BICUBIC), dtype=np.float32)))
    patches.append(("plain_flat", "none", np.full((224, 224, 3), PLAIN, dtype=np.float32)))
    rows = []
    for size, frac in SIZES.items():
        target = shape_mask("square", frac).sum() * 0.6
        masks = {s: area_matched(s, target) for s in ("V", "triangle", "disc", "square")}
        for shape, m in masks.items():
            for tex, src, arr in patches:
                img = np.full((224, 224, 3), GRAY, dtype=np.float32) * (1 - m[..., None]) + arr * m[..., None]
                name = f"{shape}_{size}_{tex}_{src}.png"
                Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(OUT / "composites" / name)
                rows.append(dict(file=name, shape=shape, size=size, texture=tex, texture_source=src,
                                 mask_area_px=round(float(m.sum()), 1)))
    with open(ROOT / "data" / "phase_b" / "round2_manifest_composites.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(ROOT / "data" / "phase_b" / "round2_dropped_texture_sources.csv", "w", newline="") as f:
        csv.writer(f).writerows([("source_id", "reason")] + dropped)
    print(f"{len(rows)} composites from {len(patches) - 1} texture patches + flat fill; {len(dropped)} sources dropped")


if __name__ == "__main__" and sys.argv[1] == "final":
    occlusions_final()
    composites_final()
