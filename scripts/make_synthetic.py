"""Render the synthetic V / rounded-V / U / O set described in the design doc.

Shapes x 3 sizes x 8 orientations x 2 contrasts x 3 versions (outline, filled,
textured), on neutral gray at 224x224. O is rotation-invariant, so it is rendered
at a single orientation. Also writes the plain gray baseline image.
"""
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "stimuli" / "synthetic"
SIZE = 224
GRAY = 128
SS = 4  # supersampling factor for anti-aliasing

SHAPES = ["V", "roundedV", "U", "O", "disc"]  # O = ring with the same band width as the V arms
SCALES = {"small": 0.3, "medium": 0.5, "large": 0.75}
ORIENTATIONS = [i * 45 for i in range(8)]  # 0 = point facing down
CONTRASTS = {"dark_on_gray": 40, "light_on_gray": 215}  # shape intensity on gray bg
VERSIONS = ["outline", "filled", "textured"]


def circle(radius, n=200):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(t) * radius, np.sin(t) * radius], 1)


def shape_polygons(shape, scale, angle_deg, n=200):
    """[(polygon, is_hole)] in [-1,1] units, point facing down at 0 deg."""
    h = scale
    d = h * 0.22  # band width, shared by the V arms and the O ring
    if shape == "disc":
        return [(circle(h * 0.6, n), False)]
    if shape == "O":
        return [(circle(h * 0.6, n), False), (circle(h * 0.6 - d, n), True)]
    else:
        # Arms run from (+-w, -h/2) down to the tip at (0, +h/2); thickness d.
        w = h * 0.6
        # Tip roundness: 0 = sharp V, 1 = U.
        r = {"V": 0.0, "roundedV": 0.35, "U": 1.0}[shape]
        outer, inner = [], []
        for s in np.linspace(-1, 1, n // 2):
            # Blend a V profile |s| with a circular-ish U profile.
            v = abs(s)
            u = 1 - math.sqrt(max(0.0, 1 - s * s))
            prof = (1 - r) * v + r * u
            y = h / 2 - prof * h
            outer.append((s * w, y))
            inner.append((s * (w - d), y - d))
        pts = np.array(outer + inner[::-1])
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return [(pts @ rot.T, False)]


def texture(rng, fg):
    """Fur-like streak texture around the foreground intensity."""
    noise = rng.normal(0, 1, (SIZE * SS, SIZE * SS))
    # Directional smoothing -> streaks.
    k = 25
    streaks = np.cumsum(noise, axis=0)
    streaks = (streaks[k:] - streaks[:-k]) / math.sqrt(k)
    streaks = np.pad(streaks, ((0, k), (0, 0)), mode="edge")
    return np.clip(fg + streaks * 30, 0, 255)


def render(shape, scale, angle, contrast, version, rng):
    fg = CONTRASTS[contrast]
    big = SIZE * SS
    canvas = np.full((big, big), GRAY, dtype=np.float64)
    mask_img = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(mask_img)
    for pts, hole in shape_polygons(shape, scale, angle):
        xy = [((x + 1) / 2 * big, (y + 1) / 2 * big) for x, y in pts]
        if version == "outline":
            draw.line(xy + [xy[0]], fill=255, width=3 * SS, joint="curve")
        else:
            draw.polygon(xy, fill=0 if hole else 255)
    mask = np.asarray(mask_img, dtype=np.float64) / 255
    fill = texture(rng, fg) if version == "textured" else np.full_like(canvas, fg)
    canvas = canvas * (1 - mask) + fill * mask
    img = Image.fromarray(canvas.astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS)
    return img.convert("RGB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Image.new("RGB", (SIZE, SIZE), (GRAY,) * 3).save(OUT / "gray_baseline.png")
    rows = []
    for shape in SHAPES:
        angles = [0] if shape in ("O", "disc") else ORIENTATIONS
        for scale_name, scale in SCALES.items():
            for angle in angles:
                for contrast in CONTRASTS:
                    for version in VERSIONS:
                        name = f"{shape}_{scale_name}_{angle:03d}_{contrast}_{version}.png"
                        render(shape, scale, angle, contrast, version, rng).save(OUT / name)
                        rows.append(dict(file=name, shape=shape, size=scale_name,
                                         orientation=angle, contrast=contrast, version=version))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} shapes + gray baseline to {OUT}")


if __name__ == "__main__":
    main()
