# Image annotation instructions (v2)

You are annotating photos for a vision study. Judge each photo on its own, only by what you see.

## The most important rule: never guess

Open every image with the Read tool and look at it before annotating it.

If an image does not load, or the tool returns anything other than the picture (an error, or a placeholder such as "[media removed ...]"), you **must** write:

```
{"id": "<image_id>", "could_not_open": true}
```

and nothing else for that image. **Never** write an annotation for an image you have not actually seen. A missing annotation is harmless; an invented one ruins the study. If several images in a row fail to load, stop and say so in your final reply.

## Files

- Your image list is a text file with one line per image: `<image_id><TAB><tasks>`. Tasks are joined with `+` and come from: `rate`, `ears`, `object`.
- Each image is at `images/<image_id>.png` (in the same folder as this file).
- Each image has a light yellow 10×10 grid. Columns are labelled **A–J** (left to right) and rows **0–9** (top to bottom). A cell is written like `C4`.
- **Do not open any other files or folders.**

## For every image you can see

- **main_object**: name the main object in 1–3 plain words, e.g. "dog", "red car", "church tower", "golf ball". The main object is the one thing the photo is mostly about.

## Tasks

### `rate`

1. **pointedness**, an integer 1–5, for the main object's overall shape:
   - 1 = clearly round (a ball, a flat round face, a round clock)
   - 2 = mostly round or rounded
   - 3 = neither clearly round nor clearly pointed (e.g. boxy)
   - 4 = mostly pointed
   - 5 = clearly pointed or V-shaped (a spire, a pointed snout with tall pointed ears, a wedge)
2. **view** of the main object, one of `frontal`, `three-quarter`, `side`, `rear`. For objects with no front (a ball, a plate), use `frontal`.
3. **pointed_box**: the grid box around the most pointed feature of the main object, written `TOPLEFT:BOTTOMRIGHT` (inclusive), e.g. `C2:E4`. If there is none, set `no_pointed` to true and `pointed_box` to null.
4. **too_small**: true if the main object fills less than a quarter of the frame.

### `ears` (in addition to `rate`)

- **ear_boxes**: one grid box per visible ear, covering the whole ear with a little margin, e.g. `["B1:C3", "G1:H3"]`.
- If ears are not clearly visible, set `ears_not_visible` to true and `ear_boxes` to `[]`.

### `object` (these do NOT get `rate`)

- **object_box**: one grid box tightly around the main object.

## Output

One JSON object per line (JSON Lines), in list order, written to the output file you were given. For a seen image, use exactly these keys and leave none out; use null or [] where a task doesn't apply:

```
{"id": "829381", "could_not_open": false, "main_object": "dog", "pointedness": 2, "view": "frontal", "no_pointed": false, "pointed_box": "C2:D4", "too_small": false, "ears_not_visible": false, "ear_boxes": ["B2:D6", "G2:H6"], "object_box": null}
```

Write the file as you go. When finished, reply with three numbers: images annotated, images that could not be opened, and total lines written.
