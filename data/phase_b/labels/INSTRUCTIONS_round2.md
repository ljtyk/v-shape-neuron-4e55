# Image annotation instructions (round 2)

You are annotating photos for a vision study. Judge each image only by what you see.

## The most important rule: never guess

Open every image with the Read tool and look at it before answering. If an image does not load, or the tool returns anything other than the picture (an error, or a placeholder such as "[media removed ...]"), write only:

```
{"id": "<image_id>", "could_not_open": true}
```

**Never** write an answer for an image you have not actually seen.

## Files

- Your list is a text file with one line per image: `<image_id><TAB><task>`. The task is `earzoom`, `body`, or `texbox <surface>`.
- Each image is at `images_r2/<image_id>.png` (in the same folder as this file).
- Every image has a light yellow 10×10 grid. Columns are **A–J** (left to right) and rows **0–9** (top to bottom). Boxes are written `TOPLEFT:BOTTOMRIGHT`, inclusive, e.g. `C3:E5`.
- **Do not open any other files or folders.**

## For every image you can see

- **main_object**: name the main animal or object in 1–3 plain words.

## Task `earzoom` (an animal, often shown close up)

- **ear_boxes**: one box per visible ear, as **tight** as possible around **the ear only**. Don't include the eyes, the rest of the head, or the background beyond a small margin. Example: `["B1:C2", "G1:H2"]`.
- If no ear is clearly visible, set `ears_not_visible` to true and `ear_boxes` to `[]`.

## Task `body`

- **body_box**: one box on the animal's **torso**: the chest, back or flank, not the head, legs or tail. Keep it inside the animal.
- If the torso is not visible, set `body_box` to null and `no_body` to true.

## Task `texbox <surface>` (surface is `fur`, `car_paint` or `stone`)

- **texture_box**: one box over an area that shows **only that surface**. No eyes, nose, mouth, ears, paws, wheels, windows, lights, lettering, object edges, background or plants. Make it as large as possible while staying clean; at least 2×2 cells if you can.
- If no clean area of at least 2×2 cells exists, set `texture_box` to null and `no_clean_area` to true.

## Output

One JSON object per line (JSON Lines), in list order, to the output file you were given. Always include all of these keys; use null or [] where a task doesn't apply:

```
{"id": "512345", "could_not_open": false, "main_object": "fox", "ear_boxes": ["D2:E3", "G2:H3"], "ears_not_visible": false, "body_box": null, "no_body": null, "texture_box": null, "no_clean_area": null}
```

Open one image, write its line under that image's ID, then move to the next. When finished, reply with three numbers: images answered, images that could not be opened, and total lines written.
