# Phase B: Confirmatory Test of the 4e:55 V-Shape Claim

Draft · Oct 27, 2026 · supersedes the original design in [v-shape-neuron-validation.md](v-shape-neuron-validation.md)

## Background

Phase A was exploratory. It ran on 49,389 ImageNet validation images that are not in the held-out pool. Full results are in [results/explore/FINDINGS.md](results/explore/FINDINGS.md). In short:

- **Holds up:** his circuit facts. The fox-snout neuron 4d:493 is by far the strongest input (weight 0.31, about 10 SD above typical), and the coyote neuron 5a:352 is the strongest reader.
- **Holds up:** round things push 4e:55 negative. Every method agreed: natural images, feature visualization, geometric primitives, and the unconstrained pixel search.
- **Does not hold up:** "V-shape, whatever the category" as what drives the positive side.
  - The top class is the round-faced Persian cat.
  - Pointed buildings rank near the bottom even when cropped tight.
  - With the cat/fox inputs removed, the neuron behaves like a vehicle detector.
  - The simplest image the optimizer can find is always a furry face.
  - A forced V beats an O but only ties two crossing bars.
- **Cars come in through their own detectors** (about 75% of their activation). About 30% comes through the fox detector. That part isn't colour, a car "face", or a snout. Head-on cars drive it least.

All of this came from looking at data and choosing the next step based on what came back. Phase B tests the main conclusions on images neither the model nor we have touched, with predictions and thresholds fixed in this document before any measurement.

## Hypotheses

- **H_V (Cammarata):** pointed structure drives 4e:55 up and round structure drives it down, whatever the category. Shape is the variable; category is incidental.
- **H_routes (Phase A reading):** two category routes feed the positive side: small-carnivore faces (with fur), and vehicles. Each has a mild shape preference within its category (e.g. upright over floppy ears). Round structure suppresses it.
- **H_category:** cat + car (+ fox) as categories. Shape within a category doesn't matter.

H_routes and H_category agree on Tests 1 and 2 and differ on Test 4. H_V differs from both on Tests 1, 2 and 4.

## Stimuli

### Sources (all unseen by the model)

1. **The held-out pool:** 611 ImageNet validation images, drawn with a fixed seed before Phase A and never run through the model (`scripts/imagenet_val.py`).
2. **ImageNet-V2 (matched-frequency):** 10 new images per ImageNet class, collected separately from the original set. *Needs download approval (~1.2 GB).*
3. **Synthetic composites** for Test 2, built by a fixed rule (below).
4. **Inpainted edits** of held-out images for Test 4. *Needs inpainting-model approval (~5 GB).*

No stimulus is chosen or rejected based on activations. Rejections happen only by the rater rules below, before measurement.

### Groups

| Test | Group | Classes (ImageNet names) | Inclusion (blind rater rule) | Target n |
| --- | --- | --- | --- | --- |
| 1 | **P: pointed non-face, non-vehicle objects** | obelisk, bell cote, church, candle, volcano, mountain tent | Mean pointedness ≥ 3.5; object fills ≥ ¼ of frame | 40 |
| 1 | **F: round-faced small animals** | Persian cat, Pekinese, Japanese spaniel, Shih-Tzu | Pointedness ≤ 2.5; view frontal or three-quarter; face visible | 40 |
| 3 | **R: round objects** | golf ball, tennis ball, ping-pong ball, wall clock, analog clock, plate, orange | Pointedness ≤ 1.5; fills ≥ ¼ of frame | 40 |
| 3 | **B: boxy objects** | carton, crate, chest, safe, washer, monitor, refrigerator, microwave, dishwasher, television, wardrobe, file | Pointedness 2.5–3.5 ("neither round nor pointed"); fills ≥ ¼ of frame | 40 |
| 4 | **Ear-edit sources** | red fox, Arctic fox, Pomeranian (upright ears); basset, clumber, Saint Bernard (floppy ears). ImageNet-V2 only. | Ears clearly visible (not ticked "not visible" by either rater); face frontal or three-quarter | 20 + 20 |
| 5 | **Cars by view** | all car, van and pickup classes in the pool and ImageNet-V2 | View label agreed by both raters | as available (~300 images; ≥ 15 head-on, else underpowered) |

If a group falls short of its target after rater filtering, it is reported with its actual n. No replacement images are drawn after measurement starts.

**Candidates as built** (`scripts/phase_b_build.py`, seed 20261027): 801 images in total. Pointed objects 110, round-faced animals 90, round objects 133, boxy objects 120, upright-eared 26, floppy-eared 30, cars 276, texture sources 16. Every candidate in a group's classes from both sources is included; inclusion is decided only by the rater rules. Texture-source images are drawn first and excluded from every other group.

### Test 2 composites (shape × texture cue conflict)

- **Shapes** (filled silhouettes): downward V (chevron), downward-pointing triangle, disc, square. Area matched within each size.
- **Textures:** fox fur, cat fur, car body paint, brick/stone, plain mid-gray.
  - Each texture comes from 4 source images, chosen at random from ImageNet-V2 images of red fox, tabby, sports car and stone wall.
  - Each patch is a 96 px crop (in the model's 224 px frame) centred on the mean of the two raters' object boxes. A patch is rejected only if a rater marks it as "not the named texture" in round 2.
- **Layout:** 3 sizes (40%, 60%, 80% of frame width), on mid-gray, 224 px.
- **Total:** 4 shapes × 5 textures × 3 sizes × 4 texture sources = **240 images**.

**Changes made Oct 28, before freezing, after checking the first build by eye (no activations involved):**
- **Patch location.** Cutting patches at the centre of the object box captured whole car fronts with wheels, cat faces with eyes, and a fox face. Those would test faces and objects, not texture. Patches now come from a **texture-only box** each rater draws in round 2: an area showing only fur, car paint or stone, with no eyes, face, ears, paws, wheels, windows, lights, lettering, object edges or background. The patch is the largest square inside the overlap of the two raters' boxes. A source is dropped if the boxes don't overlap, if the overlap is smaller than 2×2 grid cells, or if either rater reports no clean area.
- **Untextured control fill.** A plain mid-gray shape is invisible on the mid-gray background. The untextured control is now a flat dark-gray fill (value 70).
- **Count.** The untextured fill has no source images, so it gets one copy per shape and size, not four duplicates. The total is at most 4 shapes × 3 sizes × (4 textures × 4 sources + 1) = **204 images**, fewer if sources are dropped.
- **Shape sizes.** All four shapes at a given size are matched in area to 60% of a square of that width (`scripts/phase_b_round2.py`).

### Blind labeling

**Change (Oct 27, before freezing): the raters are two AI models, not two people.** Human labeling at about 2–2.5 hours per rater wasn't feasible. Instead:
- **Rater A** is Claude Sonnet and **Rater B** is Claude Opus. Different models are used so their errors are less correlated.
- Each rater's 801 images are split across 4 fresh agent instances with no context. They get only the rating instructions (`INSTRUCTIONS.md`, archived with the labels), their image list, and the images. They are told nothing about the neuron, the hypotheses, or the study's purpose, and are told not to open any other file. This makes them more blind than the study owner, who has seen Phase A.
- Boxes are given as cells on a labelled 10×10 grid drawn over each image, so they are coarse (10% of the frame per cell). This is acceptable for pointed-feature boxes, object boxes, and ear masks, which are enlarged anyway; round 2 checks every edit.
- The agreement thresholds and every other rule below are unchanged.
- Limitation: AI raters may share biases that humans don't. Agreement between two AI models is not the same as agreement between two people, and the report states this.

**The AI-rater attempt failed (Oct 27) and none of its output is used.** All 8 agents had most image reads blocked by an image request limit (153–244 blocked reads each). Several agents then wrote annotations for images they never saw; four later said so, and the rest gave no sign either way. Since invented and genuine lines can't be told apart, all output was quarantined (`scratchpad/raters/out_QUARANTINED_do_not_use`) and excluded. No activations had been computed, so blinding is unaffected.

**The second, safeguarded attempt succeeded (Oct 27–28), and its output is what Phase B uses.**

How it ran:
- Batches of 50 images, two agents at a time (Sonnet = rater A, Opus = rater B).
- Instructions (`data/phase_b/labels/INSTRUCTIONS_v2.md`) add an explicit "could not open" field and a `main_object` name for each image.

Every batch had to pass `scripts/verify_rater_batch.py`:
- **Viewing check:** every annotated image must appear in the agent's own transcript as a successful image read.
- **Name check:** at least 75% of `main_object` names must match the hidden class key.

Outcomes:
- All 801 images were labelled by both raters, from images verifiably seen.
- One batch was rejected and redone after an unviewed image and apparently swapped labels.
- One image set (batch 06) was accepted below the 75% bar, because three agents on two models independently reproduced the same names. Details are in `data/phase_b/labels/VERIFICATION_LOG.md`.

**Agreement (pre-registered thresholds):**
- Pointedness: Spearman ρ = 0.883 (≥ 0.70) → pass.
- View: Cohen's κ = 0.788 (≥ 0.60) → pass.
- `too_small`: 94% raw agreement.

Inclusion rules use the mean of the two raters' pointedness. Views are used only where both raters agree.

The original plan, kept for reference: two raters (the study owner plus one colleague) use `labeling_b/label.html`. They never see activations. Raters see exactly the square the model sees: short side resized to 224, centre-cropped, shown at 448 px.

**Round 1** (the 801 candidates; roughly 2–2.5 hours per rater). Rated per image:

- **Pointedness**, 1 (clearly round) to 5 (clearly pointed), for the main object.
- **View:** frontal / three-quarter / side / rear.
- **Pointed region box**, or "none".
- **New: object fills less than ¼ of the frame** → excluded.
- **Ear boxes** (Test 4 sources only): one box per visible ear, or "ears not clearly visible".
- **Object box** (texture sources only).

**Round 2** (texture boxes only, now that Test 4 uses occlusion; still no activations). The same two AI raters and the same verification are used (`INSTRUCTIONS_round2.md`):
- Each texture source is shown with the grid, plus the name of the surface to box.

Round-2 items:
- **Texture patches:** "is this the named texture" (yes/no).

Agreement: Spearman ρ for pointedness, Cohen's κ for view. If ρ < 0.7 or κ < 0.6, refine the instructions and relabel before freezing.

## Measurement

- **Model and neuron:** the Lucid InceptionV1 weights via Lucent, mixed4e channel 55 read before its ReLU (`scripts/neuron.py`), as in Phase A.
- **Preprocessing:** short side resized to 224, centre crop, `x*255 − 117`. Images from both sources are treated identically and not cropped to an object box, since ImageNet-V2 has no boxes. Small objects are excluded by the rater rule instead. (In Phase A, pool images were object-cropped; Phase B does not do that.)
- **Per image:** spatial **max** (primary for Tests 1, 2, 4, 5) and spatial **mean** (primary for Test 3, since the negative pole showed most clearly in means). Values are relative to the gray-image baseline (−0.2).
- **Statistics:** differences in group means, with 95% bootstrap CIs (10,000 resamples). Each test's verdict uses only its pre-specified comparison below.

## Tests and decision rules

### Test 1: Pointed objects vs. round faces

Comparison: mean max(P) − mean max(F).

| Outcome | Verdict |
| --- | --- |
| Difference ≥ +150, CI excludes 0 | Supports **H_V** |
| Difference ≤ −150, CI excludes 0 | Supports **H_routes / H_category** |
| Otherwise | Inconclusive |

(Phase A reference values, which do not count toward the result: Persian class mean 976, obelisk 120.)

### Test 2: Shape vs. texture

Regression on the 240 composites: max ~ shape + texture + size + shape × texture. Read off partial R² for shape and for texture, with bootstrap CIs.

| Outcome | Verdict |
| --- | --- |
| Shape partial R² > texture partial R², CIs not overlapping; and V and triangle > disc and square | Supports **H_V** |
| Texture partial R² > shape partial R², CIs not overlapping; and fur textures > others | Supports **H_routes / H_category** |
| Otherwise | Inconclusive |

Also reported, but not part of the verdict: the fur-texture-in-V cell vs. the fur-texture-in-disc cell, i.e. whether shape matters once texture is right.

### Test 3: Round suppression (natural and synthetic)

- **3a (natural):** mean of spatial-mean(R) − mean of spatial-mean(B). Prediction (all hypotheses): ≤ −50, CI excluding 0.
- **3b (synthetic):** disc vs. square in the Test 2 set, matched on texture and size. Prediction: disc lower, CI excluding 0.

Confirming both supports the negative side of his claim. Failing both means the round-suppression finding does not replicate, and Phase A's conclusion on the negative side is withdrawn.

### Test 4: Ear occlusion

**Changed Oct 28, before freezing.** The planned inpainted ear-shape edits were built and failed by eye. Stable Diffusion v1.5 inpainting kept foxes pointy-eared and bassets floppy-eared, even with masks extended where the new ears would grow and with higher guidance (`stimuli/phase_b/edits/`, unused). Test 4 now asks the same question ("how much of the drive do the ears carry?") by occluding them. No activations were involved in this change.

**Sources:** the ear-edit sources that pass the inclusion rules. Both raters must see the ears and rate the face frontal or three-quarter, which gives 15 upright-eared and 22 floppy-eared images.

**Three versions of each source image** (the model's 224 px square):
- **Original.**
- **Ear occlusion:** each ear covered with flat mid-gray (128), using the ear boxes from the **round-2 zoomed pass**. Round-1 boxes, on the 10% full-image grid, often covered the whole head. That would turn the test into "occlude the face" and confound H_V with H_routes. Round-2 boxes are drawn on a gridded crop around the head (about 2× the span of the round-1 ear boxes, at least 40% of the frame), so they're finer. Each rater's box for an ear is mapped back to the full image, and the union of the two raters' boxes is used.
- **Control occlusion:** gray patches of the same shapes and total area, moved so their centroid sits at the centre of the mean of the two raters' **torso boxes** (round 2). The same area is covered on the animal's body instead of the ears. Round-1's rule ("move down 1.5× box height") often put the patch on grass or snow, so it was dropped.

**Exclusions** (decided before measurement):
- Either rater marks the ears not visible in round 2.
- Either rater marks no visible torso.
- Clipping at the image edge leaves the control covering less than 80% of the ear-occlusion area.
- The control patch overlaps the ear patch.

**Effect per image:** drop_ear − drop_control, where drop = (original max − occluded max) / original max. Positive means the ears carry more of the drive than an equal-sized part elsewhere.

**Verdict:** on the **upright-eared group**, using the median effect with a bootstrap CI.

| Median effect (upright group) | Verdict |
| --- | --- |
| ≥ 0.50 | Supports **H_V**: the pointed ears carry most of the drive |
| 0.10–0.50, CI excluding 0 | Supports **H_routes**: ear shape matters inside the category but doesn't dominate |
| < 0.10, or CI including 0 | Supports **H_category** |

The floppy-eared group is reported alongside as a comparison, not as part of the verdict. H_V predicts its effect is near zero (round ears), so the upright effect should clearly exceed the floppy one.

### Test 5: Car views (replication)

Comparison: mean max(three-quarter) − mean max(head-on). Views are taken only where both raters agree.

| Outcome | Verdict |
| --- | --- |
| Difference ≥ +150, CI excludes 0 | Replicates Phase A (head-on cars drive it least) |
| Difference ≤ 0 | Fails to replicate. Supports a V-grille reading if head-on is highest. |
| Otherwise | Inconclusive |

## Overall verdict

| Pattern across Tests 1, 2, 4 | Conclusion |
| --- | --- |
| All three favour H_V | Cammarata's claim is supported |
| Tests 1 and 2 favour routes/category; Test 4 in the 10–50% band | H_routes: two category routes with a within-category shape preference |
| Tests 1 and 2 favour routes/category; Test 4 under 10% | H_category: the original "cat + car" reading |
| Any other pattern | Mixed; report which tests point where, without forcing a winner |

Test 3 is reported separately as the test of the negative side. Test 5 is reported separately as a replication.

## Procedure

1. **Downloads:** ImageNet-V2 and the inpainting model (approved Oct 27).
2. **Build candidates:** done (`scripts/phase_b_build.py`).
3. **Round 1 labeling:** both raters. Check agreement; relabel if below threshold.
4. **Generate** the ear edits and Test 2 composites from the round-1 boxes. **Round 2 labeling:** review the edits and patches. No activations are computed on any of these.
5. **Apply the inclusion rules** and write the final image list.
6. **Freeze:** write one manifest (image list, labels, this document) and record its SHA-256 hash with a timestamp. Activations are not computed before this step.
7. **Run once:** compute all measurements in a single script run.
8. **Report every test**, including inconclusive and failed ones. Any change to this plan after freezing goes in a deviations section, with the reason.

## Pitfalls

- **Inpainting artifacts.** Edits can add blur or texture seams. The control edits measure this. If control edits alone change activation by more than the shape edits, Test 4 is inconclusive, not negative.
- **Small head-on sample.** Head-on cars are about 6% of car photos. If fewer than 15 survive rater agreement, Test 5 is reported as underpowered.
- **Rater knowledge.** The raters know the hypotheses. Labels are about the image (pointedness, view), never about expected activation, and no activations are visible until after freezing.
- **ImageNet-V2 is slightly harder than the original validation set.** Absolute activations may run a little lower. All tests compare groups within Phase B, so this affects every group equally.
- **Phase A thresholds.** The ±150 and ±50 thresholds were chosen with knowledge of Phase A scales: about 15% of the Persian class mean, and about the spread among round-object classes. They are fixed here and do not change after measurement.

## Freeze record

- **Final stimulus list** (`data/phase_b/final_stimuli.csv`, from `scripts/phase_b_final_list.py`): 653 stimuli.

  | Group | n |
  | --- | --- |
  | Pointed objects | 31 (target 40; reported at actual n) |
  | Round faces | 71 |
  | Composites | 108 (8 clean texture sources: car paint 2, cat fur 2, fox fur 1, stone 3; plus the flat fill) |
  | Round objects | 53 |
  | Boxy objects | 100 |
  | Upright occlusion sets | 12 |
  | Floppy occlusion sets | 4 |
  | Cars, three-quarter | 135 |
  | Cars, head-on | 27 |
  | Cars, side / rear | 64 / 16 (not used) |

- **Analysis code** (`scripts/phase_b_run.py`) implements the decision rules above. It was dry-run on random fake activations (`results/phase_b_DRYRUN/`). Every test came out null or inconclusive, as it should on noise. Known property: bootstrap partial R² is biased upward for both shape and texture equally, so the shape-vs-texture comparison is unaffected.
- **Freeze:** `data/phase_b/FREEZE_MANIFEST.txt` lists the SHA-256 of every file the result depends on, plus a combined hash and timestamp. No activation on any Phase B stimulus had been computed before this point.
