# Validating the V-Shape Neuron (InceptionV1 4e:55)

> **Superseded.** This was the starting design. The exploratory phase (Phase A) changed the hypotheses and tests: see [results/explore/FINDINGS.md](results/explore/FINDINGS.md). The confirmatory plan is now [phase-b-design.md](phase-b-design.md).

Oct 17, 2026 · @David

## Background

The goal is to test, with a blind and pre-registered design, whether InceptionV1 neuron mixed4e:55 is a shape detector (pointed vs. round) or a category detector (cat OR car).

Neuron 4e:55 is the field's canonical polysemantic neuron. It was cited in *Feature Visualization* (2017) and *Zoom In* (2020) as responding to an arbitrary mix of cat faces and car fronts.

In [Neural Networks Are Already Clean](https://already-clean.goodfire.com/), Nick Cammarata (a *Zoom In* co-author) reinterprets it as a single V-shape detector. His evidence:

- Pointed things drive it and round things push it negative, regardless of category. Dogs appear on both sides depending on shape.
- Its strongest upstream inputs are neurons for fox snouts (4d:493), white-wolf faces (4d:477) and head-on van fronts (4d:201).
- Its downstream readers include neurons for candle flames (5a:94), coyote faces (5a:352) and martini glasses (5a:317).
- Its negative pole is O-shaped: clocks, golf balls, tennis balls, round dog faces.

The concern: this evidence is hand-picked, and he came to the neuron already looking for a single explanation. A shared property can usually be found after the fact. This test checks whether the V-shape label predicts responses it was not built from.

## Hypotheses

Three hypotheses, because shape and category are not the only explanation that fits the original examples.

- **H_cat (category):** the neuron responds to cats and cars as categories. Shape within a category should barely matter. Because its strongest inputs are fox and wolf neurons, the fair version of H_cat is "cat-like carnivore face OR car front". Under this version, pointed-snout dogs and foxes count as in-category, so they cannot separate H_cat from H_shape.
- **H_shape (Cammarata):** the neuron responds to pointed / V-shaped structure and is pushed negative by round structure, whatever the category.
- **H_frontal (confound):** the neuron responds to frontal views. Cat faces and car fronts are both seen head-on. A pointed V and a frontal view are correlated in the original examples, so a positive H_shape result is only convincing if it survives this control.

Mixed outcomes are possible (e.g. shape matters, but more strongly within cats and cars). The analysis plan handles that.

### Why blind

The risk being tested is post-hoc storytelling. So the shape and view labels for every image are assigned **before** any activation is recorded, by someone who does not see activations. Predictions and decision thresholds are fixed in this doc before the run.

## Stimulus set

About 400 natural images plus a synthetic set, crossing category with shape and view so the three hypotheses make different predictions.

| Group | Examples | Target count |
| --- | --- | --- |
| Cat, pointed, frontal | Cat face head-on, ears and chin forming a V | 30 |
| Cat, round, frontal | Persian or flat-faced cat head-on, curled cat facing camera | 30 |
| Cat, pointed, side view | Cat profile with pointed ears and muzzle | 20 |
| Car, pointed, frontal | Sharp V-shaped grille or nose, head-on | 30 |
| Car, round, frontal | VW Beetle, Fiat 500, rounded bubble car head-on | 30 |
| Car, pointed, side view | Wedge-shaped sports car in profile | 20 |
| Dog, pointed | Fox-like breeds, pointed snouts (Shiba, Doberman) | 30 |
| Dog, round | Pug, bulldog, round-faced puppies | 30 |
| Other, pointed | Bullet train nose, arrowhead, flame, church spire, martini glass | 60 |
| Other, round | Clock, golf ball, tennis ball, plate, round lamp | 60 |
| Other, frontal round faces | Owl, clock face, round-faced dolls | 30 |

**Synthetic set (468 images, `scripts/make_synthetic.py`):** V's and O's, plus intermediate shapes (U, rounded V), at 3 sizes, 8 orientations (V, rounded V, U only) and 2 contrasts (dark or light shape), on a neutral gray background, rendered at the model's input size (224×224). The O is a ring with the same band width as the V's arms, so V vs. O is not confounded with ink area; a filled disc is included as a separate shape. Each shape comes in three versions: outline, filled silhouette, and textured fill (fur or metal texture clipped to the shape). Mixed4e often ignores bare line drawings, so outlines alone would make a null result uninformative.

**Gray baseline:** also record the activation on a plain gray 224×224 image. Pre-ReLU activations are not centred at zero, so "negative" in this doc means below this baseline.

**Source and pool (`scripts/imagenet_val.py`):** natural images come from the ImageNet validation set. A fixed-seed candidate pool of 611 images is drawn by class (cats, cars, dogs, pointed and round objects, owls), held out from the verification run, and cropped to the largest ImageNet object box. Class only guarantees enough of each kind; the pointed/round and view groups in the table above are formed from the blind ratings (pointed ≥ 3.5, round ≤ 2.5).

### Blind labeling

Two raters, who never see activations, score every natural image before the run:

- **Pointedness:** 1 (clearly round) to 5 (clearly pointed / V-shaped), for the most salient object.
- **View:** frontal / three-quarter / side.
- **Pointed region:** a bounding box around the most pointed feature (ear tips, chin, grille point, flame tip), or "none" if there isn't one.

Use the mean pointedness score as a continuous variable. Report inter-rater agreement (Spearman ρ); if ρ < 0.7, refine the labeling instructions and relabel before running.

The activation is a max over spatial positions, so a small pointed patch anywhere in the frame can drive it even when the main object is round. To keep labels and measurement aligned, crop each natural image to its main object before measuring. Keep the pointed-region boxes for the spatial check.

### Freezing

Once labeling is done, write the image list, labels, crops, and this doc's predictions and thresholds to a single manifest. Record its SHA-256 hash and timestamp (e.g. in a commit or a message to a third party) before running any activations.

## Procedure

Use the exact InceptionV1 checkpoint the Distill papers used, and confirm it before running anything.

1. **Load the right model.** Use the original InceptionV1 weights from the Lucid model zoo, or a PyTorch port of those same weights (e.g. Lucent). Do **not** use torchvision's `googlenet`: it is a separate training run, so channel 55 there is a different neuron.
2. **Verify the neuron.** Run the full ImageNet validation set (50,000 images) and take mixed4e channel 55's top 20. Pass if at least 12 of the 20 are cat faces or car fronts; otherwise stop and recheck the checkpoint. Save the top 100 with their labels as data. If foxes, wolves or pointed non-animals show up there, that is already evidence for the "carnivore face" or shape reading.
3. **Hook the layer.** Record mixed4e channel 55 **before** the ReLU, so the negative pole is visible.
4. **Measure each image.** Use the model's own preprocessing. Record (a) max activation over spatial positions, (b) mean activation, and (c) the full spatial map.
5. **Run blind.** Activations are recorded only after the manifest in Freezing is hashed.
6. **Spatial check.** For the top ~30 positive images, overlay the spatial map. H_shape predicts the peak sits inside the raters' pointed-region box (ear tips, chin, grille point), not on the object as a whole. Report the fraction of peaks that fall inside the box, compared against the box's share of the image area as a chance rate.

## Causal test: shape edits

In natural images, shape and category always come together. The regression can only estimate how much each one matters; it can't separate them. The strongest test holds category fixed and changes only shape.

**Edited pairs (≈60 pairs, drawn from the natural set before activations are seen):**

| Edit | Source | Example |
| --- | --- | --- |
| Round off | 20 pointed cats, 10 pointed cars | Inpaint cat ear tips and chin rounded; soften a V-grille into a curve |
| Sharpen | 10 round cats, 10 round cars | Inpaint pointed ears onto a Persian; give a Beetle a wedge nose |
| Control edit | 10 images, any group | An edit of the same size that leaves the shape unchanged (e.g. recolor the same region) |

Rules:

- Edit only the pointed-region box. Keep colour, texture and lighting as close to the original as the inpainting allows.
- The same blind raters re-score pointedness on the edited images. Keep a pair only if its score moves by at least 1.5 points in the intended direction and both raters still give the edited image the original category.
- Control edits measure how much any inpainting changes the activation on its own. Report shape edits relative to this artifact level.

**Morph series (optional):** for 10 silhouettes, interpolate V → U → O in 7 steps with the texture held constant. H_shape predicts the activation decreases steadily along the series.

H_shape predicts rounding-off lowers activation and sharpening raises it, by more than the control edits. H_cat predicts neither edit matters beyond the controls, since category is unchanged. H_frontal predicts no change, since view is unchanged.

## Predictions

The groups where the hypotheses disagree are the ones that decide the result; cat-pointed-frontal and car-pointed-frontal agree under all three and only confirm the neuron was loaded correctly.

| Group | H_cat | H_shape | H_frontal |
| --- | --- | --- | --- |
| Cat, pointed, frontal | High | High | High |
| Cat, round, frontal | High | Low or negative | High |
| Cat, pointed, side view | High | High | Low |
| Car, pointed, frontal | High | High | High |
| Car, round, frontal | High | Low or negative | High |
| Car, pointed, side view | High | High | Low |
| Dog, pointed | Low (High under "carnivore face") | High | Depends on view |
| Dog, round | Low | Low or negative | Depends on view |
| Other, pointed | Low | High | Depends on view |
| Other, round | Low | Low or negative | Depends on view |
| Other, frontal round faces | Low | Low or negative | High |
| Synthetic V | Low | High | Low |
| Synthetic O | Low | Below gray baseline | Low |
| Edited: rounded off | No change | Drops | No change |
| Edited: sharpened | No change | Rises | No change |

The decisive comparisons are:

1. **Shape edits vs. control edits**, the only direct causal test.
2. **Round cats and cars vs. pointed "other" objects** (no animals or vehicles). H_cat says the first group wins; H_shape says the second does.

The dog groups are reported, but they don't decide the result. Pointed dogs score high under both H_shape and the carnivore-face version of H_cat.

## Analysis and decision criteria

Fit one regression on the natural images and read the verdict off fixed thresholds set here, before the run.

**Model:** max activation (minus gray baseline) ~ target category (1 if cat or car, else 0) + pointedness (blind mean score) + view (frontal / three-quarter / side) + target category × pointedness.

Target category is a single contrast, not a 4-level factor, so it and pointedness each use one degree of freedom and their partial R² can be compared fairly. As a secondary analysis, refit with dogs counted as target category (the "carnivore face" version of H_cat).

Report each factor's partial R² (variance explained after the others), with 95% bootstrap confidence intervals (1,000 resamples).

**Edited pairs:** for each pair, compute Δ = activation(edited) − activation(original). Compare the shape-edit Δs (with sign flipped for sharpening, so the H_shape prediction is always negative) to the control-edit Δs using a permutation test (10,000 permutations). Report the effect as the median Δ in units of the natural set's standard deviation.

**Power:** before the run, simulate the regression with the planned group sizes and an effect where pointedness explains 10% of variance. If power to detect it is below 80%, increase group sizes before freezing.

### Verdicts

- **H_shape supported** if all hold:
  - shape edits move the activation in the predicted direction, beyond control edits (p < 0.01), with median |Δ| ≥ 0.5 SD;
  - pointedness has a larger partial R² than target category, and its CI excludes zero;
  - pointedness stays significant with view in the model (rules out H_frontal);
  - mean activation on "other, pointed" exceeds the mean on round cats and cars combined.
- **H_cat supported** if shape edits do not beat control edits, target category has the largest partial R², and pointedness explains < 5% of variance within cats and within cars.
- **H_frontal supported** if view has the largest partial R² and pointedness loses significance once view is included.
- **Mixed** otherwise. Examples: edits work only within cats and cars, or the target category × pointedness interaction is large (the neuron is shape-driven mainly inside its original categories). Report which pattern, without forcing a winner.

Synthetic results are reported separately and are not required for any verdict. The H_shape prediction is V > O, with O's below the gray baseline. A neuron can respond to pointed natural objects but ignore drawn shapes, which would weaken H_shape but not refute it.

## Pitfalls and extensions

The biggest risks are loading the wrong neuron and letting a confound pass as shape.

**Pitfalls**

- **Wrong checkpoint.** torchvision's `googlenet` is not the Distill model; channel indices will not match. The verification step in Procedure is mandatory.
- **Other confounds.** Pointed objects may also differ in texture (whiskers, fur, chrome), contrast, or size in frame. Spot-check the top and bottom 20 images for any shared non-shape property, and crop to normalize object size where possible.
- **Peeking.** Looking at activations while choosing images or edits reintroduces the bias this test exists to remove. Freeze and hash the manifest first.
- **Inpainting artifacts.** The edits can add blur or texture seams that change the activation regardless of shape. The control edits measure this; if control |Δ| is as large as the shape-edit |Δ|, the edit test is inconclusive rather than negative.
- **Small groups.** With ~30 images per group, single outliers move means a lot. Report medians alongside means.

**Extensions (if the result is clear)**

- **Circuit check.** Rank the neuron's top upstream inputs and downstream readers by weight, and score their top images for pointedness too. H_shape predicts a pointedness-rich neighborhood.
- **Generalization across models.** Search for the most V-like channel in a ResNet-50 or ConvNeXt (highest correlation with the blind pointedness scores) and rerun the same set. Finding a similar neuron elsewhere is evidence the concept is a real joint in the data.
- **Repeat on the car-into-dogs circuit.** Apply the same blind design to Cammarata's second reinterpretation (shared pose rather than storage).
