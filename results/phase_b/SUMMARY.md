# Phase B: confirmatory result for the 4e:55 V-shape claim

Run once on Oct 28, 2026, after the freeze (`data/phase_b/FREEZE_MANIFEST.txt`, combined SHA-256 `1afb4907…49f549`; the manifest file's own SHA-256 is `e95e81ce…147113`). The full output is in `report.md`, with raw values in `activations.csv`.

## Pre-registered verdicts

| Test | Result | Verdict by the frozen rule |
| --- | --- | --- |
| **1. Pointed objects vs. round faces** | Round faces 639, pointed objects 159. Difference **−479** [−565, −394] | **Against H_V** (routes/category) |
| **2. Shape vs. texture (composites)** | Partial R²: shape 0.42 [0.31, 0.71], texture 0.16 [0.13, 0.47]. V+triangle − disc+square **+35** [+21, +48]. Fur − other **+8** [−6, +23] | **Inconclusive** (the CIs overlap) |
| **3. Round suppression** | 3a natural: round − boxy **−36** [−46, −26]; the rule needed ≤ −50 → fail. 3b synthetic: disc − square **−4.9** [−8.3, −1.6] → pass | **Partial** |
| **4. Ear occlusion** (upright, n = 12) | Median effect **+0.017** [−0.105, +0.235] | **H_category** (the ears carry no more than an equal patch of torso) |
| **5. Car views** | Three-quarter − head-on **+158** [+80, +232] | **Replicates** Phase A |

**Overall, by the frozen table: Mixed.** Test 1 points to routes/category, Test 2 is inconclusive, and Test 4 points to H_category.

## What this means for the claim

- **The strong claim fails.** "V-shape, regardless of category" is contradicted by the two tests that most directly separate shape from category.
  - Round-faced cats and dogs drive the neuron about 4× more than pointed spires, candles and obelisks (Test 1).
  - Covering the pointed ears does no more than covering an equal area of torso (Test 4).
- **The negative side holds in direction but is weaker than predicted.** Round objects sit below boxy ones and discs below squares; both differences are reliable. The natural-image gap (−36) fell short of the pre-registered −50.

## Not part of the verdict

These are observations from the reported tables, not tested by a pre-registered rule.

- **Among abstract composites, shape matters more than texture, and V's are highest.**
  - V's are the top shape for every texture.
  - Fur gives no advantage.
  - This is the one place the data lean toward the essay's V idea, but the whole effect is small: composites reach at most 115, against 600+ for real faces and cars.
  - Read it as "with no category cues present, a V is the best of these four shapes". It is not evidence that V-shape drives the neuron in natural images.
- **Caveats on sample sizes and raters.**
  - Several groups are small: 31 pointed objects, 12 occlusion sets, 1 fox-fur texture source.
  - Raters were AI models, not people (agreement ρ = 0.88, κ = 0.79).
