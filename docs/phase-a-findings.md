# Phase A findings: 4e:55 (exploratory)

These are exploratory results on 49,389 ImageNet validation images. The 611-image held-out pool was never run. Everything here is correlational, and the hypotheses for Phase B are drawn from it, so none of it counts as confirmation.

Raw outputs: `report.txt`, `inputs.csv`, `readers.csv`, `class_profile.csv`, `orientation.txt`, `top30_peaks.jpg`, `bottom30_by_*.jpg`.

## Cammarata's evidence, checked

| Claim | Result |
| --- | --- |
| Strongest input is a fox-snout neuron (4d:493) | **Holds.** Rank 1 of 528, weight 0.31. That is about 10 SD above typical weights and 2.4× the next input. |
| Wolf (4d:477) and van (4d:201) neurons are strong inputs | **Partly.** They rank 8th and 15th. The others in the top 15 are not pointed-themed: tiger beetles and mantises, barbells, paper towels, guinea pigs and hamsters, Maltese dogs, whiskey jugs. |
| Coyote-face neuron (5a:352) reads it | **Holds.** It is the strongest reader of 832. |
| Flame neuron (5a:94) reads it | **Holds, but further down.** Rank 26. On validation images it fires on candles *and* red foxes. |
| Martini-glass neuron (5a:317) reads it | **Rank 21.** On validation images its top classes are long-snouted dogs (borzoi, sheepdog, collie), not glasses. |
| Other top readers | Mixed: frogs, joysticks, mergansers, ice lollies, polecats, laptops. Not a V-themed set. |
| Round things push it negative | **Strongly holds.** The bottom by mean activation is almost all round: barometers (9 of the bottom 40), manhole covers, baseballs, lens caps, compasses, clocks, croquet balls, brake discs, acorns, coiled snakes, a round-faced mastiff. Pug ranks 824th of 1000 classes. |

## Where his claim runs into trouble

- **The top class is the Persian cat** (mean activation 976, rank 1 of 1000). It is the flat, round-faced cat. A pure V-shape detector should rank it low.
- **Pointed objects outside animals and vehicles score low:** obelisk 953rd of 1000, spindle 787th, church 625th, candle 406th, volcano 316th. Bullet train (61st) is the exception. Caveat: ImageNet photos of these are often distant and small in the frame. The held-out pool's object crops address that.
- **Synthetic shapes barely register** (about 30–110, against roughly 1,000 for top classes), and there is no preference for V over O. U's score as high as V's or higher. The ring O (55) sits inside the V range.
- **The peak sits on the nose and centre of the face**, not on the ear tips.

## Correction to the earlier verification read

At the class level, **cars are a strong second cluster**, not a weak one: beach wagon 6th, convertible 14th, ambulance 15th, minivan 17th, pickup 18th, limousine 19th, sports car 23rd. Cars rarely reach the very top images, but whole car classes score high. So the original "cat + car" description holds up better than the top-20 check suggested. The positive side looks like **frontal small-carnivore faces + vehicle fronts**.

## Current picture

The two ends of the ranking behave differently:
- **Negative end:** consistent with his claim. Round, dial-like and ball-like things push it down.
- **Positive end:** not "pointed, regardless of category." It is category-bound (carnivore faces and vehicle fronts), includes round-faced Persians, and ignores pointed objects outside those categories.

## Follow-ups (`followups.txt`, `persian_top12_bottom12.jpg`, `crop_scale.csv`)

- **The most negative inputs are not round or dial detectors.** They are mostly floppy-eared dogs (clumber, basset, Saint Bernard, springer, Irish setter, Weimaraner, Great Pyrenees), flat-faced apes (chimp, gorilla), pug and schnauzer, and boxy rail vehicles (locomotive, streetcar, trolleybus). The input weights look like **contrasts within its own categories**: upright vs. floppy ears in dogs, vans vs. boxy rail fronts in vehicles. They don't look like a V-vs-O axis. Round dials ending up at the bottom are not explained by any single strongly negative input in the top 15.
- **Cropping doesn't rescue pointed objects.** Cropped tight, obelisk, bell cote, mosque, church and spindle still rank 824th–998th of 1000. Pointed *vehicle* noses do moderately well: bullet train about 55th, airliner about 75th, warplane about 86th. The small-in-frame explanation is ruled out.
- **Persian cats really are top.** They are mostly frontal close-ups, but even the lowest non-pool Persian (423) is twice the median class mean (193). Persians (median 1012) beat tabbies (664). Within Persians, the high scores are the frontal faces and the low ones are lying down or side-on. Only 20 Persians are outside the pool.
- **Pattern across the top classes:** Persian, papillon, Pomeranian, Japanese spaniel, Arctic fox, Samoyed, hamster and red panda are fluffy, upright-eared, frontal small-animal faces.

## Updated picture

The positive side is **frontal faces of small furry animals (especially fluffy, upright-eared ones) plus vehicle fronts**. Its input wiring favours upright over floppy ears within dogs, and vans over boxy rail fronts within vehicles. It is not a general V detector: pointed architecture stays at the bottom even when cropped. Round things do sit at the bottom, but that isn't wired through a specific round-detector input; it is more likely broad inhibition.

## Feature visualization (`results/featurevis/`)

Standard Lucid recipe: FFT with decorrelated colour, jitter/scale/rotate, 512 steps, whole-channel objective, 2 seeds plus 4 diverse renders.

- **Maximizing image** (about 4,400–4,950 max, roughly 3× the best natural image): tiled cat and fox faces (pointed ears, whiskers, white muzzles) mixed with glossy car-body and window shapes with chrome highlights. This is the classic "whiskers and chrome" cat + car image. All 4 diverse renders show the same mix.
- **Minimizing image** (about −490 max, −1,300 mean): round, wrinkled, droopy ape- and pug-like faces and rounded scaly blobs. This matches the strongest negative inputs (apes, pugs, floppy-eared dogs) and the essay's "a round, flat face" caption.
- **Not reproduced:** the essay's "pale spikes fanning from a point". He made those with an unpublished "generic", "path-anchored" recipe. His argument is that standard feature visualization finds the cat route and "adds everything that route brings", including the whiskers. The standard render cannot test that argument on its own.

## Route ablation (`results/ablation/`)

This tests his argument that the whiskers "come along for the ride" with the cat route. If the neuron is a V detector, removing the cat/fox inputs should leave pointed shapes. Weights into 4e:55 only were zeroed, and the rest of the network was untouched.

- **A1:** the fox input (4d:493).
- **A2:** 17 inputs selective for cats/foxes but not cars. 4d:493 is spared because it also responds to cars (car z = 1.27).
- **A3:** A1 + A2.

| | Intact | A1 (fox only) | A3 (all cat/fox) |
| --- | --- | --- | --- |
| Top class | Persian cat | convertible | convertible |
| Top 50 images | Pomeranian, kit fox, red panda, papillon… | police van, beach wagon, convertible, pickup… | police van, beach wagon, convertible, pickup… |
| Persian / tabby / red fox (rank of 1000) | 1 / 11 / 3 | 3 / 21 / 29 | 83 / 532 / 727 |
| Beach wagon / sports car / convertible | 6 / 23 / 14 | 2 / 5 / 1 | 2 / 4 / 1 |
| Airliner / warplane / bullet train | 67 / 84 / 61 | 16 / 24 / 36 | 13 / 19 / 23 |
| Obelisk / church / spindle | 953 / 625 / 787 | 960 / 772 / 687 | 872 / 567 / 535 |
| Hare / gazelle (pointed ears, not carnivores) | 122 / 119 | 436 / 208 | 935 / 536 |
| Render (feature visualization) | cat/fox faces + car bodies | car bodies, chrome, one with fur and legs | car bodies and windows; a few small radial starbursts in 2 of 6 |

**What's left after removing the cat route is a vehicle detector, not a V detector.**
- Cars take over the top.
- Pointed *vehicles* (planes, bullet train) rise.
- Pointed non-vehicles stay in the bottom half.
- Pointed-eared non-carnivores (hare, gazelle) fall. Their earlier score came through the cat/fox route.

The fox input alone accounts for most of the cat side: removing just 4d:493 turns the neuron mainly into a car neuron. The renders agree: with the cat route gone, optimization produces cars, not generic spikes.

## Simplest images (`results/simple_shapes/`)

The objective is the centre position of the 14×14 map, in grayscale, with jitter.

**TV-regularized pixels.** Raising the penalty from 0 to 100,000 fades the image rather than simplifying it. The maximizer stays a boxy vehicle front or a fox/cat face; the minimizer stays a round, wrinkled pug face. At the strongest penalty, what survives (contrast-stretched):
- maximizer: a dark nose above a pale, downward-pointing muzzle patch, with faint pointed ears;
- minimizer: a small round pale blob.

**Geometric primitives.** Each image is up to 3 soft line segments plus 2 ellipses (ring or disc), in single gray levels on gray. There were 48 restarts per direction.
- **Maximizers** (best 924, median 614; comparable to an average Persian photo at 976, about 10× hand-drawn V's): small toy-car-like objects. A pale roof or bonnet panel sits over dark rounded "wheel" or bumper capsules, often with a thin bright chrome-like line. **No restart in the top 12 forms a V or chevron**, although two line segments could make one trivially.
- **Minimizers:** shaded spheres and balls with highlights, a cup on a saucer seen from above, dark circles. Clearly round.
- Caveat: primitives can't make fur, so this search is biased toward the vehicle route over the cat-face route.

## Head-to-head: best V vs. best O vs. free shapes (`results/head_to_head/`)

Setup, the same for every family: one gray level on mid-gray, soft edges, centre-position objective with jitter, 32 restarts × 400 steps. Scores are the centre activation averaged over the same 16 jitters.

| Family | Best max | Top-5 mean | Best min | What the optimizer builds |
| --- | --- | --- | --- | --- |
| **V** (2 segments, shared vertex) | 312 | 295 | −130 | Max: white, **wide-open** bends (143–159° opening, nearly straight), vertex pointing down. Min: the arms collapse into a **round blob on a stick**. |
| **O** (1 ring or disc) | 251 | 222 | −159 | Max: flat, wide white elliptical rings. Min: **small round discs and shaded balls**. |
| **Two bars** (same budget as V, no shared vertex) | **362** | 343 | −194 | Max: **crossing** white bars (X and + shapes). |
| **Free** (3 segments + 2 ellipses) | **726** | 659 | **−430** | Max: toy-car-like composites. Min: a **ball resting on a shelf or bar**. |

For reference: average Persian photo 976, best natural image about 1,590, hand-drawn V's about 30–110.

- **A V is not the best simple shape.** Two crossing bars beat it (362 vs. 312) with the same budget. Even when it can make a sharp V, the optimizer opens it to a shallow bend. The best O reaches 80% of the best V.
- **Round blobs are the robust minimizer.** Every family that can make one does: the V turns into a ball on a stick, the O into a small disc or ball, the free search into a ball on a shelf.
- **Simple geometry drives the neuron weakly.** The best single figure is under a third of an average Persian photo. Composites that look like vehicles do about twice as well as any single figure.

## Replicated shape comparison (`results/tighten/`), with corrections

128 restarts per condition; statistic is the top-10% mean, with a 95% bootstrap CI.
- **V vs. O:** V 319 vs. O 243, difference +76 [+54, +97]. V clearly beats O. This corrects "barely beats" above.
- **V vs. two crossing bars:** difference −12 [−36, +11], a statistical tie. This corrects "loses to" above.
- **Angle sweep: inconclusive.** With the angle fixed, the optimizer still avoided making a V (shrinking an arm to a stub, or growing a blob). A forced-V rerun was stopped: picking the shape family bakes in the hypothesis.

## Simplest image at a fixed activation, no shape vocabulary (`results/minimal_image/`)

Grayscale pixels starting from gray. Complexity (TV, or TV + L1 "ink") is minimized subject to the centre activation reaching 500, 1,000 or 2,000. 3 seeds each; all 18 runs reached their target.
- **Every solution is an animal face:** a frontal fox, cat or raccoon-like face with eyes, a dark nose, a pale muzzle, upright ears and fur. This holds even at the lowest target and with the ink penalty. A few seeds add an animal body with legs, or a vehicle-like form with a face.
- **No V, chevron or other bare geometric figure appears in any run.** The penalty makes fine texture expensive, yet the fur stays. That suggests fur is part of what this neuron needs, not decoration.
- **A possible link to the essay's "pale spikes fanning from a point":** in these images, pale fur strokes radiate outward from the nose. His generic render may be this radiating fur pattern with the face detail stripped away. That is an interpretation, not tested.
- Caveat: complexity varies a lot across seeds, so these are simple solutions, not proven minima.

## Why cars activate it (`results/car_route/`)

The activation at the peak was split into per-input contributions (weight × input activation), using the top quarter of non-pool car and cat/fox images.

| Source of activation | Cars (median peak 850) | Cats/foxes (median peak 1152) |
| --- | --- | --- |
| Car-route inputs (57 inputs; car z > 1, cat/fox z < 1) | **75%** (van detectors 4d:352 and 4d:201 lead) | 6% |
| Cat/fox-route inputs (the 18 in A3) | **31%**, almost all from the fox input 4d:493 | 93% |
| Everything else | −7% | 1% |

- **Mostly a separate route.** Three quarters of a car's activation comes from vehicle detectors, not from looking like a face.
- **But there is real overlap.** The single largest contributor on car images is the *fox* detector, 4d:493 (median 269, ahead of the van detector at 220).
- **It isn't colour.** The top car images are mostly tan, orange and red, but in grayscale the fox input keeps 82% of its contribution on cars (81% on cats/foxes).
- **It isn't a car "face".** On top car images the peak sits on the bonnet just in front of the windscreen, mostly in three-quarter views, not on the headlights or grille.
- **Tested and rejected: the "projecting snout" idea** (`results/car_view/`). Setup: 120 random non-pool car images, view-labelled by Claude from the model's crop before any measurement (labels hashed). Result: 7 head-on, 67 three-quarter, 19 side, 12 rear, 15 unclear.
  - Fox-detector (4d:493) max by view: head-on 479, three-quarter 915, side 877, rear 661. For reference, real foxes average 1,795.
  - Three-quarter − head-on: +437 [+312, +564]. Three-quarter − rear: +255 [+53, +436]. Three-quarter − side: +38 [−124, +199], no difference.
  - **Head-on cars, the most face-like and most "pointing at you" view, drive the fox detector least.** Side views, which point nowhere near the camera, do as well as three-quarter.
  - On cars the fox detector peaks on the **windscreen and upper cabin**, not the bonnet tip. On foxes it peaks mid-face.
  - So what the fox detector shares between foxes and cars is something visible in the car's side and three-quarter profile around the cabin. It isn't a car face, colour, a V, or a projecting snout. It is still unidentified.

## What Phase B should test

1. **Persian vs. pointed-faced cats,** controlled for crop and pose. Is round really worse within cats?
2. **Pointed non-animal objects cropped to fill the frame.** Does the weak response survive the scale fix?
3. **The negative end: round vs. matched non-round** objects (round vs. square dials, round vs. square covers). Is it roundness, or dials and radial texture?
4. **Ear and snout edits** with category held fixed (shape change only).
5. **Within cars:** pointed vs. round fronts.
