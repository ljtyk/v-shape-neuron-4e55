# Is InceptionV1's mixed4e:55 a V-shape neuron?

**Report:** https://ljtyk.github.io/v-shape-neuron-4e55/

This is a pre-registered test of the claim in [*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) that neuron mixed4e:55 is a V-shape detector rather than a cat/car neuron.

**Result.** The strong claim fails: round faces drive the neuron about 4× more than pointed objects, and covering the ears does no more than covering the torso. Round things push it down, more weakly than predicted. The frozen overall verdict is "mixed".

## Contents

- `index.html`: the report.
- `docs/`: original design, Phase A findings, and the Phase B design with every change and the freeze record.
- `scripts/`: all code (Lucent InceptionV1, analysis, stimulus building, rater verification).
- `data/phase_b/`: stimulus lists, AI-rater labels (raw and converted), rater instructions, verification logs, and the freeze manifest.
- `results/`: Phase B report, summary and raw activations, plus Phase A numeric outputs.
- `assets/`: generated images (feature visualizations, synthetic shapes). No ImageNet content.

ImageNet / ImageNet-V2 images and model weights are not redistributed. The stimulus lists name the source files.
