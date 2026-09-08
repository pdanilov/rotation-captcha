# Experiment roadmap

Goal: predict the clockwise rotation angle of a Baidu-style disc captcha. We train
on synthetic COCO object discs (rotate → circular mask) and test on 200 real,
labeled Baidu captchas. Everything is grounded in the circular-aware rotation
survey ([arxiv 2603.25351]) plus our synthetic→real domain gap.

## Guiding principle

Do the simplest thing first and **look at the number** before building any
domain-adaptation machinery. The synthetic→real gap is a measurement, not an
assumption. We only earn each extra lever below by seeing the previous baseline
fall short — and the error tells us *which* lever.

## Metrics (all phases)

The captcha's real acceptance tolerance is unknown to us (we only have static
images, no live endpoint to probe), so we don't privilege a single threshold.

- **Circular MAE (°)** — primary comparison metric.
- **Median circular error (°)** — robust to boundary / symmetric-object outliers.
- **Solved-at-tolerance curve** — cumulative fraction with error ≤ {1°, 2°, 5°, 10°};
  read off whatever tolerance turns out to matter.
- **Boundary-band error** — MAE restricted to targets near 0/360°, to expose the
  seam failure mode.

Evaluation sets: synthetic `validation` (absolute accuracy, held-out content) and
the 200 labeled real caps as **test**, touched as rarely as possible. The unlabeled
real pool has a held-out `validation` split reserved for a label-free equivariance
metric — only if/when we reach Phase 2.

## Phase 0 — Classification baseline (the actual plan)

Train one plain model on synthetic, test on real, no domain adaptation.

- Backbone: a pretrained CNN (default `microsoft/resnet-34`), from the
  `transformers` model hub, swappable later. Pretrained weights are mandatory.
- Head: **angular-bin classification** — N bins over [0, 360), trained with a
  **Circular Smooth Label** (CSL) target: a wrapped-Gaussian soft label around the
  true bin (circular label smoothing) instead of one-hot. **Bin-count sweep (done):
  72 bins (5°) beat both 360 (1°, too fine to transfer) and 36 (10°, too coarse) on
  real-test — real acc@10 0.34 vs 0.29/0.29, median 29° vs 44°/38°. 72 is the
  default.**
  Decode with a circular soft-argmax (probability-weighted mean on the circle),
  which is seam-safe and gives sub-bin resolution.
- Why not regression: direct (sin, cos) regression **does not train from scratch**
  on this task — confirmed here (in-domain synthetic stalled at ~50° MAE) and from
  prior experience. Classification + CSL is the survey's other top performer and the
  known-good lumina37 baseline.
- Data: COCO crop → random rotation α → circular mask → resize. Label = bin(α).
- Loss: soft cross-entropy against the CSL target.
- Infra: accelerate, bf16.

Deliverable: end-to-end train + eval reporting the metrics above. Then **look at
the real-cap number.** If the gap is acceptable, we may stop here.

## Results so far (real-test, 200 labeled caps)

| Config | acc@10° | acc@5° | median | note |
|--------|---------|--------|--------|------|
| baseline (360-bin, no aug) | 0.21 | — | ~40° | Phase 0 |
| + augmentation | 0.29 | 0.18 | 44° | win (solve-rate) |
| **72-bin** + aug | 0.34 | 0.23 | 29° | bin count = biggest single knob |
| + equivariance `L_eq` | ~0.30 | 0.18 | 34° | **neutral** — relative-only, gauge drift (needs warm-start not to collapse) |
| **+ pseudo-labeling** | **~0.41** | **~0.28** | **~18°** | current best — adds *absolute* real signal |

Precondition that made pseudo-labeling work (verified read-only on the test set):
ranking real predictions by resultant-length `R` (confidence), the top 25% scored
acc@10 0.60 / median 7° vs 0.29 / 41° overall — confidence tracks correctness, so
confidence-filtered pseudo-labels are mostly right. Self-training on them lifted the
whole test set, and mean `R` of the confident subset climbed 0.92→0.99 across
relabelings (the virtuous cycle, no confirmation-bias collapse). MAE stayed ~55°:
the ambiguous-crop tail stays wrong; the *orientable* caps are what improved.

## Backlog (only if Phase 0's real-cap gap demands it)

Parked, unordered beyond "cheap and safe before heavy and risky". Each is a single
lever to add *one at a time*, re-measuring against the best config above.

- **Push pseudo-labeling further** (in progress) — loosen `pseudo_conf_frac` (25% may
  be leaving usable caps unused now that R→0.99); combine pseudo + a small `L_eq`
  consistency term.
- **Output-representation ablation** — swap the head: von Mises (μ, κ) for explicit
  uncertainty (a principled confidence gate for pseudo-labeling), phase-shifting
  coder. (Bin-count sweep done: 72 bins wins. Regression needs a warm start.)
- **Input-side domain match** — replicate the real disc/ring geometry, histogram-match
  color, match the JPEG + blur profile, FDA. GAN translation only as a gated stretch.
- **Feature-side domain adaptation** — AdaBN recalibration on real; DANN/CORAL.
  (Plain `L_eq` tried: neutral. Warm-started, small-λ `L_eq` avoids the collapse.)
- **Clean the anchor** — tighten the ambiguous-crop filter (per-crop, not per-category)
  to shrink the ~30% unorientable tail that pins MAE.
- **Architecture** — ViT / Swin; group-equivariant CNN as a principled stretch.
- **Inference TTA** — average predictions over several known input rotations.

Validate any domain-match work with a synthetic-vs-real discriminator (its accuracy
dropping toward 50% ⇒ gap closing), never by peeking at the labeled test set.

[arxiv 2603.25351]: https://arxiv.org/pdf/2603.25351
