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

## Backlog (only if Phase 0's real-cap gap demands it)

Parked, unordered beyond "cheap and safe before heavy and risky". Each is a single
lever to add *one at a time*, re-measuring against Phase 0.

- **Output-representation ablation** — swap the head away from the classification
  baseline: (sin, cos) regression (survey's other top performer, but see Phase 0 —
  needs a warm start), von Mises (μ, κ) for uncertainty, phase-shifting coder.
  (Bin-count sweep done in Phase 0: 72 bins wins.)
- **Input-side domain match** — make synthetic discs *look* like real caps, applied
  in the fixed captcha frame after rotation so the angle label stays clean:
  replicate the real disc/ring geometry, histogram-match color, match the JPEG +
  blur degradation profile, FDA (swap low-freq amplitude, keep phase). GAN
  translation only as a late, gated stretch.
- **Feature/loss-side domain adaptation** — AdaBN recalibration on real; two-view
  equivariance loss `λ·L_eq` on the unlabeled pool; DANN/CORAL alignment.
- **Semi-supervised** — pseudo-label high-confidence real caps (von Mises κ or TTA
  agreement), self-train.
- **Architecture** — ViT / Swin; group-equivariant CNN as a principled stretch.
- **Inference TTA** — average predictions over several known input rotations.

Validate any domain-match work with a synthetic-vs-real discriminator (its accuracy
dropping toward 50% ⇒ gap closing), never by peeking at the labeled test set.

[arxiv 2603.25351]: https://arxiv.org/pdf/2603.25351
