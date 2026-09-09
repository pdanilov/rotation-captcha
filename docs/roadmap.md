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

Two families below: **synthetic-only** (no real data in training) and **+ domain
adaptation** (pseudo-labeling on unlabeled real). Compare within a family.

| Config | acc@10° | acc@5° | median | note |
|--------|---------|--------|--------|------|
| _synthetic-only_ | | | | |
| baseline (360-bin, no aug) | 0.21 | — | ~40° | Phase 0 |
| + augmentation | 0.29 | 0.18 | 44° | win (solve-rate) |
| **72-bin** + aug | 0.34 | 0.23 | 29° | bin count = biggest single knob |
| + clean anchor (orient≤10) | 0.33 | — | ~30° | noisy est. (std 0.034); see EMA below |
| **+ EMA (clean anchor)** | **0.42** | **0.28** | **16°** | new syn-only baseline; stable (std 0.009) |
| _+ domain adaptation_ | | | | |
| + equivariance `L_eq` | 0.30 | 0.18 | 34° | **closed** — see below |
| + pseudo-labeling (frac 0.25) | 0.40 | 0.26 | 19° | adds *absolute* real signal |
| **+ pseudo-labeling (frac 0.50)** | **0.45** | **0.28** | **13°** | pre-clean-anchor/EMA; re-run pending |

Figures are last-15-epoch averages, n=200. **Measurement fix (important):** real-test
swings ±0.05 acc@10 epoch-to-epoch and is *decoupled* from syn-median (our early-stop
monitor), so a single syn-selected epoch lands on a near-random real point — the old
pre-EMA runs' saved checkpoints were unreliable (one landed at 0.28 while its own
last-15 mean was 0.33). **Weight EMA** (`ema_decay=0.999`, torch `AveragedModel`) cut
that variance ~4× (std 0.034→0.009) and raised the mean (0.33→0.42): eval and the saved
checkpoint now read averaged weights, so the deployed model sits near the running mean.
EMA is on by default for all runs; the pseudo-labeling rows predate it and need a re-run
on the clean+EMA base before their 0.45 is comparable.

`pseudo_conf_frac` swept: 0.50 beat
0.25 and 0.75 (0.75's extra noise bloated MAE to 60) — sweet spot in the middle,
now the default. **Equivariance is closed (re-tested on the clean+EMA anchor, and the
loss itself verified bug-free).** On the dirty anchor: λ=1 collapsed real-test (gauge
drift, acc@10→0.07); warm-started small-λ neutral; pseudo+small-λ *hurt* (0.40→0.38).
**Re-swept on the clean+EMA anchor** (the fair retest, since all the above predate it):
same failure, the clean anchor did *not* rescue it — with a 20-epoch warmup, λ=0.05
slowly drifts real down (0.45→0.40, under the λ=0 baseline 0.42) and λ=0.2 collapses it
the instant L_eq activates (0.45→0.31, median→74°), while satisfying eq *worse*
(eq_median 7→14). **Verified it's not an implementation bug:** rotate_vec maps vec(a)→
vec(a+δ) exactly, and end-to-end on orientable crops decode(v2)−decode(v1) matches a
known δ=70° to 0–5° (right sign, right magnitude). The deeper reason it can't help: the
trained model is *already* equivariant to ~2° on the crops that are orientable, so L_eq
has no signal to add there; on ambiguous crops the resultant is noise, so L_eq only
injects noise (and drifts the absolute frame when weighted up). Relative-only consistency
is not the lever here.

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

- ~~**Clean the anchor**~~ — **done.** Per-crop orientability filter (not per-category):
  a disposable filter model trained on a disjoint COCO slice scores each crop by its
  median circular error over 12 rotations (`score_orientability.py`); crops with
  `orient_median > 10°` are dropped (`filter_orientable.py` → a new raw slice). Threshold
  read off the per-band contact sheet (`docs/images/orientability_bands.png`): the tail
  is symmetric objects (orange, umbrella, donut) and textures. Dropped ~20% of crops;
  syn-only baseline 0.34→0.42 acc@10 (with EMA). Model-free self-similarity was tried
  first and **failed** (Spearman≈0 vs model error — it measures symmetry, not semantic
  orientability). Still open: whether to re-run the crop set with the category blocklist
  removed, and whether to also filter before pseudo-labeling.
- **Output-representation ablation** — von Mises (μ, κ) for explicit uncertainty (a
  principled confidence gate for pseudo-labeling, vs the current R). Phase-shifting
  coder. (Bin-count sweep done: 72 wins. Regression needs a warm start.)
- **Input-side domain match** — replicate the real disc/ring geometry, histogram-match
  color, match the JPEG + blur profile, FDA. GAN translation only as a gated stretch.
- **Architecture** — ViT / Swin; group-equivariant CNN as a principled stretch.
- **Inference TTA** — average predictions over several known input rotations.
- ~~Equivariance `L_eq`~~ — **closed** (see results table); relative-only, doesn't help.

Validate any domain-match work with a synthetic-vs-real discriminator (its accuracy
dropping toward 50% ⇒ gap closing), never by peeking at the labeled test set.

[arxiv 2603.25351]: https://arxiv.org/pdf/2603.25351
