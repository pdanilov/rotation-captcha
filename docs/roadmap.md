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
| pseudo (frac 0.50), pre-clean/EMA | 0.45 | 0.28 | 13° | old base; superseded |
| **+ pseudo (frac 0.50) on clean+EMA** | **0.56** | **0.44** | **6.3°** | **current best**; EMA teacher |

Current best stack: clean orientability-filtered anchor + weight EMA + pseudo-labeling
with an EMA teacher — **real acc@10 0.56, median 6.3°** (last-15 mean 0.559±0.008, saved
epoch 59). The domain-adaptation lift over the synthetic-only clean+EMA baseline (0.42) is
real and stable. The virtuous cycle held: the confident subset's mean R climbed
0.82→0.95→0.99→1.00 across relabelings with no confirmation-bias collapse. MAE stayed ~53°
— the ambiguous-crop tail is still wrong (unfixable without dropping those caps); it's the
*orientable* caps that jumped, which is exactly what median 6.3° reflects.

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

## Where the real-test MAE comes from (diagnosis of the best model)

At acc@10 0.56 / median 6.3° the MAE is still ~53°. A four-step attribution (A: error
histogram; B: confidence×correctness fork on `R`; C: a baseline-invariant self-consistency
"thermometer" — rotate each cap by known angles, check the recovered base angle agrees; D:
eyeball the tail; contact sheet in `docs/images/mae_diagnosis_bands.png`):

- **The tail is not noise, it's 180° flips.** 55% of caps are ≤10°; the error histogram
  has a hard second peak at 160–180° (22% of caps). 33% of caps (err>45°) account for 92%
  of the total MAE.
- **Half the tail is confident, and it's flips.** Splitting err>45° by `R`: ~45 caps are
  low-R (the model *knew* it was unsure — honest ambiguity), but ~21 are high-R **and**
  wrong, and 18 of those 21 are flips — the model is *confidently 180° off*.
- **Three self-orientability classes** (from the thermometer): *orientable* (clear single
  up, 55%), *polarity-ambiguous* (axis clear, up/down splits into two modes 180° apart,
  22%), *scattered* (no recoverable up at all, 23%, median 60°).
- **The captchas are AI-generated *scenes*, not single objects.** Every tail cap carries a
  「图片由AI生成」("AI-generated") watermark and shows cityscapes, landscapes, industrial
  views, abstract textures — not the single upright object our COCO-object anchor is built
  from. So the domain gap is one of *content type*, not just style.

**Verdict:** the high MAE is mostly a property of the captcha *content*, not a model bug.
~23% (scattered) have no recoverable up even in principle; ~22% (polarity-ambiguous) are
partly underdetermined; only ~10% (≈20 orientable-but-flipped caps) are a clean, fixable
model error. Median 6.3° and 55% solved is near the achievable ceiling for this content.
**MAE is the wrong headline metric here** — it is dominated by an irreducible tail; median
and acc@10 reflect the solvable caps and should lead.

**The fix (planned, one lever): broaden the anchor with upright scene images.** The
fixable ≈20 caps and part of the polarity-ambiguous group share one disease — the model
gets the *axis* but the *polarity* (which way is up), because polarity is a *scene-level*
cue (sky up, ground down, gravity, building verticals) that a COCO-*object* crop never
teaches. The medicine for both is the same: train on upright full scenes, not just object
crops. This reverses the earlier "full scenes are a poor proxy" call — that predated
knowing the real caps *are* scenes; for AI-scene captchas, upright scenes are the *better*
proxy. **Source: Places365** (scene images with a canonical up). Must go through the same
per-crop orientability filter (scenes have their own no-up tail), then mix with the
existing object anchor (the real pool is mixed) and re-measure acc@10/median vs 0.56.
Expected yield differs by class: orientable-flipped should recover almost fully (cue is
present, model just inverts it); polarity-ambiguous only partly (some caps genuinely
underdetermine up); scattered not at all.

## Backlog (only if Phase 0's real-cap gap demands it)

Parked, unordered beyond "cheap and safe before heavy and risky". Each is a single
lever to add *one at a time*, re-measuring against the best config above.

- **→ NEXT: scene anchor (Places365)** — the one lever the MAE diagnosis points to.
  Add upright full **scene** images (Places365, canonical up) to the object-crop anchor to
  teach the *scene-level* up cues (horizon, sky, gravity, building verticals) that resolve
  the 180° polarity flips — the fixable part of the tail. Same pipeline as the object
  anchor: crop→disc, run the orientability filter (scenes have their own no-up tail), mix
  with the current clean object slice, retrain (clean+EMA+pseudo), compare acc@10/median
  vs 0.56. See "Where the real-test MAE comes from" above for why (real caps are AI scenes,
  not objects) and the expected per-class yield. This reverses the original object-crop
  decision, which predated knowing the real caps are scenes.
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
