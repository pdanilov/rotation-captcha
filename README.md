# rotation-captcha

Estimate the clockwise rotation angle of a Baidu-style **rotation captcha** (a 152×152
disc showing one rotated image) — trained on synthetic data, evaluated on 200 real,
hand-labeled captchas, **without ever training on real labels**.

Best result so far: **acc@10° = 0.76, median error 2.4°, MAE ≈ 20°** on a held-out 150-cap
test split — competitive with a published solver that additionally fine-tunes on the real
labels. The full experiment log, dead ends, and the reasoning behind each decision live in
[docs/roadmap.md](docs/roadmap.md).

## Approach

- **Task as classification, not regression.** The angle is predicted over 72 bins (5°) with
  a Circular Smooth Label soft target and decoded with a seam-safe circular soft-argmax
  (sub-bin precision). Classification is robust to the noisy targets of ambiguous crops and
  yields a confidence signal (resultant length `R`) that pseudo-labeling relies on.
- **Synthetic data = rotated discs.** A source image is center-squared, resized, rotated by
  a known angle, and masked to the inscribed circle — the same geometry as the real captcha.
- **Clean the anchor.** COCO object crops contain a large rotationally-*unorientable* tail
  (symmetric objects, textures) that injects wrong targets. A disposable filter model scores
  each crop's orientability (median error over 12 rotations); crops above the threshold are
  dropped.
- **Scenes, not just objects.** The real captchas are AI-generated *scenes*, so diverse
  upright Places365 scenes (365 classes) are mixed into the anchor — this taught the
  scene-level "up" cues that fixed most of the 180° polarity flips and roughly halved MAE.
- **Domain adaptation via pseudo-labeling.** An EMA teacher (Mean Teacher) labels unlabeled
  real captchas; the most confident fraction (ranked by `R`) is self-trained on, gated by a
  verified precondition that confidence tracks correctness.
- **Honest evaluation.** Weight EMA stabilizes the noisy real metric; the 200 labeled caps
  are split into `val` (50, early-stop / hyperparameter selection) and `test` (150, reported,
  never used for selection).

## Pipeline

```
crop_coco_objects.py / ingest_places_scenes.py   # source images -> rotatable crop slice
        │
score_orientability.py  ->  filter_orientable.py # score & drop unorientable crops
        │
merge_slices.py                                  # union COCO + Places into one anchor
        │
build_hf_dataset.py                              # -> HF imagefolder {train, validation}
        │
rotcaptcha <preset> --coco-slice <slice> ...     # train + eval (src/rotcaptcha/cli.py)
        │
diagnose_tail.py                                 # attribute a model's residual error
```

Each crop slice is content-addressed: its directory name carries a hash of the generation
params, so identical params reproduce the same slice and provenance is self-describing (see
`docs/roadmap.md`, "best practices" for the naming/reproducibility discipline).

## Usage

Environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync

# build a training slice (example: COCO objects, cleaned)
uv run python scripts/crop_coco_objects.py --split train --sample-size 10000
uv run python scripts/score_orientability.py --crops <slice> --checkpoint <filter.pt>
uv run python scripts/filter_orientable.py --crops <slice> --max-median 10
uv run python scripts/build_hf_dataset.py --crops <slice>

# train (named presets in src/rotcaptcha/config.py: default, pseudo, eqv, smoke, ...)
uv run rotcaptcha pseudo --coco-slice <slice> --epochs 100 --patience 10

uv run pytest        # tests
```

Runs log to a local [trackio](https://github.com/gradio-app/trackio) dashboard
(`trackio show --project rotation-captcha`).

## Layout

| path | what |
|------|------|
| `src/rotcaptcha/` | the trainable package: `cli` (train/eval loop), `config` (presets), `heads` (angle heads), `data`, `model`, `geometry`, `augment`, `metrics` |
| `scripts/` | data-prep tools (crop, ingest, score, filter, merge, build, diagnose) |
| `data/` | `raw/` source crops + `hf/` HF datasets + the real captchas (git-ignored) |
| `docs/roadmap.md` | the full experiment narrative and rationale |
| `tests/` | unit tests |

## Status

Working, at a competitive quality level. The next planned experiment (queued in the roadmap)
is an ablation: whole COCO images with no crops/filter/scenes, to measure how much of the
above machinery was actually necessary in the label-free regime.
