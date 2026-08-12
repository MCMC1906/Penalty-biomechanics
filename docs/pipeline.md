# Pipeline

End-to-end flow of the project, from raw match video to model comparison. Each stage reads the previous stage's output — there is no orchestration script that runs everything; each command below is run manually, in order.

## Overview

```
                      PRIVATE (your own video required)                    REPRODUCIBLE (data included / obtainable)
        ┌──────────────────────────────────────────────────┐   ┌───────────────────────────────────────────────┐

  [match video]
        │
        │  01_extraction/clips_creator.py  (annotate → extract → crop)
        ▼
  dataset_manual.csv  +  clips_cropped/crop_*.mp4
        │
        │  02_biomechanics/run_biomechanics.py
        ▼
  biomechanics_dataset_complete.csv  +  proximal_distal_sequence.csv
        │
        │  02_biomechanics/build_clips_master.py
        ▼
  clips_master.csv
        │
        │  (manual copy: clips_master.csv + biomechanics_dataset_complete.csv
        │   → 03_statistical_analysis/ and 04_models/)
        ▼
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                               ▼
  03_statistical_analysis/*.py    04_models/*.py
  (6 independent scripts)         (5 scripts, model_comparison.py runs last)
```

## Stage by stage

### 1. Annotate — `01_extraction/clips_creator.py annotate`

**Input:** raw match video(s) (any of `.mp4 .avi .mov .mkv .mts .m4v`)
**Output:** `dataset_manual.csv` (`event_id, video_name, t0_frame, kick_foot, goal_zone, gk_guessed, result`)


```bash
python 01_extraction/clips_creator.py annotate --videos .
```

Interactive OpenCV tool: scrub to the ball-contact frame (T0), mark the kicking foot, click the goal zone, mark whether the GK guessed right, and the outcome (Goal/Saved/Post/Out).

**The values written here must be recognised by the mappings in stage 4.** The
tool writes `kick_foot` in Portuguese (`Direito`/`Esquerdo`); `run_biomechanics.py`
translates it via `FOOT_MAP`. If a spelling is missing from that map the value
becomes `NaN` and the pipeline silently falls back to auto-detecting the kicking
leg, which is only ~74% accurate — enough to corrupt a quarter of the
Natural/Crossed labels without any error being raised. Stage 4 now aborts if
more than 5% of values are unrecognised, and prints the mapped count. Check it.

### 2. Extract — `01_extraction/clips_creator.py extract`

**Input:** `dataset_manual.csv` + raw videos
**Output:** `clips_raw/*.mp4` — one 20-frame clip per event, ending at T0

```bash
python 01_extraction/clips_creator.py extract --csv dataset_manual.csv --videos . --clips clips_raw/
```

### 3. Crop — `01_extraction/clips_creator.py crop`

**Input:** `clips_raw/*.mp4`
**Output:** `clips_cropped/crop_*.mp4` — kicker isolated via YOLOv8 + ByteTrack, letterboxed to 512x1024
**Not written, and this is a defect:** the per-frame crop box, scale and
letterbox padding. `process()` computes the smoothed boxes and discards them;
there is no `crop_*_boxes.csv` and no file-writing call anywhere in
`01_extraction/`. An earlier version of this document described the sidecar as
though it existed. See [CHANGELOG K2](../CHANGELOG.md#k2).

That sidecar file would not be optional bookkeeping. The crop follows the player
frame by frame and re-scales him to a fixed canvas, so inside the cropped clip
a player running at 20 km/h barely moves. Any displacement-based feature
computed there measures the tracker's lag, not motion over the ground. The box
geometry makes the transform invertible, so stage 4 can measure speeds in
original-video coordinates.

```bash
python 01_extraction/clips_creator.py crop --clips clips_raw/ --output clips_cropped/
```

Steps 1-3 can also be run in one shot:

```bash
python 01_extraction/clips_creator.py all --videos . --output .
```

### 4. Pose extraction + kinematics — `02_biomechanics/run_biomechanics.py`

**Input:** `clips_cropped/*.mp4` + `dataset_manual.csv`
**Output:**
- `biomechanics_dataset_complete.csv` — per frame, feeds stages 5-7
- `proximal_distal_sequence.csv` — per clip, merged in at stage 5
- `summary_stats.csv` — descriptive table only, not read by anything downstream

```bash
python 02_biomechanics/run_biomechanics.py clips_cropped/ --output . --meta dataset_manual.csv
```

MediaPipe Pose (33 keypoints) → signal filtering (`signal_filter.py`) → joint
angles, angular velocities, trunk posture, running speed, proximal-distal
sequence (`kinematics.py`).

Two coordinate systems are used deliberately:

- **Angles** are computed from **2D image coordinates**. An angle read off a 2D
  image is a projection, so it depends on where the broadcast camera was — the
  same kick filmed from two angles gives two different "knee angles", and this
  dataset mixes four camera positions. MediaPipe's 3D world landmarks would
  remove most of that; they are not used. See
  [CHANGELOG K1](../CHANGELOG.md#k1).
- **Displacement** (running speed, foot speed, support-foot position) is
  computed **inside the crop**, which follows and re-scales the player, so these
  are not ground measurements. Worse, a missing keypoint becomes the coordinate
  (0, 0) before differencing, which yields a plausible-looking wrong number
  rather than a NaN — 387 km/h and 133 m/s appear in the dataset.

No provenance columns are written: `angle_source` and `translation_frame` do
not exist, though earlier versions of this document described both.

**Check `foot_used` in the output against your annotation file.** There is no
`Foot annotation: N/N rows mapped` line and no error when the mapping fails —
the pipeline falls back silently to auto-detecting the kicking leg, which is
~74% accurate, and the Natural/Crossed label is defined by the foot. See
[CHANGELOG K3](../CHANGELOG.md#k3).

### 5. Aggregate to per-clip features — `02_biomechanics/build_clips_master.py`

**Input:** `biomechanics_dataset_complete.csv` + `proximal_distal_sequence.csv`
**Output:** `clips_master.csv` (sep=`,`, decimal=`.` — the file used by every script in steps 6-7)

```bash
python 02_biomechanics/build_clips_master.py -d . -o .
```

**Manual step:** copy `clips_master.csv` and `biomechanics_dataset_complete.csv` into `03_statistical_analysis/` and `04_models/` before continuing.

### 6. Statistical analysis — `03_statistical_analysis/`

Six independent scripts, each reads `clips_master.csv` and/or `biomechanics_dataset_complete.csv` and writes its CSV/TXT/PNG outputs **into the working directory** — so run them from inside the folder, and expect the results to land there. Order doesn't matter between them.

All of them share `stats_utils.py`, which applies **Bonferroni and
Benjamini-Hochberg FDR** and prints the number of hits expected by chance next
to the observed count. With 217 features tested per target, ~11 reach p < 0.05
by chance alone, so the uncorrected count is not interpretable on its own.

`stats_utils.py` also holds the definition of what counts as a feature and of
the Natural/Crossed label. Each script used to carry its own copy of the
exclusion list and they drifted; `04_models/` keeps a byte-identical copy of the
module so each stage runs standalone.
Each script used to carry its own copy, and they drifted — when the signed
lateral features were excluded, only one script was updated and the others kept
testing them.

```bash
python 03_statistical_analysis/statistical_tests_fase1.py
python 03_statistical_analysis/statistical_tests_fase2.py
python 03_statistical_analysis/statistical_tests_fase3.py
python 03_statistical_analysis/reg_cros.py
python 03_statistical_analysis/bio_analysis.py
python 03_statistical_analysis/practical_insights.py
```

### 7. Models — `04_models/`

Four independent scripts feed the fifth. `model_comparison.py` reads
`tabular_results.csv`, `dl_results.csv`, and `interpretable_results.csv`, so it
must run last. `deep_learning_models.py` also reads `tabular_results.csv` for
its benchmark line, so run `tabular_models.py` first.

All of them share `eval_utils.py`:

- **ROC-AUC is the primary metric** (chance = 0.500). Both targets are near
  balanced, and PR-AUC drifts above the class prevalence under fold noise,
  which made chance-level models look like results.
- 5-fold CV **repeated over 5 seeds**, plus a second estimate with
  `StratifiedGroupKFold` on `video_name` so clips from one source video never
  straddle the split.
- Results carry the **95% range across folds**. A range that includes 0.500 is
  not a result.
- Non-feature columns are excluded centrally: detection-quality metrics
  (`mean_visibility_score` and its aggregates) and the signed lateral features
  (`lateral_trunk_lean`, `support_foot_x`, `deception_lateral`), which are not
  comparable across the camera positions in this dataset, plus two derived
  columns that reintroduce them. 217 of 250 numeric columns are used.

```bash
python 04_models/tabular_models.py
python 04_models/deep_learning_models.py
python 04_models/interpretable_model.py
python 04_models/reg_cross_analysis.py
python 04_models/model_comparison.py   # last -- aggregates the previous four
```

## Private vs. reproducible

| | Private | Reproducible |
|---|---|---|
| **Stages 1-3** (`01_extraction/`) | Raw match video — real footage, not published | — |
| **Stage 4** (`02_biomechanics/run_biomechanics.py`) | Depends on `clips_cropped/`, which depends on the private video | — |
| **Stage 5** (`02_biomechanics/build_clips_master.py`) | Depends on stage 4's output | — |
| **Stages 6-7** (`03_statistical_analysis/`, `04_models/`) | — | Only need `clips_master.csv` + `biomechanics_dataset_complete.csv` — not bundled in the repo, but obtainable from the author (see [Dataset](../README.md#dataset)); once you have them, these two stages run with no video and no GPU at all — `deep_learning_models.py` trains in minutes on CPU at this dataset size |

In short: stages 1-5 cannot be reproduced without the original video (never
published, real match footage of identifiable players). Stages 6-7 are fully
reproducible by anyone who obtains the two intermediate CSVs.

That split is about reproducibility, not about where the work is. Measurement
quality is decided in stages 3-4 — whether angles come from 3D world landmarks,
whether displacement is measured in a fixed reference frame — and no amount of
modelling in stages 6-7 recovers what is lost there. In this project both of
those decisions went the wrong way and have not been revisited, which is the
single biggest limitation on what any of the results can mean.
