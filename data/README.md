# Data

## Overview

414 unique penalty-kick event clips, a 20-frame window ending at ball contact,
18 biomechanical variables per frame extracted via MediaPipe Pose. Two of them
(`lateral_trunk_lean`, `support_foot_x`) are present in the files but excluded
from all analysis, along with two derived columns that reintroduce them —
see [Columns that are not features](#columns-that-are-not-features).

**The clips are not all 25 fps, and the window is defined in frames.** Across
the 414 clips: 345 at 25 fps, 28 at 29.94, 22 at 30.03, 10 at 50 and 9 at 59.88.
The 20-frame window therefore covers 760 ms of run-up at 25 fps but only 634 ms
at 30 and 317 ms at 60. Two consequences for anyone using these files: any
"N ms before contact" figure is exact for the 83% at 25 fps and an overestimate
for the other 69 clips, and the early/mid/late phases span different real
durations from row to row. Angular velocities are unaffected — the kinematics
uses each clip's real fps for `dt`. See
[CHANGELOG K4](../CHANGELOG.md#k4).

The sample in `sample/` is restricted to 25 fps clips, so anything run against
it has a single exact time base.

The clips come from **28 source videos**, and the largest contributes 80 clips.
Clips from one video share camera pose, lighting and sometimes the same kicker,
so they are not independent — the analysis scripts cross-validate with folds
grouped by `video_name` for this reason.

## Required Files

These two files are required to run the scripts in
`03_statistical_analysis/` and `04_models/`:

### clips_master.csv

Per-clip dataset (414 rows, 265 columns) — sep=`,`, decimal=`.`.
One row per clip, aggregating the frame-by-frame measurements into
metadata (`outcome`, `foot_used`, `macro_zone`, `gk_guessed`, ...) plus
phase-level features per biomechanical variable: value at contact, and
mean/std/max/min for the early (frames 0-6), mid (frames 7-13), and late
(frames 14-19) phases, plus the late-minus-early delta. This is the file
every script in `03_statistical_analysis/` and `04_models/` reads by default.

Of the 250 numeric non-metadata columns, **217 are used as features** — see
[Columns that are not features](#columns-that-are-not-features). The statistical
scripts and the model scripts share one definition, so this is a single number;
earlier versions of the pipeline had them screening 235 and 219 respectively.

### biomechanics_dataset_complete.csv

Per-frame dataset (8,280 rows = 414 clips x 20 frames) — sep=`,`, decimal=`.`.
One row per frame per clip, with the raw joint angles, angular velocities,
trunk posture, and running speed before phase aggregation. Used for the
frame-by-frame / timeline analyses (e.g. at what frame a signal becomes
statistically significant).

## How this dataset was measured

The pipeline records no provenance columns, so the provenance of these files is
fixed and stated here instead:

| Aspect | This dataset |
|---|---|
| Joint angles | Computed from **2D image coordinates**. An angle read off an image is a projection, so it depends on how the kicker was oriented towards the camera. |
| Translation (speeds, support-foot position) | Computed **inside the tracking crop** — see the warning below. |

The current pipeline does not compute
angles from 3D world landmarks and maps translation back to original-video
coordinates, tagging rows with `angle_source` and `translation_frame`. `pose_extractor.py` reads image coordinates only, `pose_world_landmarks`
appears nowhere in the repository, no crop geometry is written, and neither
column exists. Those are open problems, not solved ones — see
[CHANGELOG K1 and K2](../CHANGELOG.md#k1).

The 3D alternative was generated outside this repository and compared head to
head, and the comparison cannot be reproduced from what is published here.
Controlling for the
presence of the translation features, the difference is **0.673 vs 0.646
ROC-AUC with overlapping intervals**. The 2D dataset was kept because it also
carries the translation features — which is a decision to keep a representation
because it carries columns this same document says are not physically
meaningful. See [`../CHANGELOG.md`](../CHANGELOG.md), K1 and K2.

**`foot_used` has unverified provenance.** A key-mapping bug meant annotated
values could fail to map, after which the pipeline fell back silently to
auto-detecting the kicking leg from ankle displacement — about 74% accurate.
Whether these published files are affected has **not** been checked against the
original annotation, and it should be before the numbers are relied on. Note
that patching the column after the fact would not be enough: the features are
computed relative to whichever leg was chosen, so `kick_knee_angle` and
`supp_knee_angle` would refer to the wrong leg in the affected clips. Only
re-running the pose extraction fixes that. See
[CHANGELOG K3](../CHANGELOG.md#k3).

**Two legacy columns.** `kicking_leg` (a duplicate of `foot_used`) and
`camera_angle` are present here but are no longer produced by the pipeline.
Nothing downstream reads them. Dropping `camera_angle` was a step backwards: it
is the largest confounder in the project — it motivates excluding two feature
families and conditions every projected angle — and it is now unrecoverable for
new clips.

**Nothing here has a measured annotation reliability.** `target_zone`, T0,
`foot_used` and `gk_guessed` come from one annotator, with no blind
re-annotation and no second rater. `gk_guessed` in particular is a subjective
judgement used both as a target and as the human baseline.

**`macro_zone` is oriented from the kicker's point of view.** `Left` means the
kicker's left as he faces the goal, which is the orientation of the annotation
grid. This matters for the Natural/Crossed label, and it was not written down
anywhere before.

## Reading the data correctly

**Translation features are not in physical units.** `running_speed_kmh`,
`kick_foot_speed_ms` and `support_foot_x/y` are displacement measurements, and
here they were computed inside the tracking crop, which follows and re-scales
the player every frame. They therefore capture motion relative to the tracker
rather than over the ground — the median late-phase run-up speed of **4.99
km/h** reflects that, not real running speed, which for a penalty approach is
two to four times higher.

They also contain outright measurement failures. Displacements are computed as
`np.diff(np.nan_to_num(...))`, so a missing keypoint becomes the coordinate
(0, 0) and produces a displacement the size of the image diagonal — and nothing
clips these series, unlike the angular ones. Across the 414 clips,
`running_speed_kmh` reaches **387 km/h** and `kick_foot_speed_ms` reaches
**133 m/s**, against medians of 2.95 km/h and 2.88 m/s; 40 and 83 frame-rows
respectively sit beyond any physically possible value. Any `max`-based aggregate
of these columns is therefore partly a detector of detection failure rather than
a measure of movement.

`support_foot_x/y` have a further problem: the reference point is the kicking
ankle at **frame 0**, roughly 800 ms before contact and mid-run-up, used as a
proxy for the ball. What they measure is the extent of the approach, not where
the support foot was planted relative to the ball. A failed measurement is
stored as an exact 0 rather than NaN — **27 of the 414 clips (6.5%)** — which
looks like a plausible value and lands at `support_foot_angle_deg` = 0°, in the
middle of the right-footed cluster.

They do carry some information, but they are not what drives the results: on the
Natural vs Crossed task the 31 translation columns on their own reach ROC-AUC
0.555 [0.324, 0.667] — a range that includes chance — against 0.697 for the full
217-feature set. An earlier version of this file put that figure at 0.689 and
concluded that most of the result rested on these columns; that came from an
earlier configuration and does not hold. Use them as features, never as
measurements.

**`macro_zone` is derived from `target_zone`** and only covers the six corner
and side zones: `Left` (zones 1/4/7, n=190) and `Right` (zones 3/6/9, n=166).
Central kicks (n=45) and off-target kicks (n=13) fall outside it, which is why
the direction models run on 356 clips rather than 414.

**`gk_guessed` is not independent of `outcome`.** All 71 saved penalties are
labelled as guessed, so 37% of that target's positive class is defined by the
result rather than by an independent judgement of whether the goalkeeper read
the movement. Treat P1 (predictability) and P3 (outcome) as overlapping
questions.

**Elbow angles are frequently missing.** `left_elbow_angle` is absent in ~35%
of frames and the missingness is not random — it depends on which side of the
body faces the camera, and therefore on the kicking foot (40% missing for
right-footed kickers, 19% for left-footed). Features derived from it are
heavily imputed.

## Columns that are not features

These columns exist in the files but are excluded from every model and
statistical test. The scripts enforce this centrally
(`04_models/eval_utils.py` and `03_statistical_analysis/stats_utils.py`), so
including them by hand will produce numbers that do not match the published
results. Together they remove 33 of the 250 numeric non-metadata columns,
leaving 217.

| Group | Count | Why excluded |
|---|---|---|
| `mean_visibility_score` and its aggregates | 15 | MediaPipe detection confidence — describes recording quality, not the kicker's body |
| `lateral_trunk_lean` and its aggregates, plus `deception_lateral` | 15 | Signed along the image x axis; the dataset mixes four camera positions, so the same value means different things in different clips |
| `support_foot_x` | 1 | Same reason |
| `support_foot_angle_deg` | 1 | `arctan2(support_foot_y, support_foot_x)` — a bijective re-encoding of `support_foot_x` in polar coordinates. The sign of x survives as "near 0°" vs "near ±180°", which makes it a near-direct proxy for the kicking foot. Excluding `support_foot_x` while keeping this excluded nothing |
| `deception_score` | 1 | Documented as the mean of `deception_torsion` and `deception_trunk`, but still averages in the lateral component. Both clean components remain available as their own columns |

`support_foot_y` is retained (it is a depth measure, not a signed lateral one),
and `support_foot_distance` is retained (a magnitude carries no sign, so the
camera-inversion argument does not apply to it) — both share the
crop-reference-frame problem described above.

## How to Obtain

These files are not included in the repository. The dataset took many hours of
manual annotation, so it is shared on request rather than published outright:

1. **Contact the author** — if you have a real use for it, an arrangement can
   be found.
2. **Generate them from scratch** by running the full pipeline (see
   [`../docs/pipeline.md`](../docs/pipeline.md)) with your own match video.
   Note that a dataset generated today will match this one in provenance: 2D
   angles, crop-relative speeds
   coordinates (or be `NaN`). The published results correspond to this dataset,
   not to a regenerated one.

The sample below exists so the scripts can be run and verified without either.

Note on privacy: these two CSVs contain no video and no player names, which is
why they can be shared when the source footage cannot. `video_name` does
identify the match.

## Sample

Two files, both covering the same 20 clips, so you can verify the scripts run
correctly before obtaining the full dataset:

- **`sample/dataset_sample.csv`** — 400 rows (20 clip_ids x 20 frames each),
  same format and columns as `biomechanics_dataset_complete.csv`.
- **`sample/clips_master_sample.csv`** — 20 rows (one per clip), same format and
  columns as `clips_master.csv`.

The 20 clips are a real subset of the dataset, not synthetic rows, chosen so
that the sample exercises the code rather than being a random draw:

| | |
|---|---|
| Frame rate | **all 25 fps** — one exact time base, so `MS_PER_FRAME` holds |
| Natural / Crossed | 8 / 8 |
| Left / Right zone | 8 / 8, plus 2 Center and 2 Out so the zone filter is exercised |
| Kicking foot | 13 right / 7 left |
| `gk_guessed` | 10 / 10 |
| Outcome | 11 goal, 5 saved, 2 out, 2 post |
| Source videos | 12, none contributing more than 5 clips, so grouped CV has real groups |

A random 20-clip draw tends to produce a degenerate sample — one class of
`gk_guessed`, one outcome, or too few videos to group by — and several scripts
then have nothing to test. This selection avoids that.

At 20 clips the sample is still far too small for the models to produce
meaningful scores. It is there to confirm that the pipeline runs end to end, not
to reproduce any result. All twelve analysis scripts run to completion on it.
