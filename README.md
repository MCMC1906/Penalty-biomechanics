# Penalty Biomechanics

Does a penalty taker's body give away the kick before contact? This project
tests that question on 414 real penalties using pose estimation and machine
learning — and reports what the data actually supports, including where the
answer is no.

## Key Results

Primary metric is **ROC-AUC, whose chance level is 0.500** regardless of class
balance. Brackets are the 95% range across folds, from 5-fold cross-validation
repeated over 5 seeds. **A range that includes 0.500 is not a result.**

| Task | Best model | ROC-AUC [95% fold range] | Verdict |
|---|---|---|---|
| **Natural vs Crossed** | SVM (RBF) | **0.697 [0.604, 0.786]** | Signal, with the qualifications below |
| P2 — Direction (Left vs Right) | XGBoost | 0.547 [0.411, 0.654] | Not separable from chance **in this representation** — see below |
| P4 — Centre vs cornered | XGBoost | 0.616 [0.451, 0.747] | Untested — 45 clips in the positive class |
| P1 — Predictability by the GK | Logistic Regression | 0.554 [0.492, 0.643] | Not separable from chance |
| P3 — Outcome (goal vs not) | — | — | 10 of 217 features at p<0.05, against 10.9 expected by chance |

**The finding is Natural vs Crossed.**

    Natural : right foot -> Right   OR   left foot -> Left
    Crossed : right foot -> Left    OR   left foot -> Right

Left/Right are goal zones **as seen by the kicker facing the goal**, so a
crossed kick is one where the leg swings across the body midline — a
right-footed player striking with the inside of the foot. The label is
symmetric in the foot by construction, which is the point of it.

It holds at **0.713 [0.607, 0.815]** when folds are grouped by source video, so it is not an
artefact of which broadcast a clip came from. Grouping not degrading it is the
claim that number supports; it came out slightly *higher* than the random-split
estimate, which with 28 groups — one of them holding 80 clips — is fold noise,
not extra evidence.

Why this works when left-vs-right does not: the kinematic features are defined
relative to the kicking and support legs, so a crossed kick is the same
body-relative action whichever foot performs it. "Left vs Right" merges two
mirror-image actions under one label, the signal flips sign with the foot, and a
univariate filter (`SelectKBest` with `f_classif`) cannot see it.

It is not foot detection: a classifier given only the kicking foot reaches
**ROC-AUC 0.503**.

**It also survives a label-permutation test that respects the grouping**: with
labels shuffled *within* each source video and scored with
`StratifiedGroupKFold`, **p = 0.005** over 200 shuffles. And it holds within the
right-footed subset alone — **0.658 [0.520, 0.759]** on n=276, where the XOR
with the kicking foot is constant, so this is not the model recovering the foot
and exploiting the label's construction. The left-footed subset (n=80) gives
0.634 [0.387, 0.881], too wide to read either way.

**Three qualifications:**

- **It is not the translation features.** An earlier version of this document
  said those columns reach ROC-AUC 0.689 of the 0.697 on their own, and
  concluded that most of the result rested on the least defensible measurement
  in the project. Run against the current feature set, **the 31 translation
  columns alone reach 0.555 [0.324, 0.667]** — a range that includes chance.
  The 0.689 came from an earlier configuration and does not hold. What does
  remain true is that four of the six features surviving Bonferroni are
  `running_speed` variants with negligible effect sizes (d = 0.04–0.17), so
  that count should not be read as six independent findings.
- **Where the signal sits in time is unresolved.** Restricting the model to the
  earliest frames gives ROC-AUC **0.592 [0.505, 0.691]** on the early phase
  alone (60 features), **0.493 [0.378, 0.572]** on early+mid (120), against
  0.697 on all 217. That pattern is not monotone: adding the mid phase makes it
  worse than using the early phase alone, which no account of a real signal
  arriving progressively would predict. It does not establish anticipation, and
  it does not establish that the separation comes only from the contact frame
  either. It is most likely fold noise in restricted feature sets at n=356, and
  the question stays open.
- **This target is not privileged, it is well posed.** Given the kicking foot —
  which a goalkeeper always knows — Natural/Crossed and Left/Right are the same
  partition, so the gap between them is representation rather than signal. These
  features are defined relative to the kicking leg, so the same value points to
  Left in a right-footed kick and to Right in a left-footed one, and cancels
  under the plain direction label. An exploratory run supplying the foot and
  mirroring the features by it brings Left/Right up to the level reported here.
  That run is not part of this repository and no figure from it is quoted; it is
  noted because it explains why the body-relative label separates and the
  direction label does not.

**Accuracy, framed against the right baseline:**

| Reference | Accuracy |
|---|---|
| Always predict "Crossed" (majority class, 202 of 356) | 56.7% |
| Best model | **65.3%** (+8.5 pp) |
| Goalkeeper, in real time | 48.9% |

The model beats the goalkeeper by 16.4 pp, but a constant prediction already
beats the goalkeeper by 7.8 pp without looking at the kicker at all. What the
biomechanics contribute is **+8.5 pp over the majority class**.

**No usable anticipation.** Across 80 frame-level tests, 13 reach p < 0.05
against 4.0 expected by chance, and after Benjamini-Hochberg correction **six
survive — every one of them at frame 18 or 19**, i.e. **0–40 ms before
contact** — well inside the ~200 ms a goalkeeper needs to initiate a dive.
Useful for video analysis and scouting, not for a real-time decision.

Deep learning does not help: LSTM and CNN 1D sit at ROC-AUC 0.50–0.53 on both
targets, with every fold range spanning chance.

See [CHANGELOG.md](CHANGELOG.md) for the methodological corrections applied to
this project and the results they changed.

## Methodology

What the evaluation does, and why:

- **ROC-AUC as the primary metric.** Both P1 and P2 are close to balanced
  (46/54 and 53/47). PR-AUC's chance level is the class prevalence and it
  drifts above it under fold noise, which made chance-level models look like
  results in earlier versions of this work.
- **Repeated cross-validation** (5 folds × 5 seeds). Model-to-model gaps were
  smaller than the fold-to-fold spread under a single split.
- **Cross-validation grouped by source video** (`StratifiedGroupKFold`). The
  414 clips come from 28 videos, one contributing 80, so clips are not
  independent.
- **All preprocessing inside the pipeline.** Imputation, scaling and feature
  selection are refit within every fold.
- **Correction for multiple testing.** The statistical tests cover **217**
  features per target, so ~11 reach p < 0.05 by chance; Bonferroni and
  Benjamini-Hochberg are both reported, next to the number expected by chance.
  The statistical and model scripts now share one definition of what counts as
  a feature, so the two counts cannot diverge (they were 235 and 219).
  Correction is applied **within** each target, not across the four, which is
  worth keeping in mind for Natural vs Crossed: it is the fourth target,
  defined after three null ones, and at a study-wide 0.05/940 = 5.3 × 10⁻⁵ its
  two interpretable survivors (p = 0.00015) would not survive.
- **Rules validated out of fold.** Decision-tree rules are built on training
  folds and applied to unseen clips: the Natural vs Crossed rule scores
  **72.4% ± 18.9%** out of fold, covering 15.8% of unseen clips, against 100%
  in-sample leaf purity — the in-sample figure overstates the rule by 27.6 points.
- **Label permutation, grouped.** Labels are shuffled *within* each source
  video and scored with `StratifiedGroupKFold`. Shuffling globally builds a null
  that assumes clips are independent, which is the assumption the grouped CV
  exists to relax.
- **Controls that the earlier version did not run:** the model restricted to the
  early and mid phases (step 2b), the model within each kicking foot separately
  (step 2c), and a permutation test on the main target at all (step 2d).

## Project Structure

```
Penalty-biomechanics/
├── 01_extraction/            # kicker detection/tracking (YOLOv8 + ByteTrack) and clip extraction
│   └── clips_creator.py
├── 02_biomechanics/          # pose estimation (MediaPipe) and kinematic feature calculation
│   ├── pose_extractor.py
│   ├── kinematics.py
│   ├── signal_filter.py
│   ├── run_biomechanics.py
│   └── build_clips_master.py
├── 03_statistical_analysis/  # statistical tests and exploratory analysis
│   ├── stats_utils.py            # feature selection, the Natural/Crossed label,
│   │                             # multiple-testing correction, empty-table guard
│   ├── statistical_tests_fase1.py
│   ├── statistical_tests_fase2.py
│   ├── statistical_tests_fase3.py
│   ├── reg_cros.py
│   ├── bio_analysis.py
│   └── practical_insights.py
├── 04_models/                # ML/DL models and comparison
│   ├── eval_utils.py             # shared CV, metrics and feature exclusion
│   ├── stats_utils.py
│   ├── tabular_models.py
│   ├── deep_learning_models.py
│   ├── interpretable_model.py
│   ├── reg_cross_analysis.py
│   └── model_comparison.py
├── data/                     # dataset documentation and a 20-clip sample
│   ├── README.md                 # file formats, how to obtain the full dataset
│   └── sample/
│       ├── dataset_sample.csv        # 400 rows (20 clips x 20 frames), per-frame format
│       └── clips_master_sample.csv   # 20 rows, per-clip format -- 25 fps only, balanced
├── outputs/                  # figures and the write-up
│   ├── plots/                    # 5 figures (matplotlib, generated by the analysis scripts)
│   └── reports/
│       └── patterns_report.md    # full findings write-up
├── docs/
│   ├── pipeline.md
│   └── results_summary.md
├── CHANGELOG.md              # corrections applied, and defects still open
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.12. This installs the CPU build of PyTorch, which is
enough to run everything — `deep_learning_models.py` trains in minutes on CPU
at this dataset size. To use an NVIDIA GPU, install torch separately from the
PyTorch index afterwards, e.g. for CUDA 12.1:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline — How to Reproduce

Stages 1–3 require your own match video, **not included in the repository**
(private data, identifiable players). Stages 4–7 run from two intermediate
CSVs — see [Dataset](#dataset).

1. Install dependencies (see [Requirements](#requirements)).

2. **(Requires your own video)** Annotate, extract and crop:
   ```
   python 01_extraction/clips_creator.py all --videos . --output .
   ```
   Produces `dataset_manual.csv`, the raw clips, the cropped clips, and a
   `crop_*_boxes.csv` per clip holding the per-frame crop geometry.

3. Pose extraction and kinematics, then aggregation to per-clip features.
   `clips_dir` is a required positional argument:
   ```
   python 02_biomechanics/run_biomechanics.py clips_cropped/ --output . --meta dataset_manual.csv
   python 02_biomechanics/build_clips_master.py -d . -o .
   ```
   **Check `foot_used` in the output against your annotation file before going
   further.** If the values in `kick_foot` are not literally `Right`/`Left`,
   `FOOT_MAP` maps them to `NaN` and the pipeline falls back silently to
   auto-detecting the kicking leg, which is only ~74% accurate — and the
   Natural/Crossed label is defined by the foot. Nothing warns you. See
   [CHANGELOG K3](CHANGELOG.md#k3).

   Copy both CSVs to `03_statistical_analysis/` and `04_models/`.

4. Statistical analysis. **Run from inside the folder** — the scripts read
   `clips_master.csv` from the working directory. Order does not matter:
   ```
   cd 03_statistical_analysis
   python statistical_tests_fase1.py
   python statistical_tests_fase2.py
   python statistical_tests_fase3.py
   python reg_cros.py
   python bio_analysis.py
   python practical_insights.py
   cd ..
   ```

5. Models — `tabular_models.py` first (the deep-learning script reads its
   benchmark from `tabular_results.csv`), `model_comparison.py` last:
   ```
   cd 04_models
   python tabular_models.py
   python deep_learning_models.py
   python interpretable_model.py
   python reg_cross_analysis.py
   python model_comparison.py
   cd ..
   ```

Generated plots and result files are written to the working directory — i.e.
into the folder the script was run from — and are not version-controlled (see
`.gitignore`).

## Dataset

- 414 penalty-kick clips, a 20-frame window ending at ball contact, from 28
  source videos — the largest contributes 80 clips.
- **The clips are not all 25 fps** and the window is defined in frames, not in
  seconds: 345 of the 414 are 25 fps, and the other 69 run at 29.94, 30, 50 or
  59.88. The window therefore covers 760 ms of run-up at 25 fps but as little as
  317 ms at 60. Every "ms before contact" figure below is exact for the 83% at
  25 fps and an overestimate for the rest. See
  [CHANGELOG K4](CHANGELOG.md#k4).
- 15 biomechanical variables per frame (joint angles and angular velocities),
  aggregated to 250 numeric per-clip columns, of which **217 are used**:
  detection-quality metrics, camera-dependent lateral features, and two derived
  columns that reintroduce them are excluded (see [Limitations](#limitations)).
  The statistical and model scripts share one definition, so this is a single
  number rather than the 235/219 split earlier versions carried.
- Joint angles are computed from **2D image coordinates**. An angle read off an
  image is a projection and depends on the kicker's orientation towards the
  camera; see [Limitations](#limitations).
- Class balance: 317 goals / 71 saved / 13 out / 13 post; direction 53.4% Left;
  Natural vs Crossed **56.7% Crossed** (202 Crossed, 154 Natural of the 356
  clips with a Left/Right target zone).
- The two intermediate CSVs needed to reproduce stages 4–7 are not in the
  repository — contact the author, or regenerate them from your own footage.
  Formats are documented in [`data/README.md`](data/README.md).
- A 20-clip sample of both files is included under `data/sample/`, so the
  analysis scripts can be run end to end before obtaining the full dataset. It
  is restricted to 25 fps clips and balanced across the label, the kicking foot,
  `gk_guessed` and the outcome, over 12 source videos; all twelve scripts run to
  completion on it. See [`data/README.md`](data/README.md#sample).
- Source videos are not published: real match footage with identifiable players.

## Limitations

**Translation features are measured in a moving reference frame, and contain
measurement failures.** `running_speed_kmh`, `kick_foot_speed_ms` and
`support_foot_x/y` are displacement measurements, but the crop follows and
re-scales the player every frame, so they capture motion relative to the tracker
rather than over the ground — the median run-up speed of ~5 km/h reflects this.

They are also contaminated. Displacements are computed as
`np.diff(np.nan_to_num(...))`, so a missing keypoint becomes the coordinate
(0, 0) and yields a displacement the size of the image diagonal, and nothing
clips these series. Across the full dataset `running_speed_kmh` reaches
**387 km/h** and `kick_foot_speed_ms` reaches **133 m/s**, against medians of
2.95 km/h and 2.88 m/s. The `late_max_*` and `max_running_speed_kmh` aggregates
are therefore partly detectors of detection failure. `support_foot_x/y` additionally use the kicking ankle at frame 0 — mid
run-up — as a proxy for the ball, so they measure the extent of the approach
rather than foot placement relative to the ball.

They do not, however, carry the main result: the 31 translation columns on their
own reach ROC-AUC 0.555 [0.324, 0.667] on Natural vs Crossed, a range that
includes chance, against 0.697 for the full feature set. An earlier version of
this document put that figure at 0.689 and concluded the opposite; it came from
an earlier configuration and does not hold. Recovering true ground speed needs
the per-frame crop geometry, which the pipeline computes and then discards —
see [CHANGELOG K2](CHANGELOG.md#k2).

**Angles are 2D projections.** `pose_extractor.py` reads image coordinates only;
`pose_world_landmarks` is not used anywhere in this repository. An angle measured
off an image depends on how the kicker was oriented towards the camera, and this
dataset mixes four camera positions of which more than half are side views. The
camera problem is handled here only for the *signed* lateral features; it
applies to every angle.

A 2D-versus-3D comparison was run outside this repository (0.673 vs 0.646
ROC-AUC, overlapping intervals). It cannot be reproduced from what is published
here, so treat it as a note about work done rather than as evidence — earlier
versions of this document presented it as settled, and stated that the pipeline
already used world landmarks. It does not.

**`torso_torsion_angle` does not measure torsion.** It is the unsigned angle
between the shoulder line and the hip line in the image plane; axial rotation
appears in a projection as a shortening of the shoulder line, not as an angle
between the two lines, and the value jumps towards 180° when the projected
shoulder vector inverts. In practice it tracks orientation relative to the
camera.

**The signal filter does not filter.** `SignalFilter(window=3, poly=2)` fits a
parabola through three points, which reproduces them exactly — the transform is
the identity.

**Signed lateral features are excluded.** `lateral_trunk_lean` and
`support_foot_x` are measured along the image x axis, and the dataset mixes
four camera positions. Between front and back views the sign inverts; side
views encode the run-up direction instead of the goal's lateral axis, so no
single transformation reconciles them. Side views are more than half the
dataset.

**`gk_guessed` is partly determined by the outcome.** All 71 saved penalties
are labelled as guessed, so 37% of that target's positive class is defined by
the result rather than by an independent reading of the movement. P1 and P3 are
therefore not independent questions.

**No player identity is recorded.** The same penalty taker may appear in both
training and test folds. This is the most important missing metadata field, and
it bears directly on the main result: whether a kick is natural or crossed is
substantially a personal habit, so a model that recognises a kicker can memorise
his preferred technique. Grouping folds by source video handles camera pose and
lighting but not this, because the same takers appear across compilations.

**The kicking-foot provenance of the published dataset is not established.** A
bug meant the annotation could fail to map and the pipeline would fall back
silently to auto-detecting the kicking leg, which is ~74% accurate. Whether the
published CSVs are affected has not been checked against the original
annotation — and it could not be repaired after the fact even if it were, since
the features themselves are computed relative to whichever leg was chosen. See
[CHANGELOG K3](CHANGELOG.md#k3).

**No annotation reliability was measured.** T0, the goal zone, the kicking foot
and `gk_guessed` come from one annotator, with no blind re-annotation and no
second rater. `gk_guessed` is simultaneously a target (P1) and the human
baseline the model is compared against.

**Statistical power.** 414 clips against 217 features tested per target; the
fold ranges are wide enough that an effect the size of the Natural vs Crossed
one would be hard to detect on P1 or P2 at this sample size.

**Timing.** Every FDR-surviving frame-level signal arrives 0–40 ms before
contact — one or two frames, at 25 fps. Useful for analysis and scouting; not
for a real-time dive decision. And see the qualification above: a signal that
appears only at the contact frame may be the strike itself rather than anything
that precedes it.

**Rules are descriptive.** Decision-tree rules are reported with out-of-fold
precision (~61%), not in-sample leaf purity (~94%), which overstates them by
roughly 30 percentage points.

## Future Work

In rough order of how much they would change what can be claimed:

- **Regenerate the crop geometry** so translation features can be measured in
  original-video coordinates, and clip them to physical limits. They carry most
  of the main result and are the least defensible measurement in the project.
- **Record player identity** so folds can be grouped by kicker.
- **Measure annotation reliability**: re-annotate 40 clips blind and report
  agreement on T0, zone, foot and `gk_guessed`.
- **More clips (~800+)**: the fold ranges are wide enough that a real effect
  could go undetected at n=414. A learning curve on the existing data would show
  whether that is worth the effort. It would also take the 45 central kicks to
  roughly 90, which is what P4 needs to be testable at all.
- **External validation** of the Natural vs Crossed result on other competitions
  — it is the only finding worth trying to replicate.

## Citation

If you use this work, please cite:

```
Martim Cardoso (2026). Penalty Biomechanics.
GitHub: [repository URL after publication]
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
