# Changelog — methodology corrections

Every change below was made because a specific claim did not survive checking.
Before/after figures come from re-running the analysis on the same 414 clips.

**Scope note.** Everything in "Corrections applied" changed code in
`03_statistical_analysis/` and `04_models/`, and stages 6–7 were re-run from
scratch afterwards.

Everything in **[Known and not fixed](#known-and-not-fixed)** is a defect that
is still present. Those entries used to sit in this list, written in the past
tense, describing extraction code that was never written — the repository
claimed to compute joint angles from 3D world landmarks, to write a
`crop_*_boxes.csv` sidecar, and to abort when the kicking-foot annotation
failed to map, and it does none of the three. That was the same failure this
file exists to prevent: believing a claim because it sounds reasonable instead
of checking it. They have been moved and rewritten as open problems.

---

## Corrections applied to the published results

**1. Early stopping on the test fold (`04_models/deep_learning_models.py`)**
The training loop scored the *test* fold every epoch, kept the best-scoring
weights, and reported that same fold — the maximum of up to 120 noisy
evaluations on ~80 samples. Fixed with an inner validation split (20% of the
training fold), and the test fold is now evaluated exactly once.
Before: "LSTM reaches PR-AUC 0.579 on P1 and clearly beats every tabular
model." After: LSTM and CNN 1D sit at ROC-AUC 0.50–0.53 on **both** targets,
with every fold range spanning chance. The claim is withdrawn.

**2. PR-AUC as the headline metric (all model scripts)**
Both targets are close to balanced (53/47 and 46/54). PR-AUC's chance level is
the class prevalence and it drifts above it under fold noise, so the
`beats_baseline` flag was marking models with ROC-AUC 0.52 as successes.
ROC-AUC is now primary (chance = 0.500) and every result carries the 95% range
across folds.

**3. A single CV split (all model scripts)**
One `random_state=42` decided the entire model ranking, with model-to-model
gaps smaller than the fold-to-fold spread. Now 5 folds × 5 seeds, plus a
second estimate with `StratifiedGroupKFold` on `video_name` (28 source videos,
one contributing 80 clips). Before: P2 SVM 0.549 from one split.
After: XGBoost 0.547 **[0.411, 0.654]** — the range spans chance.

**4. No multiple-testing correction (`stats_utils.py`)**
217 features are tested per target, so 10.9 reach p < 0.05 by chance alone; the
frame-by-frame timelines ran a further 100 tests uncorrected. Bonferroni and
Benjamini-Hochberg are now applied across each family, printed next to the
number expected by chance. Before: "knee angle reliable from ~480 ms before
contact, in theory enough lead time to react." After: of 80 timeline tests, 13
reach p < 0.05 against 4.0 expected by chance and **6 survive FDR, all of them
at frame 18–19 (0–40 ms before contact)**. The anticipation claim is withdrawn.

**5. In-sample leaf purity reported as rule confidence**
"94% confidence, 28 historical cases" was the purity of a leaf in a tree fit on
all the data. Rules are now built on training folds and applied to clips the
tree has never seen. Before: 100% in-sample. After: **72.4% ± 18.9%**, covering
15.8% of unseen clips — the in-sample figure overstates the rule by 27.6 points.

**6. Class-weighted purity read as a sample fraction**
`tree_.value` stores class-*weighted* proportions when the tree is fit with
`class_weight='balanced'`, so purity read off it is not the real fraction of
clips in the leaf. It is now recomputed from the samples that actually land
there, via `tree.apply()`.

**7. Missing majority-class baseline (`reg_cross_analysis.py`)**
The model's accuracy was compared to the goalkeeper's 48.9% for "+13.8 pp",
without noting that always predicting the majority class already scores 56.7% —
beating the goalkeeper by 7.8 pp without looking at the kicker. Reports now lead
with the majority-class baseline: at 65.3% accuracy the biomechanics are worth
**+8.5 pp**.

**8. Feature-selection leakage (`interpretable_model.py`, `statistical_tests_fase2.py`, `practical_insights.py`)**
`KNNImputer` and `SelectKBest` were fit on the full dataset before the CV loop,
so the reported scores were optimistic. All preprocessing now lives inside the
pipeline and is refit per fold.

This entry was previously only half true: `statistical_tests_fase2.py` still
imputed globally, outside its CV loop, for as long as the entry claimed
otherwise. It no longer does.

---

## Bugs

**12. The kicking foot was never read from the annotation (`run_biomechanics.py`)**
`FOOT_MAP` only knew `Right`/`Left`; the annotation tool writes `Direito`/
`Esquerdo`. All 725 rows mapped to `NaN`, so the pipeline fell back to
auto-detecting the kicking leg for every clip — which is 74.2% accurate. Since
the Natural/Crossed label is defined by the foot, **26% of that label was
wrong**, and nothing in the output said so.

Found by comparing `foot_used` in the output against the original annotation:
316/98 expected, 249/165 produced.

**This is not fixed.** `FOOT_MAP` still accepts only `Right`/`Left`, there is
no normalisation of keys, no `Foot annotation: N/N rows mapped` line, and no
error when values fail to map — see [K3](#k3). An earlier version of this entry
said all four had been done. Silent degradation was the real defect, and it is
still possible.

**13. `reg_cross_analysis.py` could not run at all.** It read `clips_master.csv`
with `sep=";" decimal=","` while the pipeline writes comma-separated —
immediate `KeyError: 'macro_zone'`.

**14. Feature exclusions applied in only one script of six.** Each script in
`03_statistical_analysis/` carried its own copy of the exclusion list plus an
ad-hoc `'visibility' not in c` filter, and they drifted: when the signed
lateral features were excluded, five scripts kept testing them.

Now defined once per stage folder — in `03_statistical_analysis/stats_utils.py`
and `04_models/eval_utils.py` — and imported. The two folders keep separate
copies of `stats_utils.py` so that each stage runs standalone without a package
install.

**15. Signed lateral features excluded (`eval_utils.py`, `stats_utils.py`)**
`lateral_trunk_lean` and `support_foot_x` are measured along the image x axis,
and the dataset mixes four camera positions. Between front and back views the
sign simply inverts, which mirroring would fix; but side views encode the
run-up direction instead of the goal's lateral axis — a different quantity, not
a mirrored one — and they are more than half the dataset.

Both are excluded from every model and test, along with `deception_lateral`,
which derives from `lateral_trunk_lean`.

Two derived columns defeated that exclusion, and this entry previously did not
mention either:

- `support_foot_angle_deg` = `degrees(arctan2(support_foot_y, support_foot_x))`
  is `support_foot_x` in polar coordinates. The sign of x survives intact as
  "near 0°" versus "near ±180°", which makes it a near-direct proxy for the
  kicking foot. Excluding `support_foot_x` while keeping this excluded nothing.
- `deception_score` was described here as the mean of `deception_torsion` and
  `deception_trunk`. It is not: `build_clips_master.py` still averages in the
  lateral component. Rather than change the extraction code and invalidate the
  dataset, the composite is excluded downstream — both clean components remain
  available as their own columns, so nothing is lost.

`support_foot_distance` = `sqrt(x² + y²)` is deliberately **kept**: it is a
magnitude, carries no sign, and the camera-inversion argument does not apply
to it.

With these two additions the model and statistical scripts screen the same
**217** of 250 numeric columns. The columns are still computed and written per
frame in case the camera geometry is ever resolved, but nothing downstream
reads them.

**16. Detection quality used as a feature (`eval_utils.py`)**
`mean_visibility_score` and its 14 aggregates are MediaPipe confidence scores.
A model using them partly learns which broadcast a clip came from. Excluded.

With 14, 15 and 16 together, **217 of 250 numeric columns are used**, by the
statistical scripts and the model scripts alike. Previously the model scripts
screened 219 and the statistical scripts 235, and this entry claimed that
re-running the statistical scripts would apply the tighter list — it would not
have, because none of the six carried the lateral exclusions. Now they share
one definition, so the two counts cannot diverge again.

**17. A hardcoded input width (`deep_learning_models.py`)**
`N_FEATURES = 16` stayed fixed when `BIO_COLS` dropped to 15, and the LSTM
raised a shape error on the first batch. Now derived from the data at load
time, so the two cannot drift apart again. The stale tabular benchmarks
(`rf_benchmark = 0.657` when the Random Forest scored 0.592) are likewise read
from `tabular_results.csv` instead of being written by hand.

**18. NaN read as true (`model_comparison.py`)**
A missing `roc_beats_chance` flag came through as `NaN`, which is truthy in
Python, so chance-level models were printed as results. The flag is now derived
from the fold range when absent.

**19. `pip install -r requirements.txt` failed.** `torch==2.5.1+cu121` is a
local version tag that does not exist on PyPI. Pinned to `torch==2.5.1`, with
the CUDA index documented separately.

**20. Phase comparison declared winners inside the noise (`statistical_tests_fase2.py`)**
Phases were ranked by PR-AUC — with the chance line drawn on the plot at 0.500,
which is the chance level for ROC-AUC, not for PR-AUC — using a fixed 0.02
threshold to call two phases "equivalent", on a single 5-fold split, with the
imputer fit outside the CV loop.

This entry described the fix long before the fix existed; the script still did
all four. It now uses ROC-AUC with a 0.500 chance line, 5 folds × 5 seeds, all
preprocessing inside the pipeline, and the observed fold-to-fold spread as the
threshold for calling a difference real. When no phase clears chance it says so
instead of naming a best one. It also now covers Natural vs Crossed, which is
the only target where "when does the signal appear?" has something to locate.

**21. ~~Two reported figures were transposed~~ — this correction was itself wrong**
The goalkeeper's accuracy by kick type was originally given as 58.9% on Crossed
and 35.7% on Natural. A previous version of this entry called that a
transposition and swapped the two, concluding that goalkeepers read *natural*
kicks better.

The swap was wrong, and it was caused by defect 22 below: the number came from
`practical_insights.py`, which used the opposite class convention from
`reg_cross_analysis.py`, so it was compared against the wrong label.

Arithmetic settles it. Under the project's convention the groups are 154
Natural and 202 Crossed, and the goalkeeper's overall accuracy on these 356
kicks is 48.9%. Only one assignment reproduces that:

    202 × 0.589 + 154 × 0.357 = 174.0  ->  48.9%   [OK]
    154 × 0.589 + 202 × 0.357 = 162.8  ->  45.7%   [X]

So **Crossed 58.9%, Natural 35.7%** — the original figures. Goalkeepers read
*crossed* kicks better and are wrong about two thirds of the time on natural
ones. The conclusion in the current `results_summary.md` reflects this.

**22. The Natural/Crossed label was defined two different ways**
`04_models/reg_cross_analysis.py` had `Natural = right foot → Left`;
`03_statistical_analysis/reg_cros.py` and `practical_insights.py` had
`Natural = right foot → Right`. ROC-AUC, accuracy and Mann-Whitney p-values are
all invariant to swapping two class names, so no metric was wrong and nothing
ever failed — but every sentence of the form "when X is higher the kick tends
to be Crossed" pointed the wrong way in half the scripts, the class counts were
reported as 202 Natural / 154 Crossed when the convention in force made it the
reverse, and correction 21 was derived from the clash.

The rule is now defined once, in `stats_utils.build_shot_type()`:

    Natural : right foot -> Right  OR  left foot -> Left
    Crossed : right foot -> Left   OR  left foot -> Right

with Left/Right as goal zones **seen by the kicker facing the goal**, so a
crossed kick is the leg swinging across the body midline.

**23. Four scripts crashed on the sample shipped with the repository**
`reg_cross_analysis.py`, `reg_cros.py`, `bio_analysis.py` and
`practical_insights.py` built a table of per-feature test rows and then sorted
it by `p_value`. When a group falls below the minimum of 5 observations no rows
are produced, and the sort raised `KeyError` instead of saying so. That is the
exact situation on `data/sample/`, whose stated purpose is to let a reader
confirm the scripts run — so a third of the pipeline failed at the first thing
a reader would try, including both scripts that produce the headline result.
Guarded via `stats_utils.is_empty()`.

**24. No permutation test on the main target; the others ignored the grouping**
`reg_cross_analysis.py` never ran one at all. Where one did run, it shuffled
labels globally and scored with a plain `StratifiedKFold`, building a null that
assumes clips are independent — the assumption the grouped CV exists to relax,
in a dataset where 414 clips come from 28 videos. `permutation_test()` now
accepts `groups`, permutes labels *within* each source video, and evaluates
with `StratifiedGroupKFold`.

**25. Two `README` commands did not run.** `run_biomechanics.py` was documented
without its required positional `clips_dir`, and the stage 6–7 scripts were
documented as invoked from the repository root when they read
`clips_master.csv` from the working directory.

**26. A majority-class accuracy was printed in a ROC-AUC column.**
"Prediction from kicking foot alone | 0.567" appeared in the robustness table of
three documents, under a column headed ROC-AUC, as the argument that the
Natural/Crossed result is not foot detection. 0.567 is the majority-class
*accuracy*, and accuracy is blind to exactly this kind of association. The
conclusion survives — the true figure is ROC-AUC ≈ 0.503, so the foot really
does carry almost nothing — but the argument as stated did not support it, and
it was the same metric confusion this project spent three documents warning
about.

---

## Known and not fixed

These are defects, not history. Each was previously written in this file in the
past tense, as though the code already did the right thing; none of them does.
They are listed here so that a reader knows what the pipeline actually is.

<a name="k1"></a>
**K1. Joint angles are 2D projections, not 3D world landmarks.**
`pose_extractor.py` reads `lm.x` and `lm.y` only; `pose_world_landmarks` is not
used anywhere in the repository, and there is no `angle_source` column. An angle
measured off an image is a projection, so it depends on how the kicker was
oriented towards the camera, and this dataset mixes four camera positions of
which more than half are side views. Every angle-based figure in this project
carries that dependence.

The 2D-versus-3D comparison quoted elsewhere (0.673 vs 0.646 ROC-AUC with
overlapping intervals) was run outside this repository and cannot be reproduced
from it. Treat it as a note about work done, not as evidence a reader can check.

<a name="k2"></a>
**K2. The crop geometry is never written, so translation features cannot be
un-warped.** `clips_creator.py` computes the smoothed per-frame crop boxes and
discards them; there is no `crop_*_boxes.csv` and no `to_csv` call anywhere in
`01_extraction/`. Consequently `running_speed_kmh`, `kick_foot_speed_ms` and
`support_foot_x/y` are measured inside a window that follows and re-scales the
player, and there is no way to recover ground coordinates after the fact.

Two further problems in the same family, neither previously documented:

- Displacements are computed as `np.diff(np.nan_to_num(...))`, so a missing
  keypoint becomes the coordinate (0, 0) and produces a displacement the size of
  the image diagonal. Nothing clips these series — the anatomical limits in
  `kinematics.py` are applied to the angular quantities only. Across the 414
  clips, `running_speed_kmh` reaches **387 km/h** and `kick_foot_speed_ms`
  reaches **133 m/s**, against medians of 2.95 km/h and 2.88 m/s. The
  `late_max_*` and `max_running_speed_kmh` aggregates are therefore partly
  detectors of detection failure.
- `support_foot_x/y` use the kicking ankle at frame 0 as a proxy for the ball.
  Frame 0 is roughly 800 ms before contact, mid-run-up, so these columns measure
  the extent of the approach and not where the support foot was planted relative
  to the ball. A failed measurement is stored as an exact 0 rather than NaN —
  27 of the 414 clips.

<a name="k3"></a>
**K3. The kicking-foot annotation is still mapped without a guard.** `FOOT_MAP`
accepts `Right`/`Left` only, there is no key normalisation, no
`Foot annotation: N/N rows mapped` line, and no error when values fail to map.
The bug in correction 12 can recur silently and produce a dataset in which a
quarter of the Natural/Crossed labels are wrong.

Related and unresolved: it is not established whether `foot_used` in the
*published* dataset came from the annotation or from the 74.2%-accurate
auto-detection. Correcting it after the fact in the CSV would not help, because
the features themselves were computed relative to whichever leg the detector
chose — `kick_knee_angle` and `supp_knee_angle` would refer to the wrong leg in
those clips, and that cannot be repaired without re-running the pose extraction.
See `data/README.md`.

<a name="k4"></a>
**K4. The clips are not all 25 fps.** Measured across all 414 clips, from the
`time_sec` column that `pose_extractor.py` writes from each video's own frame
rate:

| Frame rate | Clips | Run-up covered by the 20-frame window |
|---|---|---|
| 25 fps | 345 (83.3%) | 760 ms |
| 29.94 fps | 28 | 634 ms |
| 30.03 fps | 22 | 633 ms |
| 50 fps | 10 | 380 ms |
| 59.88 fps | 9 | 317 ms |

The window is defined in *frames* (20), not in seconds, so the 69 clips that are
not 25 fps cover between 317 and 634 ms of run-up instead of 760. Two
consequences: every "N ms before contact" figure is exact only for the 83% at
25 fps, and the early/mid/late phases span different real durations from row to
row, so "the approach run" is not the same interval in every clip. Angular
velocities are unaffected — `kinematics.py` uses each clip's real fps for `dt`.
The timing claims are what suffer.

The 29.94 and 59.88 values are NTSC (30000/1001 and 60000/1001), so these are
broadcasts or re-encodes from a 30 fps chain rather than measurement error.

`reg_cross_analysis.py` audits the frame rate and warns. The aggregation itself
still treats all clips alike, and fixing that means re-running stages 3–5.

**The sample in `data/sample/` is now restricted to 25 fps clips**, so anything
run against it has a single, exact time base. That is a property of the sample,
not of the dataset.

<a name="k5"></a>
**K5. `signal_filter.py` does not filter.** It is configured with
`window=3, poly=2`, and a second-degree polynomial through three points
reproduces those points exactly — the transform is the identity to within
floating-point error. The documented sub-pixel smoothing does not happen, and
the jitter it was meant to remove propagates amplified into `kick_angular_vel`
and squared into `kick_angular_accel`.

<a name="k6"></a>
**K6. `torso_torsion_angle` does not measure torsion.** It is the unsigned angle
between the shoulder line and the hip line *in the image plane*. True axial
rotation appears in a projection as a shortening of the shoulder line, not as an
angle between the two lines, and when the trunk turns past the frontal plane the
projected shoulder vector inverts and the value jumps towards 180° — the sample
maximum is 164°. In practice it tracks orientation relative to the camera. The
plain-language descriptions in the analysis scripts have been corrected to say
what it measures; the column itself is unchanged.

<a name="k7"></a>
**K7. Keypoints that were never detected are stored as 0.0, not NaN**, contrary
to the comment in `signal_filter.py` claiming they will be ignored downstream.
The kinematics then computes angles to the corner of the image, and only values
outside [2°, 178°] are caught.

<a name="k8"></a>
**K8. No annotation reliability was ever measured.** T0, the goal zone, the
kicking foot and `gk_guessed` come from a single annotator with no blind
re-annotation and no second rater. `gk_guessed` is both a target (P1) and the
human baseline the model is compared against, and it is a subjective judgement.

<a name="k9"></a>
**K9. Player identity is not recorded.** Cross-validation is grouped by source
video, which handles camera pose and lighting, but the same penalty taker
appears across compilations. Natural-versus-crossed is substantially a personal
habit, so a model that recognises a kicker can memorise his preferred technique.
This is the most plausible remaining route by which the main result could be
inflated, and grouping by video only partly blocks it.

Fixing K1–K3 requires re-running stages 3–5 and therefore regenerating the
dataset, which would invalidate every published figure. That is the reason they
are open, not an argument that they do not matter.

---

## What the corrections revealed

Three of the four research questions are negative results, and those hold up:
nothing predicts whether the goalkeeper guesses right, nothing predicts the
outcome, and the direction of the kick is not separable from chance by a model
that is not told which foot is kicking.

**Natural vs Crossed** is the one target where there is something: ROC-AUC
**0.697 [0.604, 0.786]**, holding at **0.713 [0.607, 0.815]** with folds grouped
by source video, and the only target where any feature survives Bonferroni.
Removing the two derived lateral columns (219 → 217 features) did not move it —
0.697 before and after. The project's own hypothesis, written in the header of
what was then a secondary script, is that classifying the *type* of movement
should beat classifying absolute direction, because the kinematics are defined
relative to the kicking leg — so a crossed kick is the same body-relative action
whichever foot performs it.

Three checks were added because the result had not been probed properly, and it
survived all three:

1. **Grouped label permutation.** Shuffling labels *within* each source video,
   so the null keeps the group structure instead of assuming clips are
   independent, the observed score is beaten by 1 of 200 shuffles: **p = 0.005**.

2. **Within a single kicking foot.** The label is an XOR of the foot with the
   target side, so a sceptic's first question is whether the model recovers the
   foot and exploits the construction. On the 276 right-footed kicks alone,
   where the XOR is constant, it still reaches **0.658 [0.520, 0.759]**. The 80
   left-footed kicks give 0.634 [0.387, 0.881] — too few to read.

3. **Without the translation features.** An earlier version of this file said
   that those columns reach ROC-AUC 0.689 of the 0.697 on their own, and
   concluded that most of the result rested on the least defensible measurement
   in the project. Run against the current feature set, the 31 translation
   columns alone reach **0.555 [0.324, 0.667]** — a range that includes chance.
   The 0.689 came from an earlier configuration and does not hold. The claim is
   withdrawn.

Two qualifications remain:

- **The six Bonferroni survivors are not six findings.** The threshold is
  0.05/217 = 2.30 × 10⁻⁴; four of the six are `running_speed` variants with
  Cohen's d of 0.04–0.17, which the write-up declines to interpret. The two
  interpretable ones are `contact_trunk_inclination` (p = 0.000148) and
  `contact_kick_knee_angle` (p = 0.000153), both at the contact frame. Read
  those two, not the count.

- **Where the signal sits in time is unresolved.** Every FDR-surviving
  frame-level signal is at frame 18–19, which invites the reading that the model
  is seeing how the foot struck the ball at the moment it struck it — close to
  tautological given that Natural/Crossed *is* the striking technique. Step 2b
  was added to settle that, and did not: the early phase alone gives 0.592
  [0.505, 0.691], early+mid gives 0.493 [0.378, 0.572], all features give 0.697.
  A non-monotone pattern, where adding the mid phase is worse than the early
  phase alone, fits no account of a real signal and is most likely fold noise in
  restricted feature sets at n=356. It establishes neither anticipation nor its
  absence.

**27. The gap between Natural/Crossed and Left/Right is now explained**
An earlier version of this file said it remained unexplained. It does not. The
two labels are the same partition once the kicking foot is known, and the
features are defined relative to the kicking leg — so the same value points to
Left in a right-footed kick and to Right in a left-footed one, cancels
marginally, and is invisible to a univariate filter. An exploratory run
supplying the foot and mirroring the features by it brings Left/Right up to the
level reported for Natural vs Crossed.

That run is not part of this repository and no figure from it is quoted in the
write-ups, because it was crude by design: it mirrored every column rather than
only the lateralised ones, used more features than clips, and was not
permutation-tested. It is recorded here because of what it settles, not because
of what it scored. It confirms the project's own hypothesis — that classifying
the type of movement works because the kinematics are body-relative — and it
costs the project its claim that Natural vs Crossed is a privileged target. It
is not. It is the representation in which the signal is already aligned.

**28. A fifth target: centre vs cornered**
The 45 kicks aimed at the middle column had never been tested against anything —
`macro_zone` drops them before every direction analysis. All four models land
around 0.62 with no fold range clearing chance; a grouped permutation gives
p = 0.105, against the 0.01 that five targets on one dataset would require. An
exploratory feature screen, not published here, found the largest individual
effects in the project, sitting in the early and mid phases rather than at
contact — the opposite of the Natural vs Crossed pattern.

With 45 positives against 356 a null does not distinguish absence from lack of
power, so it is recorded as untested rather than as a negative result. Note also
that it is the fifth target tested on this dataset with no correction across
that family — the reason the 0.01 threshold is cited above rather than 0.05.

**One prediction in this file was half wrong.** A previous version argued that
because Natural/Crossed and Left/Right are the same partition once the kicking
foot is known, the gap between them was that one model had been denied the foot
— and that supplying it should close the gap. Supplying the foot moves the
direction target only a little; correction 27 shows that what closes the gap is
mirroring the features by the foot, not naming it. The diagnosis was right about
where to look and wrong about the mechanism.

The corrections did not produce the Natural/Crossed result — they removed what
was standing in front of it, and the checks since have not knocked it down.
