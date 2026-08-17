# Penalty Biomechanics — Patterns Report

> **Note on these figures.** All are from the current 217-feature run. Two
> derived columns that reintroduced excluded lateral quantities were removed
> (219 → 217); the headline did not move. Separately, the Natural/Crossed class
> names were the wrong way round in half the scripts: the convention is now
> `Natural = right foot → Right`, which makes **Crossed the majority class**.
> See [`../../CHANGELOG.md`](../../CHANGELOG.md), 15 and 21–26.


Every figure here comes from repeated 5-fold cross-validation (5 seeds) with
all preprocessing refit inside each fold, and every p-value is corrected for
multiple testing. The primary metric is **ROC-AUC, chance = 0.500**; brackets
are the 95% range across folds. A range that includes 0.500 is not a result.

Where an earlier version of this report made a stronger claim, the corrected
figure is given alongside it. See [`CHANGELOG.md`](../../CHANGELOG.md) for the
full list of corrections.

## 1. Dataset Overview

- 414 penalty-kick clips, a 20-frame window ending at ball contact. 345 clips
  are 25 fps (760 ms of run-up); the other 69 run at 29.94–59.88 fps and cover
  between 317 and 634 ms, so the "ms before contact" figures below are exact for
  the former and overestimates for the latter — see
  [CHANGELOG K4](../../CHANGELOG.md#k4)
- 15 biomechanical variables per frame via MediaPipe Pose, aggregated to 250
  per-clip features of which **217 are used** (detection-quality metrics and
  camera-dependent lateral features are excluded)
- 28 source videos; the largest contributes 80 clips, so cross-validation is
  also run with folds grouped by video
- 154 Natural, 202 Crossed (356 clips with a Left/Right target zone)
- Outcome: 317 goals, 71 saved, 13 off target, 13 post

## 2. The one robust finding — Natural vs Crossed

**SVM (RBF), ROC-AUC 0.697 [0.604, 0.786]** — the whole fold range above
chance, and all four models clear it.

| Robustness check | ROC-AUC |
|---|---|
| Random 5-fold, 5 seeds | 0.697 [0.604, 0.786] |
| **Folds grouped by source video** | **0.713 [0.607, 0.815]** |
| **Label permutation within source video** (200 shuffles) | **p = 0.005** |
| Right-footed kicks only (n=276) | 0.658 [0.520, 0.759] |
| Left-footed kicks only (n=80) | 0.634 [0.387, 0.881] |
| Translation features only (31 columns) | 0.555 [0.324, 0.667] |
| Prediction from kicking foot alone | 0.503 |

Grouping by video does not hurt the result, so it is not an artefact of which
broadcast a clip came from. It is not foot detection: the kicking foot alone
gives 0.503. It survives permuting the labels within each source video, which
keeps the group structure intact in the null. And it survives inside the
right-footed subset alone, where the XOR with the foot is constant — so it is a
property of the movement, not of how the label is built. The left-footed subset
is too small to read.

**Statistical support:** of 217 features tested (Mann-Whitney U), 35 reach
p < 0.05 against 10.9 expected by chance, **11 survive Benjamini-Hochberg FDR
and 6 survive Bonferroni**. This is the only target in the project where
anything survives Bonferroni.

**What those 6 are, though.** The Bonferroni threshold is 0.05/217 = 2.30 × 10⁻⁴.
Four of them are `running_speed` variants with effect sizes of d = 0.04–0.17,
noted below as uninterpretable; the other two are `contact_trunk_inclination`
(p = 0.000148) and `contact_kick_knee_angle` (p = 0.000153). Read those two, not
the count of six.

This does **not** mean the result rests on the translation columns: on their own
those 31 features reach only 0.555 [0.324, 0.667], a range that includes chance.

The discriminative features with a non-negligible effect size:

| Feature | p | Cohen's d | Natural | Crossed |
|---|---|---|---|---|
| `contact_trunk_inclination` | 0.00015 | +0.40 | 7.8° | 4.6° |
| `contact_kick_knee_angle` | 0.00015 | −0.38 | 146.7° | 156.7° |
| `late_max_kick_ankle_angle` | 0.00132 | −0.38 | 155.9° | 164.4° |
| `delta_kick_ankle_angle` | 0.00088 | −0.32 | 1.3° | 8.3° |

Crossed kicks show a **straighter kicking knee at contact**, a **more open
ankle**, and a **more upright trunk**. Natural kicks lean further forward.

The four features with the smallest p-values are all `running_speed` variants
(p < 0.0001) but with effect sizes of d = 0.04–0.17 — negligible. That
combination signals skewed distributions rather than a real group difference,
and these are precisely the features measured in the tracker's moving crop.
They are not interpreted here.

**Why this works when left-vs-right does not.** The kinematic features are
defined relative to the kicking and support legs, so a crossed kick is the same
body-relative action whichever foot performs it — the leg swinging across the
torso. "Left vs Right" merges two mirror-image actions under one label and the
signal cancels.

That cancellation can be undone by changing the representation rather than the
data — see section 4. Natural vs Crossed is therefore not a privileged target;
it is the framing in which the signal is already aligned.

**Accuracy, framed against the right baseline:**

| Reference | Accuracy |
|---|---|
| Always predict "Crossed" (majority class, 202 of 356) | 56.7% |
| Best model | **65.3%** (+8.5 pp) |
| Goalkeeper, in real time | 48.9% |

The model beats the goalkeeper by 16.4 pp, but a constant prediction already
beats the goalkeeper by 7.8 pp without looking at the kicker. What the
biomechanics contribute is +8.5 pp over the majority class. *(The earlier
version of this report gave "+13.8 pp over the goalkeeper" as the headline; the
majority-class baseline was missing.)*

## 3. Goalkeeper Predictability (P1) — one small effect, no prediction

**Logistic Regression, ROC-AUC 0.554 [0.492, 0.643]** — the fold range spans
chance. No model classifies individual kicks above chance.

Of 217 features, 32 reach p < 0.05 against 10.9 expected by chance. Two survive
FDR correction (q = 0.0425), none survives Bonferroni, and they describe the
same thing:

| Feature | p | q | Cohen's d | Guessed | Not guessed |
|---|---|---|---|---|---|
| `late_mean_kick_hip_angle` | 0.00023 | 0.0425 | −0.41 | 148.2° | 151.6° |
| `late_min_kick_hip_angle` | 0.00039 | 0.0425 | −0.36 | 126.2° | 132.6° |

Kickers who keep the kicking-leg hip **more extended** through the late swing
are harder for the goalkeeper to read. `contact_kick_hip_angle` points the same
way (d = −0.28) but does not survive correction (q = 0.171).

This is a small effect on one feature family. It is the strongest individual
signal in the project outside Natural vs Crossed, and it still does not
translate into above-chance classification.

*Note: an earlier version reported that the LSTM reached PR-AUC 0.579 on this
target and "clearly beat every tabular model". That figure was the best of ~120
epochs scored on the test fold itself. With early stopping on a proper inner
validation split, LSTM and CNN 1D both sit at ROC-AUC 0.50–0.53 on both
targets, with every fold range spanning chance.*

## 4. Kick Direction (P2) — no detectable signal

**XGBoost, ROC-AUC 0.547 [0.411, 0.654].** Grouped by source video: 0.557.

**This is a null result for this representation, not a claim that direction
cannot be predicted from these features.** Left/Right and Natural vs Crossed are
the same partition once the kicking foot is known, and the features are defined
relative to the kicking leg, so the direction signal flips sign with the foot
and cancels under this label. An exploratory run supplying the foot and
mirroring the features by it brings this target up to the level reported for
Natural vs Crossed in section 2. That run is not part of this repository and no
figure from it is quoted here; it is noted because it explains why the
body-relative label separates and this one does not.

Statistical tests: 20 of 217 features reach p < 0.05, against **10.9 expected
by chance**. **Nothing survives Bonferroni or FDR** (smallest q = 0.168). The
largest effect size is d = 0.30; the features with the smallest p-values have
effects of d = 0.06–0.14, all negligible.

The earlier report described a "PR-AUC ceiling at ~0.635 confirmed after 3
rounds of feature engineering". The diagnosis — that the bottleneck was signal
rather than model capacity — was right. What the corrected analysis adds is
that the signal is not weak but absent at this sample size, and that PR-AUC was
the wrong metric to see it with: on a 53/47 split it sits at the class
prevalence and drifts above it under fold noise.

## 4b. Centre vs cornered (P4) — a lead, not a result

The 45 kicks aimed at the middle column of the goal sit outside every other
direction analysis in this project: `macro_zone` covers only the corner and side
zones, so they are dropped before P2 and before Natural vs Crossed. This target
puts them back in, and unlike the others it does not depend on the kicking foot,
so none of the representation problem in section 2 applies.

All four models land around **0.62** and **none clears chance** — the best fold
range is [0.475, 0.776]. The grouped label permutation gives p = 0.105, well
short of the 0.01 that five targets on one dataset would require.

An exploratory feature screen, not part of this repository, found the strongest
individual effects of any target in this project, and found them in the **early
and mid phases** rather than at contact. That is the opposite of the Natural vs
Crossed pattern, and would be genuine anticipation if it held up. It is the
reason this target is worth returning to; it is not a reason to believe it yet.

**Read this as untested, not as negative.** With 45 clips in the positive class
against 356, a null does not distinguish "no signal" from "not enough clips",
and the asymmetry runs one way: a clear positive would have meant something, an
absence means very little. At ~800 clips this becomes ~90 central kicks and
becomes testable. It is the first thing worth running if the dataset grows.

## 5. Outcome (P3) — nothing

10 of 217 features reach p < 0.05 — **fewer than the 10.9 expected by chance**.
Smallest q-value 0.366. There is no evidence relating pre-contact biomechanics
to whether the penalty is scored.

## 6. Signal timeline — no usable anticipation

80 frame-level Mann-Whitney tests — 4 distinct base variables × 20 frames.
The five features selected by `f_classif` map onto four underlying frame
variables, and testing per feature rather than per variable used to put the same
test in the family twice. Uncorrected: 13 significant, against 4.0 expected by
chance. After Benjamini-Hochberg: **6 survive, and every one of them is at
frame 18 or 19 — 0 to 40 ms before contact.**

| Base variable | Frame | ms before contact | q |
|---|---|---|---|
| `kick_knee_angle` | 18 | 40 | 0.0001 |
| `kick_knee_angle` | 19 | 0 | 0.0031 |
| `trunk_inclination` | 18 | 40 | 0.0031 |
| `trunk_inclination` | 19 | 0 | 0.0031 |
| `supp_knee_angle` | 19 | 0 | 0.0215 |
| `kick_ankle_angle` | 18 | 40 | 0.0236 |

A goalkeeper needs roughly 200 ms to initiate a dive. Nothing here arrives in
time.

The earlier version of this report listed first-significant frames from
uncorrected tests and concluded that the knee angle was readable 480 ms before
contact, with "enough lead time to react in theory". Those were uncorrected
multiple-comparison artefacts. The warning sign was in the same table:
`mean_visibility_score` appeared as a "signal" at 320 ms — it is MediaPipe's
detection confidence, not a body movement.

The per-frame PR-AUC table for P2 has also been removed: it used the wrong
metric and its values oscillated above and below the baseline with no pattern.

## 7. Decision rules

| Rule | In-sample leaf purity | Out-of-fold precision | Coverage |
|---|---|---|---|
| Natural vs Crossed | 100% | **72.4% ± 18.9%** | 15.8% of clips |
| P2 direction | 91.0% | **60.2% ± 25.4%** | 12% of clips |

In-sample purity overstates the Natural vs Crossed rule by 27.6 points and the
direction rule by roughly 30. A
depth-3 tree will always find a clean-looking corner of the data it was shown;
the out-of-fold number is what answers "how often is this right on a new
penalty". The earlier "94% confidence, 28 historical cases" rule is, in
practice, barely better than a coin flip.

## 8. What Did Not Work

- **Feature engineering** (3 rounds, 12 features): all redundant with existing features
- **Stacking ensemble:** nested CV consumes training data, does not beat SVM with 414 samples
- **Hyperparameter tuning SVM:** unstable across folds, C=1 default already optimal
- **Deep learning, both targets:** 414 clips do not support the extra capacity; LSTM and CNN 1D sit at chance
- **Translation features as physical quantities:** measured inside the tracking crop, so they capture tracker lag rather than ground speed

## 9. Practical Implications

**For scouting and video analysis, with the timing caveat.** Every surviving
signal is at the contact frame or one frame before it, so what follows describes
what can be read *from the strike*, not what can be anticipated before it.
Natural vs Crossed can be classified at
65.3% accuracy, 8.5 pp above always guessing the majority class. That is a real
but modest edge, and it is the only model in this project worth using.

**For goalkeeper training.** Nothing here supports a real-time cue. Every
frame-level signal that survives correction arrives within 40 ms of contact,
and a dive takes ~200 ms to initiate.

**For kickers.** The clearest actionable finding is the reverse of the original
question: kickers who keep the kicking-leg hip more extended through the late
swing are harder to read.

## 10. What this project shows

Across 217 features and five targets, the body does **not** reliably
betray where a penalty is going, and nothing measurable before contact predicts
whether it is scored. What it does betray, robustly enough to survive grouping
by source video and Bonferroni correction, is whether the kick is **natural or
crossed** — the geometry of the leg swinging across the torso — and even that
only becomes measurable in the final 40 ms.

That is a negative result for the question the project set out to answer, and a
positive one for a sharper question the analysis uncovered along the way.
