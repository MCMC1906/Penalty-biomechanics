# Results Summary

> **All figures below are from the current 217-feature run.** Two derived
> columns (`support_foot_angle_deg`, `deception_score`) that reintroduced the
> signed lateral quantities this project excludes have been removed, taking the
> feature set from 219 to 217. The headline did not move: ROC-AUC 0.697 before
> and after.
>
> Two things did change and are **not** cosmetic:
> - **The Natural/Crossed class names were the wrong way round in half the
>   scripts.** The convention is now fixed as `Natural = right foot → Right`,
>   which makes **Crossed the majority class (202 of 356)**. Metrics are
>   invariant to the swap; interpretive sentences and class counts are not.
> - **The goalkeeper-readability figures were re-derived** — see §3.

## How to read these numbers

- **Primary metric is ROC-AUC, chance = 0.500** regardless of class balance.
  Brackets are the 95% range across folds. **A range that includes 0.500 is not
  a result.** Earlier versions of this document led with PR-AUC; on the near
  balanced targets here (46/54 and 53/47) it sits at the class prevalence and
  drifts above it under fold noise, which made chance-level models look like
  findings.
- 5-fold cross-validation **repeated over 5 seeds**, with all preprocessing
  refit inside each fold, plus a second estimate with folds **grouped by source
  video** (28 videos, the largest contributing 80 clips).
- 217 features tested per target, so **10.9 reach p < 0.05 by chance alone**.
  Bonferroni (threshold 2.30 × 10⁻⁴) and Benjamini-Hochberg are reported next to
  that expectation. The statistical scripts and the model scripts screen the
  same 217.

See [`../CHANGELOG.md`](../CHANGELOG.md) for the corrections applied and what
each of them changed.

---

## Natural vs Crossed — the one result that holds

**SVM (RBF), ROC-AUC 0.697 [0.604, 0.786]**, accuracy 65.3%. All four models
clear chance (LR 0.652, RF 0.681, XGBoost 0.671).

| Robustness check | ROC-AUC |
|---|---|
| Random 5-fold, 5 seeds | 0.697 [0.604, 0.786] |
| **Folds grouped by source video** | **0.713 [0.607, 0.815]** |
| **Label permutation within source video** (200 shuffles) | **p = 0.005** |
| Right-footed kicks only (n=276) | 0.658 [0.520, 0.759] |
| Left-footed kicks only (n=80) | 0.634 [0.387, 0.881] |
| Translation features only (31 columns) | 0.555 [0.324, 0.667] |
| Prediction from kicking foot alone | 0.503 |

Grouping by video does not hurt it, so it is not an artefact of which broadcast
a clip came from. Note that the grouped figure coming out *higher* is not extra
evidence — with 28 groups, one holding 80 clips, the grouped partitions are very
uneven. "Grouping does not degrade it" is the claim it supports.

It is also not foot detection. The kicking foot alone reaches ROC-AUC 0.503:
of 276 right-footed kicks 156 went Right and of 80 left-footed 46 went Right, so
P(Natural | right foot) = 0.565 against P(Natural | left foot) = 0.575 — almost
nothing to exploit. And it survives inside a single foot: on the 276 right-footed
kicks alone, where the XOR with the foot is constant, the model still reaches
0.658 [0.520, 0.759]. The 80 left-footed kicks give 0.634 but with an interval
from 0.387 to 0.881 — too few clips to read either way.

The permutation test is the strongest of these checks: shuffling labels *within*
each source video, so the null preserves the group structure rather than
assuming clips are independent, the observed 0.697 is beaten by 1 of 200
shuffles (**p = 0.005**).

**Statistical support:** 35 of 217 features reach p < 0.05 (against 10.9
expected by chance), **11 survive BH-FDR and 6 survive Bonferroni**. This is
the only target where anything survives Bonferroni.

**What those six are.** The Bonferroni threshold is 0.05/217 = 2.30 × 10⁻⁴. Four
of the six are `running_speed` variants (p < 0.0001) with effect sizes of
d = 0.04–0.17 — negligible, and the combination of a tiny p-value with a
negligible effect signals skewed distributions rather than a group difference.
The other two are `contact_trunk_inclination` (p = 0.000148) and
`contact_kick_knee_angle` (p = 0.000153). Of the remaining features in the table
below, `late_std_running_speed_kmh` misses at 0.000234 and the two ankle
features at 0.00088 and 0.00132 do not come close. Read the two interpretable
survivors, not the count of six.

Note that this does **not** mean the result rests on the translation columns.
Those 31 features on their own reach only 0.555 [0.324, 0.667] — a range that
includes chance. An earlier version of this document put that figure at 0.689
and drew the opposite conclusion; it came from an earlier configuration and does
not hold.

Two further qualifications:

- **Where the signal sits in time is unresolved.** Restricting the model to the
  earliest frames gives 0.592 [0.505, 0.691] on the early phase alone (60
  features) and 0.493 [0.378, 0.572] on early+mid (120), against 0.697 on all
  217. That is not monotone — adding the mid phase makes it worse than the early
  phase alone — and no account of a signal arriving progressively through the
  run-up predicts it. So this control neither establishes anticipation nor
  establishes that the separation comes only from the contact frame. Most likely
  it is fold noise in restricted feature sets at n=356. The frame-level tests
  still put every surviving signal at frames 18–19, and that remains the basis
  for the "no usable anticipation" conclusion.

  *Not monotone: adding the mid phase is worse than the early phase alone. Most
  likely fold noise at n=356 — it establishes neither anticipation nor its
  absence.*

- **Correction is within-target, not study-wide.** This is the fourth target,
  defined after three null ones. At 0.05/868 across the study, the two surviving
  angle features (p ≈ 0.00015) would not survive.

The features with a non-negligible effect:

| Feature | p | d | Natural | Crossed |
|---|---|---|---|---|
| `contact_trunk_inclination` | 0.00015 | +0.40 | 7.8° | 4.6° |
| `contact_kick_knee_angle` | 0.00015 | −0.38 | 146.7° | 156.7° |
| `late_max_kick_ankle_angle` | 0.00132 | −0.38 | 155.9° | 164.4° |
| `delta_kick_ankle_angle` | 0.00088 | −0.32 | 1.3° | 8.3° |

Crossed kicks show a straighter kicking knee at contact, a more open ankle, and
a more upright trunk. The four features with the smallest p-values are all
`running_speed` variants with d = 0.04–0.17 (negligible) — skewed
distributions, not a group difference, and measured in the tracker's moving
crop. They are not interpreted.

**Why this works when left-vs-right does not.** Classifying the *type* of
movement captures a body-crossing motion — hip rotation, torso torsion — with a
more consistent biomechanical signature than "left vs right" in isolation,
which depends on which foot is kicking and merges two mirror-image physical
actions into one label. The features are defined relative to the kicking and
support legs, so a crossed kick is the same body-relative action whichever foot
performs it. Mechanically, the direction signal flips sign with the foot, and
`SelectKBest` with `f_classif` is a univariate filter — no feature has a
marginal association with Left/Right, so the interaction never reaches the
classifier.

**Accuracy against the right baseline:**

| Reference | Accuracy |
|---|---|
| Always predict "Crossed" (majority class, 202 of 356) | 56.7% |
| Best model | **65.3%** (+8.5 pp) |
| Goalkeeper, in real time | 48.9% |

The model beats the goalkeeper by 16.4 pp, but a constant prediction already
beats the goalkeeper by 7.8 pp without looking at the kicker. The biomechanics
contribute **+8.5 pp over the majority class**. *(An earlier version gave
"+13.8 pp over the GK" as the headline; the majority-class baseline was
missing.)*

**Timing.** Of 80 frame-level tests (4 distinct base variables × 20 frames), 13
reach p < 0.05 against 4.0 expected by chance, and **six survive FDR correction
— every one at frame 18 or 19**, i.e. 0 to 40 ms before contact.
`trunk_inclination`, `kick_knee_angle` and `kick_ankle_angle` first survive at
frame 18; `supp_knee_angle` only at frame 19. Nothing is significant at 200 ms
or earlier. Two caveats: the tests previously double-counted, because several
selected features map onto the same underlying frame variable and each generated
its own identical row inside the FDR family — the five selected features reduce
to four distinct base variables;
and the clips are not all 25 fps — 345 of 414 are, and for the other 69 the
same 20 frames span as little as 317 ms, so "40 ms" is an overestimate there. Useful for video analysis and scouting, not for a dive decision.
*(An earlier version reported the knee angle as readable from ~480 ms, "in
theory enough lead time to react". Those were uncorrected multiple-comparison
artefacts.)*

**Not symmetric across feet.** Right-footed kickers (n=276) yield 9 features
surviving FDR; left-footed (n=80) yield none — though with 80 clips that is
as likely to be a power problem as a real asymmetry. At the model level the
picture is the same: **0.658 [0.520, 0.759]** on the right-footed subset, which
clears chance, against **0.634 [0.387, 0.881]** on the left-footed one, which is
too wide to read. The right-footed result is the one that matters, because
within a single foot the XOR with the foot is constant — so the signal is a
property of the movement, not of the label's construction.

**No effect on scoring.** Natural 81.8% goals vs Crossed 78.2%, chi² p = 0.481.

---

## P1 — Goalkeeper Predictability

**Logistic Regression, ROC-AUC 0.554 [0.492, 0.643]** — the fold range spans
chance. No model classifies individual kicks above chance.

Of 217 features, 32 reach p < 0.05 against 10.9 expected by chance; two survive
FDR correction (q = 0.0425), and none survives Bonferroni:

| Feature | p | q | d | Guessed | Not guessed |
|---|---|---|---|---|---|
| `late_mean_kick_hip_angle` | 0.00023 | 0.0425 | −0.41 | 148.2° | 151.6° |
| `late_min_kick_hip_angle` | 0.00039 | 0.0425 | −0.36 | 126.2° | 132.6° |

Kickers who keep the kicking-leg hip **more extended** through the late swing
are harder to read. This is the strongest individual signal in the project
outside Natural vs Crossed. It does not survive Bonferroni, and it does not
become a working classifier.

**Why P1 is harder than P2.** P2 predicts the kicker's own physical action; P1
predicts a *goalkeeper's* perceptual read of that action, one extra inferential
step removed from the biomechanics being measured.

The phase-3 analysis tested this directly:

- The Spearman correlation between P1's and P2's top feature rankings is not
  significant (**rho = 0.486, p = 0.329** on 6 common features) — the two
  targets are not simply the same signal seen from two angles.
- Conditionally, the goalkeeper does guess more often when the kick is
  biomechanically "readable" by the P2 model: **55.2% when the model was right
  (n=181) vs 42.3% when it was wrong (n=175)**, chi² p = 0.019.
- Concordance analysis nonetheless suggests the goalkeeper is picking up
  different cues from the ones driving the model.

**A caveat on the target itself.** All 71 saved penalties are labelled as
guessed, so 37% of the positive class is defined by the outcome rather than by
an independent reading of the movement. P1 and P3 are not independent
questions.

*An earlier version reported the LSTM at PR-AUC 0.579 and concluded that deep
learning "clearly outperforms every tabular model" on this target — the
opposite pattern from P2. That figure was the best of ~120 epochs scored on the
test fold itself. With early stopping on a proper inner validation split, LSTM
and CNN 1D sit at ROC-AUC 0.50–0.53 on **both** targets. The pattern does not
exist.*

---

## P2 — Kick Direction (Left vs Right)

**XGBoost, ROC-AUC 0.547 [0.411, 0.654].** Grouped by source video: 0.557.

**This is a null result for this representation, not a claim that the direction
of a penalty cannot be predicted from these features.** Left/Right and Natural
vs Crossed are the same partition once the kicking foot is known, and the
features are defined relative to the kicking leg — so the same value points to
Left in a right-footed kick and to Right in a left-footed one, and cancels. An
exploratory run supplying the foot and mirroring the features by it brings this
target up to the level reported for Natural vs Crossed. That run is not part of
this repository and no figure from it is quoted here; it is noted because it
explains why the body-relative label separates and this one does not.

20 of 217 features reach p < 0.05 against 10.9 expected by chance, and **nothing
survives Bonferroni or FDR** (smallest q = 0.168). The largest effect size is
d = 0.30; the smallest p-values belong to features with d = 0.06–0.14.

**What was tested and discarded:** 3 rounds of feature engineering,
hyperparameter tuning, and stacking — none improved on the plateau. The
diagnosis at the time was that the bottleneck was signal rather than model
capacity, and that was right. What the corrected analysis adds is that the
signal is not weak but absent at this sample size, and that PR-AUC was the
wrong instrument to see it with.

**What it would take.** More clips: at n=414 the fold ranges are wide enough
that an effect the size of the Natural vs Crossed one could go undetected.
Player identity in the metadata, so folds can be grouped by kicker. Crop
geometry, so translation features can be measured in a fixed reference frame
rather than inside the moving crop.

---

## P4 — Centre vs cornered: a lead, not a result

The 45 kicks aimed at the middle column of the goal sit outside every other
direction analysis here: `macro_zone` covers only the corner and side zones, so
those clips are dropped before P2 and before Natural vs Crossed. This target
puts them back in, and it does not depend on the kicking foot — so none of the
representation problem above applies.

All four models land around **0.62**, and **none of them clears chance**: the
best fold range is [0.475, 0.776]. The grouped label permutation gives
p = 0.105, well short of the 0.01 that five targets on one dataset would
require.

An exploratory feature screen, not part of this repository, found the strongest
individual effects of any target here, concentrated in the early and mid phases
rather than at contact. That is the opposite of the Natural vs Crossed pattern
and would be genuine anticipation if it held up — which is the reason this is
worth returning to, and not a reason to believe it yet.

**It is untested rather than negative.** With 45 clips in the positive class
against 356, a null does not distinguish "no signal" from "not enough clips",
and the asymmetry runs one way: a clear positive would have meant something, an
absence means very little. At ~800 clips this becomes ~90 central kicks and
becomes testable, and it is the first thing worth running if the dataset grows.

---

## P3 — Outcome (goal vs not)

10 of 217 features reach p < 0.05 — **fewer than the 10.9 expected by chance**.
Smallest q-value 0.366. No evidence that pre-contact biomechanics relate to
whether the penalty is scored.

---

## What the goalkeeper actually reads

The clearest finding about the goalkeeper is not about the model at all:

| Kick type | GK guessed correctly |
|---|---|
| Crossed (n=202) | **58.9%** |
| Natural (n=154) | **35.7%** |

chi² p < 0.0001. Goalkeepers read **crossed** kicks far better than natural
ones, and are wrong about two thirds of the time when the kicker goes to his
natural side.

*(This figure has been wrong once in each direction. The number came from
`practical_insights.py`, which used the opposite class convention from
`reg_cross_analysis.py`, so it was checked against the wrong label and
"corrected" the wrong way. Arithmetic settles it: the goalkeeper's overall
accuracy on these 356 kicks is 48.9%, and only one assignment reproduces that —*
202 × 0.589 + 154 × 0.357 = 174.0 → 48.9% ✓, *against* 154 × 0.589 + 202 × 0.357
= 162.8 → 45.7% ✗.*)*

---

## What Did Not Work

- **Feature engineering, tuning and stacking for P2** — 3 rounds, no
  improvement. The bottleneck was signal, and there turned out to be none.
- **Deep learning on both targets** — with 414 clips the extra capacity of a
  neural sequence model does not pay off; LSTM and CNN 1D sit at chance. Note
  that the sequence models were never run on Natural vs Crossed, which is the
  target with a signal and the one where a model that sees the whole time course
  would be most informative.
- **Using P2's important features to explain P1** — the two targets do not
  share a significant feature-importance ranking (rho = 0.486, p = 0.329), so
  there is no shortcut from "what predicts direction" to "what predicts the
  goalkeeper's guess".
- **Translation features as physical quantities** — computed inside the
  tracking crop, so they measure tracker lag rather than ground speed. They
  cannot be read in km/h. Nor do they carry the result: on their own the 31
  translation columns reach only 0.555 [0.324, 0.667] on Natural vs Crossed.
- **Comparing models on a single split** — the gap between the best and second
  best model was smaller than the fold-to-fold spread. An earlier version
  concluded that SVM underperformed Random Forest on Natural vs Crossed; under
  repeated CV the ordering reverses, and the difference is noise either way.
- **Mirror augmentation as a training technique** — still not implemented. It
  was used once, in a small and deliberately crude form, purely as a diagnostic
  to establish that Natural vs Crossed and P2 are the same result in two
  representations. That run mirrored every column rather than only the
  lateralised ones and is not part of any model reported here.
