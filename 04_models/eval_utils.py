"""
eval_utils.py
=============
Shared evaluation utilities for 04_models/.

Why this file exists
--------------------
Every model script used to carry its own copy of the EXCLUDE set and its own
5-fold loop with a single random_state. That made three problems easy to miss:

  1. A single CV split. The gap between the best and second-best model was
     smaller than the fold-to-fold standard deviation, so "model X is best"
     was not a supported claim. -> repeated CV over several seeds.

  2. Clips are not independent. 414 clips come from 28 source videos (one
     video contributes 80 clips). Clips from the same video share camera pose,
     lighting and possibly the same kicker, so a random split can leak
     video-level idiosyncrasies. -> GroupKFold by video_name as a second,
     stricter estimate.

  3. Pose-detection quality (mean_visibility_score) was being fed to the models
     as if it were a biomechanical variable. -> excluded here, centrally.

Primary metric is ROC-AUC, not PR-AUC. Both targets are close to balanced
(P2 53/47, P1 46/54), and on balanced data PR-AUC sits near the class
prevalence and drifts above it under fold noise, which makes a model with
ROC-AUC 0.52 look like it "beats the baseline". ROC-AUC has a fixed 0.5
chance level and does not have that failure mode. PR-AUC is still reported,
alongside its own baseline, for continuity with the earlier results.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

# -- Configuration ----------------------------------------------------

N_SPLITS  = 5
N_REPEATS = 5      # 25 fits per model -- enough to separate signal from split luck
SEED      = 42

# -- Columns that are not biomechanical features ----------------------
#
# Metadata, target-derived labels, and categorical codes.
#
# mean_visibility_score and everything derived from it are MediaPipe
# detection-confidence metrics. They describe recording/detection quality,
# not the kicker's body, and a model that uses them is partly learning which
# broadcast a clip came from. They were previously included (15 of the 250
# features); they are excluded here.
EXCLUDE = {
    # camera_angle and kicking_leg are no longer produced by the pipeline;
    # kept here so older CSVs still load cleanly.
    'clip_id', 'video_name', 'event_id', 'camera_angle',
    'target_zone', 'zone_name', 'macro_zone', 'gk_guessed',
    'gk_guessed_label', 'outcome', 'foot_used', 'foot_used_label',
    'foot_right', 'kicking_leg', 'n_frames',
    'support_foot_lateral_zone', 'support_foot_depth_zone',
    'support_foot_quadrant', 'proximal_distal_valid',
    'peak_hip_frame', 'peak_knee_frame', 'peak_ankle_frame',
}

# Substrings that disqualify a column from being used as a feature.
EXCLUDE_SUBSTRINGS = (
    'visibility',              # pose-detection quality, not biomechanics
    'lateral_trunk_lean',      # signed left/right -- not comparable across camera angles
    'support_foot_x',          # idem
    'deception_lateral',       # derived from lateral_trunk_lean
    'support_foot_angle_deg',  # = arctan2(y, x) -- reintroduces the sign of x
    'deception_score',         # mean that still includes the lateral component
)

# Why the two lateral families are excluded
# -----------------------------------------
# Both are signed left/right quantities measured along the image x axis. The
# dataset mixes four camera positions (frente, costas, esquerda, direita):
#
#   frente vs costas  -- 180 degrees apart, so the sign is inverted between them
#   esquerda/direita  -- side views, where the goal's left-right axis points
#                        into the screen; image x encodes the run-up direction
#                        instead, so it is a different quantity altogether
#
# Mirroring fixes the first case but not the second, and the side views are
# more than half the dataset -- too many to drop. The columns therefore mean
# different things in different clips and are not usable as features for a
# left-vs-right target. They are still computed and stored per frame, so the
# decision is reversible if the camera geometry is ever resolved.
#
# Two derived columns defeated that exclusion until they were added above:
#
#   support_foot_angle_deg = degrees(arctan2(support_foot_y, support_foot_x))
#       A bijective re-encoding of support_foot_x in polar coordinates -- the
#       sign of x survives intact as "near 0 deg" vs "near +/-180 deg", which
#       makes it a near-direct proxy for the kicking foot. Excluding
#       support_foot_x while keeping this was excluding nothing.
#   deception_score
#       README and CHANGELOG 15 describe it as the mean of deception_torsion
#       and deception_trunk; build_clips_master.py still averages in the
#       lateral component. Both clean components remain available as their own
#       columns, so dropping the composite loses no information.
#
# support_foot_distance = sqrt(x**2 + y**2) is deliberately NOT excluded: it is
# a magnitude and carries no sign, so the camera-inversion argument does not
# apply to it.


def feature_columns(df: pd.DataFrame, verbose: bool = True) -> list:
    """Numeric feature columns, with metadata and quality metrics removed."""
    feats, dropped_quality = [], []
    for c in df.columns:
        if c in EXCLUDE or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(s in c for s in EXCLUDE_SUBSTRINGS):
            dropped_quality.append(c)
            continue
        feats.append(c)
    if verbose and dropped_quality:
        print(f"  {len(feats)} features "
              f"({len(dropped_quality)} detection-quality columns excluded)")
    return feats


# -- Cross-validation -------------------------------------------------

def _score_split(pipe, X, y, tr, te):
    m = clone(pipe)
    m.fit(X.iloc[tr], y[tr])
    proba = m.predict_proba(X.iloc[te])[:, 1]
    if len(np.unique(y[te])) < 2:
        return None
    return {
        'roc_auc' : roc_auc_score(y[te], proba),
        'pr_auc'  : average_precision_score(y[te], proba),
        'accuracy': accuracy_score(y[te], (proba >= 0.5).astype(int)),
    }


def repeated_cv(X, y, pipe, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Repeated stratified CV. Returns per-fold scores over all repeats.

    n_repeats > 1 is the point: it separates a real difference between models
    from the luck of one particular partition.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    rows = [s for tr, te in cv.split(X, y)
            if (s := _score_split(pipe, X, y, tr, te)) is not None]
    return pd.DataFrame(rows)


def grouped_cv(X, y, groups, pipe, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Stratified CV that keeps every clip from the same source video on the
    same side of the split. This is the honest estimate: at prediction time
    the model would face a video it has never seen."""
    rows = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + r)
        for tr, te in cv.split(X, y, groups=groups):
            s = _score_split(pipe, X, y, tr, te)
            if s is not None:
                rows.append(s)
    return pd.DataFrame(rows)


# -- Uncertainty ------------------------------------------------------

def summarize(scores: pd.DataFrame, prefix: str = '') -> dict:
    """Mean, std and a 95% interval over folds for each metric.

    The interval is the 2.5/97.5 percentile of the fold scores. Folds from a
    repeated CV are not independent, so this is a descriptive spread, not a
    formal confidence interval -- but it is the right order of magnitude and
    far more informative than a bare mean.
    """
    out = {}
    for col in scores.columns:
        v = scores[col].values
        out[f'{prefix}{col}_mean'] = round(float(np.mean(v)), 4)
        out[f'{prefix}{col}_std']  = round(float(np.std(v)), 4)
        out[f'{prefix}{col}_lo']   = round(float(np.percentile(v, 2.5)), 4)
        out[f'{prefix}{col}_hi']   = round(float(np.percentile(v, 97.5)), 4)
    return out


def beats_chance(scores: pd.DataFrame, metric: str = 'roc_auc',
                 chance: float = 0.5) -> bool:
    """True only if the lower end of the fold spread clears chance level.

    Replaces the old `beats_baseline = mean > baseline` flag, which marked
    PR-AUC 0.486 vs baseline 0.464 as a success on a model whose ROC-AUC
    was 0.52 -- i.e. noise.
    """
    return float(np.percentile(scores[metric].values, 2.5)) > chance


def _permute_within_groups(y, groups, rng):
    """Shuffle labels inside each group, leaving the group structure intact."""
    y_ = y.copy()
    for g in np.unique(groups):
        m = groups == g
        y_[m] = rng.permutation(y[m])
    return y_


def permutation_test(X, y, pipe, n_perm: int = 200, n_splits=N_SPLITS,
                     seed=SEED, metric: str = 'roc_auc', groups=None) -> dict:
    """Label-permutation test: how often does a shuffled-label run reach the
    observed score? Gives an empirical p-value that does not assume anything
    about the distribution of AUC under the null.

    `groups` is not optional in spirit. Shuffling labels globally and scoring
    with a plain StratifiedKFold builds a null that assumes clips are
    independent -- exactly the assumption the grouped CV exists to relax. When
    clips from one source video share camera pose, lighting and sometimes the
    kicker, that null is too easy to beat and the p-value comes out optimistic.
    Passing `groups` permutes labels WITHIN each video and evaluates with
    StratifiedGroupKFold, so both the observed score and the null are computed
    under the same, stricter, regime.
    """
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    grouped = groups is not None

    def one_run(y_):
        if grouped:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed)
            splits = cv.split(X, y_, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = cv.split(X, y_)
        s = [_score_split(pipe, X, y_, tr, te) for tr, te in splits]
        vals = [r[metric] for r in s if r is not None]
        return float(np.mean(vals)) if vals else np.nan

    observed = one_run(y)
    null = np.array([one_run(_permute_within_groups(y, groups, rng) if grouped
                             else rng.permutation(y))
                     for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    p = (np.sum(null >= observed) + 1) / (len(null) + 1)   # +1: never report p=0
    return {'observed': round(observed, 4),
            'null_mean': round(float(null.mean()), 4),
            'null_p95': round(float(np.percentile(null, 95)), 4),
            'p_value': round(float(p), 4),
            'n_perm': int(len(null)),
            'grouped': grouped}
