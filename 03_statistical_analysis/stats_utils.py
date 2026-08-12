"""
stats_utils.py
==============
Multiple-testing helpers shared by the statistical scripts.

Why
---
This project runs a lot of tests: 250 features x 3 targets in phase 1, and
20 frames x N features in every frame-by-frame timeline. At alpha = 0.05, 250
independent tests produce about 12.5 significant results by chance alone --
so "22 features are significant" is barely more than the noise floor, and
"10 features are significant" is BELOW it.

Two corrections are offered because they answer different questions:

  - Bonferroni controls the chance of ANY false positive. Very strict; the
    right choice when a single claimed finding has to hold up.
  - Benjamini-Hochberg controls the expected PROPORTION of false positives
    among the ones called significant. Less strict; the right choice for
    screening, e.g. deciding which frame a signal first appears at.

Reporting the uncorrected count on its own invites reading noise as signal.
"""

import numpy as np
import pandas as pd


def bonferroni(p_values, alpha: float = 0.05):
    """Family-wise threshold alpha/n. Returns (reject, threshold)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.nan
    thr = alpha / n
    return p < thr, thr


def benjamini_hochberg(p_values, alpha: float = 0.05):
    """Benjamini-Hochberg FDR control.

    Returns (reject, q_values). q_values are the BH-adjusted p-values, so a
    feature with q = 0.08 means: if we called everything down to this one
    significant, about 8% of those calls would be expected to be false.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([])

    order   = np.argsort(p)
    ranked  = p[order]
    ranks   = np.arange(1, n + 1)

    # Step-up adjustment, then enforce monotonicity from the largest p down
    q_sorted = np.minimum.accumulate((ranked * n / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.empty(n, dtype=float)
    q[order] = q_sorted
    return q < alpha, q


def expected_false_positives(n_tests: int, alpha: float = 0.05) -> float:
    """How many 'significant' results this many tests yield by chance alone.

    Print this next to the raw count. If the observed count is close to (or
    below) this number, there is nothing to explain.
    """
    return n_tests * alpha


def multiplicity_summary(p_values, alpha: float = 0.05, label: str = '') -> dict:
    """One-line verdict on a family of tests."""
    p = np.asarray(p_values, dtype=float)
    p = p[~np.isnan(p)]
    n = len(p)
    n_raw  = int((p < alpha).sum())
    rej_b, thr_b = bonferroni(p, alpha)
    rej_q, q     = benjamini_hochberg(p, alpha)
    exp_fp = expected_false_positives(n, alpha)

    out = {
        'label': label,
        'n_tests': n,
        'n_sig_uncorrected': n_raw,
        'expected_by_chance': round(exp_fp, 1),
        'n_sig_bonferroni': int(rej_b.sum()),
        'bonferroni_threshold': thr_b,
        'n_sig_fdr_bh': int(rej_q.sum()),
        'min_q_value': round(float(q.min()), 4) if n else np.nan,
        'excess_over_chance': round(n_raw - exp_fp, 1),
    }
    return out


def print_multiplicity_summary(summary: dict):
    s = summary
    head = f"  Multiple testing{' -- ' + s['label'] if s['label'] else ''}:"
    print(head)
    print(f"    {s['n_tests']} tests  |  p<0.05 uncorrected: {s['n_sig_uncorrected']}  "
          f"|  expected by chance: {s['expected_by_chance']}  "
          f"(excess {s['excess_over_chance']:+})")
    print(f"    surviving Bonferroni (p < {s['bonferroni_threshold']:.2g}): "
          f"{s['n_sig_bonferroni']}  |  surviving BH-FDR: {s['n_sig_fdr_bh']}  "
          f"|  smallest q-value: {s['min_q_value']}")
    if s['n_sig_bonferroni'] == 0 and s['n_sig_fdr_bh'] == 0:
        print("    -> nothing survives correction: no individual feature is an "
              "established finding here.")

# =====================================================================
# Feature selection -- the single definition of "is this a feature?"
# =====================================================================
#
# This used to live in six separate copies, one per script in this folder,
# and they drifted: when the signed lateral features were excluded, five
# scripts kept testing them. It is defined here once and imported.
#
# NOTE ON THE DUPLICATE FILE. 04_models/ carries a byte-identical copy of
# this module (apart from its header line) so that each stage runs
# standalone without a package install or sys.path juggling. The two are
# kept in sync by tests/test_analysis.py, which fails if they diverge.
# "Defined once per stage folder, with a test enforcing agreement" is the
# accurate description -- not "defined once".

EXCLUDE = {
    # identifiers and metadata
    'clip_id', 'video_name', 'event_id', 'camera_angle',
    # targets and anything derived from a target
    'target_zone', 'zone_name', 'macro_zone', 'gk_guessed',
    'gk_guessed_label', 'outcome', 'foot_used', 'foot_used_label',
    'foot_right', 'kicking_leg', 'shot_type', 'y',
    # counts and categorical codes
    'n_frames',
    'support_foot_lateral_zone', 'support_foot_depth_zone',
    'support_foot_quadrant', 'proximal_distal_valid',
    'peak_hip_frame', 'peak_knee_frame', 'peak_ankle_frame',
}

# Substrings that disqualify a column from being used as a feature.
EXCLUDE_SUBSTRINGS = (
    'visibility',              # MediaPipe detection confidence, not biomechanics
    'lateral_trunk_lean',      # signed along image x -- see note below
    'support_foot_x',          # idem
    'deception_lateral',       # derived from lateral_trunk_lean
    'support_foot_angle_deg',  # = arctan2(y, x) -- reintroduces the sign of x
    'deception_score',         # mean that still includes the lateral component
)

# Why the lateral family is excluded
# ----------------------------------
# These are signed left/right quantities measured along the image x axis, and
# the dataset mixes four camera positions:
#
#   front vs back  -- 180 degrees apart, so the sign inverts between them
#   left/right     -- side views, where the goal's lateral axis points into
#                     the screen; image x encodes the run-up direction instead
#
# Mirroring fixes the first case but not the second, and side views are more
# than half the dataset.
#
# Two derived columns defeated that exclusion until they were added above:
#
#   support_foot_angle_deg = degrees(arctan2(support_foot_y, support_foot_x))
#       A bijective re-encoding of support_foot_x in polar coordinates: the
#       sign of x survives intact as "near 0 deg" vs "near +/-180 deg".
#   deception_score
#       Documented as the mean of deception_torsion and deception_trunk, but
#       build_clips_master.py still averages in the lateral component. Both
#       clean components remain available as their own columns, so excluding
#       the composite loses no information.
#
# support_foot_distance = sqrt(x**2 + y**2) is NOT excluded: it is a magnitude
# and does not carry the sign, so the camera-inversion argument does not apply
# to it.


def feature_columns(df, verbose: bool = True) -> list:
    """Numeric feature columns, with metadata and non-comparable columns removed."""
    feats, dropped = [], []
    for c in df.columns:
        if c in EXCLUDE or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if any(s in c for s in EXCLUDE_SUBSTRINGS):
            dropped.append(c)
            continue
        feats.append(c)
    if verbose:
        print(f"  {len(feats)} features "
              f"({len(dropped)} detection-quality / camera-dependent columns excluded)")
    return feats


# =====================================================================
# Natural vs Crossed -- the single definition of the label
# =====================================================================
#
# CONVENTION (fixed 2026-08; three scripts previously disagreed):
#
#   Natural : right foot -> Right   OR   left foot -> Left
#   Crossed : right foot -> Left    OR   left foot -> Right
#
# Left/Right are zones of the goal AS SEEN BY THE KICKER facing it, which is
# the orientation of the annotation grid in 01_extraction/clips_creator.py.
# So a "crossed" kick is one where the leg swings across the body midline: a
# right-footed player striking with the inside of the foot sends the ball to
# his own left.
#
# The label is symmetric in the foot by construction, which is the whole point
# of the target: it is the same body-relative action whichever foot performs it.
#
# WARNING. This flips the class names relative to the version of
# 04_models/reg_cross_analysis.py shipped before 2026-08, which had
# Natural = right foot -> Left. ROC-AUC, accuracy and Mann-Whitney p-values are
# invariant to swapping two class names, so no metric changes -- but every
# sentence of the form "when X is higher the kick tends to be CROSSED" inverts,
# and the class counts swap (154 Natural / 202 Crossed, not the reverse).

NATURAL = 'Natural'
CROSSED = 'Crossed'


def shot_type_series(df):
    """Natural/Crossed label from foot_right + macro_zone. NaN where undefined.

    Returns a pandas Series aligned to df, with values 'Natural', 'Crossed'
    or None (central kicks, off-target kicks, missing foot).
    """
    fr   = pd.to_numeric(df.get('foot_right'), errors='coerce')
    zone = df.get('macro_zone').astype(str) if 'macro_zone' in df.columns else None
    if zone is None:
        raise KeyError("macro_zone is required to build the Natural/Crossed label")

    natural = ((fr == 1) & (zone == 'Right')) | ((fr == 0) & (zone == 'Left'))
    crossed = ((fr == 1) & (zone == 'Left'))  | ((fr == 0) & (zone == 'Right'))

    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    out[natural] = NATURAL
    out[crossed] = CROSSED
    return out


def build_shot_type(df):
    """Subset of df with a defined Natural/Crossed label, plus the columns
    'shot_type' (str) and 'y' (1 = Crossed), index reset."""
    sub = df.copy()
    sub['shot_type'] = shot_type_series(sub)
    sub = sub[sub['shot_type'].notna()].reset_index(drop=True)
    sub['y'] = (sub['shot_type'] == CROSSED).astype(int)
    return sub


# =====================================================================
# Small guards
# =====================================================================

def is_empty(df, label: str = '') -> bool:
    """True (and prints why) when a results table came back empty.

    Every script in this project builds a DataFrame of per-feature test rows
    and then sorts it by 'p_value'. When a group falls below the minimum of 5
    observations no rows are produced, and the sort raised KeyError instead of
    saying so. Call this before sorting.
    """
    if df is None or len(df) == 0:
        print(f"  No test had enough observations per group"
              f"{' -- ' + label if label else ''}; section skipped.")
        return True
    return False
