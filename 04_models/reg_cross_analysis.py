"""
reg_cross_analysis.py
=====================
Final analysis: Natural vs Crossed.

  Natural : right foot -> Right   OR   left foot -> Left
  Crossed : right foot -> Left    OR   left foot -> Right

Left/Right are goal zones as seen by the kicker facing the goal, so a Crossed
kick is one where the leg swings across the body midline. The label is defined
once, in stats_utils.build_shot_type(), and imported here -- this file used to
carry the opposite convention from reg_cros.py and practical_insights.py.

Hypothesis: classifying the TYPE of movement (natural/crossed) may have a
stronger signal than classifying the absolute DIRECTION (P2), because the
body's crossing motion has a more consistent biomechanical signature (hip
rotation, torso torsion) than "left vs right" on its own, which depends on
the foot used.

Steps:
  1. Build the Natural/Crossed label from foot_right + macro_zone
  2. LR, RF, SVM, XGBoost -- repeated CV + CV grouped by source video
  2b. Phase-restricted models (early / early+mid / all) -- the anticipation test
  2c. Per-foot models -- is the signal body-relative or an artefact of the XOR?
  2d. Grouped label-permutation test on the best model
  3. Decision Tree (depth=3, min_leaf=15) with out-of-fold rule validation
  4. Frame-by-frame timeline (Mann-Whitney U) with Benjamini-Hochberg FDR
  5. Practical report in non-technical language

Corrections relative to the earlier version
-------------------------------------------
  1. The file could not be run at all: it read clips_master.csv with
     sep=";" decimal="," while the file that the rest of the pipeline
     produces is comma-separated. Now uses the pandas default, like every
     other script.

  2. The model was compared against the goalkeeper's 48.9% without
     mentioning that always predicting "Natural" already scores 56.7% on this
     class split. The report now states the majority-class baseline first and
     measures the model against it; the GK comparison is kept but framed
     against the same reference.

  3. The tree rule was reported with its in-sample leaf purity ("94%
     confidence"). It is now also applied out of fold, and the report leads
     with that number.

  4. The frame-by-frame timeline ran 20 frames x N features of Mann-Whitney
     tests at p<0.05 with no correction, so "the signal becomes reliable at
     frame X" was partly reading noise. Benjamini-Hochberg FDR is now applied
     across all the tests in the timeline and the corrected column is the one
     used for the "first reliable frame".

  5. The timeline tested the same base variable more than once. Features are
     mapped to their base variable before testing (late_max_kick_knee_angle and
     contact_kick_knee_angle both map to kick_knee_angle), so two selected
     features sharing a base produced two identical rows with identical
     p-values. BH over a family containing exact duplicates is not valid, and
     the surviving count was inflated. Base variables are now deduplicated.

  6. The class names were the opposite of the ones used by reg_cros.py and
     practical_insights.py. Metrics are invariant to the swap, but every
     interpretive sentence and every class count inverted between scripts. The
     label now comes from stats_utils.

  7. There was no permutation test on this target at all, and no phase-
     restricted or per-foot model. Steps 2b-2d add them.

Reads:  clips_master.csv, biomechanics_dataset_complete.csv
Output:
  - natural_crossed_results.csv
  - natural_crossed_timeline.csv
  - natural_crossed_by_phase.csv     (step 2b)
  - natural_crossed_by_foot.csv      (step 2c)
  - natural_crossed_practical.txt

Usage:
  python reg_cross_analysis.py
  python reg_cross_analysis.py --master clips_master.csv --frames biomechanics_dataset_complete.csv
"""

import argparse
import re
import warnings
# Silence the noisy-but-harmless deprecation churn only. Convergence failures,
# degenerate folds and constant-column warnings are exactly the diagnostics
# that matter at this sample size, so they are left switched on.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold

from eval_utils import (feature_columns, repeated_cv, grouped_cv,
                        summarize, beats_chance, permutation_test)
from stats_utils import benjamini_hochberg, build_shot_type, is_empty

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RESULTS_PATH  = "natural_crossed_results.csv"
TIMELINE_PATH = "natural_crossed_timeline.csv"
PHASE_PATH    = "natural_crossed_by_phase.csv"
PERFOOT_PATH  = "natural_crossed_by_foot.csv"
REPORT_PATH   = "natural_crossed_practical.txt"

CONTACT_FRAME = 19   # last frame of the clip (0-19) = moment of contact with the ball
MS_PER_FRAME = 40    # nominal 25fps; the timeline uses the value read from the data
DEFAULT_FPS  = 25.0  # the whole dataset is nominally 25fps -- verified per clip below
K_FEATS = 40
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TOP_N_TIMELINE = 5
REACTION_TIME_MS = 200   # rough floor for a goalkeeper to initiate a dive

# Feature selection and the EXCLUDE list live in eval_utils.py so that every
# script in this folder uses the same definition of "is a biomechanical
# feature" -- including the exclusion of mean_visibility_score, which is a
# pose-detection quality metric and not a property of the kicker.

# Translation of the base variables (biomechanics_dataset_complete.csv) into plain language,
# used to compose the sentences in the practical report.
BASE_VAR_DESC = {
    "lateral_trunk_lean": "the lateral trunk lean",
    "trunk_inclination": "the forward trunk inclination",
    # NOT axial torsion. This is the unsigned angle between the shoulder line
    # and the hip line in the IMAGE PLANE. True axial rotation shows up in a
    # projection as a shortening of the shoulder line, not as an angle between
    # the two lines, and when the trunk turns past the frontal plane the
    # projected shoulder vector inverts and the value jumps towards 180 deg. In
    # practice it tracks orientation relative to the camera. Described here as
    # what it measures, not as what it was named.
    "torso_torsion_angle": "the angle between the shoulder line and the hip line in the image",
    "kick_hip_angle": "the hip angle of the kicking leg",
    "kick_hip_angular_vel": "the rotation speed of the kicking-leg hip",
    "kick_knee_angle": "the knee angle of the kicking leg",
    "supp_knee_angle": "the knee angle of the support leg",
    "kick_ankle_angle": "the ankle angle of the kicking leg",
    "kick_ankle_angular_vel": "the rotation speed of the kicking ankle",
    "supp_ankle_angle": "the ankle angle of the support leg",
    "kick_angular_vel": "the angular velocity of the kicking motion",
    "kick_angular_accel": "the angular acceleration of the kicking motion",
    "kick_foot_speed_ms": "the foot speed at the moment of the kick",
    "running_speed_kmh": "the approach running speed",
    "left_elbow_angle": "the left elbow angle",
    "right_elbow_angle": "the right elbow angle",
    "mean_visibility_score": "the video detection quality",
    # These are NOT positions relative to the ball. The reference point is the
    # kicking ankle at frame 0 -- roughly 800 ms before contact, mid-run-up --
    # so what they measure is the extent of the approach, not where the support
    # foot was planted next to the ball. They are also measured inside the
    # tracking crop. support_foot_x is excluded from the models; support_foot_y
    # is retained.
    "support_foot_x": "the horizontal offset between the support foot and the kicking ankle at the start of the clip",
    "support_foot_y": "the vertical offset between the support foot and the kicking ankle at the start of the clip",
}

PREFIXES = ["contact_", "early_std_", "early_mean_", "early_max_", "early_min_",
            "mid_std_", "mid_mean_", "mid_max_", "mid_min_",
            "late_std_", "late_mean_", "late_max_", "late_min_", "delta_"]

PHASE_LABELS = {"early": "initial", "mid": "intermediate", "late": "final"}
STAT_LABELS = {"mean": "average", "max": "maximum value", "min": "minimum value", "std": "variability"}


def base_variable(feature_name: str) -> str:
    """Maps a tabular feature (e.g. contact_lateral_trunk_lean,
    late_mean_kick_hip_angle) to the base frame-by-frame variable
    (e.g. lateral_trunk_lean, kick_hip_angle)."""
    for p in PREFIXES:
        if feature_name.startswith(p):
            return feature_name[len(p):]
    return feature_name


# -- Step 1: label ------------------------------------------------

def build_label(df: pd.DataFrame) -> pd.DataFrame:
    """Natural/Crossed label. Thin wrapper over the shared definition.

    The rule itself lives in stats_utils.build_shot_type() so that this script,
    reg_cros.py and practical_insights.py cannot drift apart again -- they did,
    and for a while this file called "Natural" what the other two called
    "Crossed". y = 1 means Crossed.
    """
    return build_shot_type(df)


def _ms_per_frame(seq: pd.DataFrame) -> float:
    """Milliseconds per frame, read from the data instead of assumed.

    Every "N ms before contact" figure in this project rests on 25fps. Nothing
    in the pipeline enforces it, and broadcast footage is not always 25fps, so
    the value is derived from time_sec when that column is present and the
    assumption is checked out loud.
    """
    if "time_sec" not in seq.columns or "frame" not in seq.columns:
        return 1000.0 / DEFAULT_FPS
    s = seq.sort_values(["clip_id", "frame"])
    d = s.groupby("clip_id")["time_sec"].diff()
    d = d[d > 0]
    if d.empty:
        return 1000.0 / DEFAULT_FPS

    per_clip = (1.0 / d.groupby(s.loc[d.index, "clip_id"]).median()).round(1)
    counts   = per_clip.value_counts().sort_index(ascending=False)
    ms       = float(d.median() * 1000.0)

    print(f"  Frame rate across clips: "
          + ", ".join(f"{int(n)} clip(s) at {f:g}fps" for f, n in counts.items()))
    off = int((per_clip.round(0) != round(DEFAULT_FPS)).sum())
    if off:
        # This is not a formality. The clip window is defined in FRAMES (20),
        # not in seconds, so a clip at 30fps covers 633 ms of run-up and one at
        # 60fps covers 317 ms, against 760 ms at 25fps. Consequences:
        #   - "N ms before contact" is only exact for the 25fps clips;
        #   - the early/mid/late phases span different real durations per clip,
        #     so "the approach run" is not the same interval in every row.
        # Angular velocities are unaffected: kinematics.py uses the real fps for
        # dt. The timing claims are what suffer.
        print(f"  [WARNING] {off} clip(s) are not {DEFAULT_FPS:g}fps. The 20-frame "
              f"window is defined in frames, so those clips cover a shorter span "
              f"of run-up, and every 'ms before contact' figure below is exact "
              f"only for the {DEFAULT_FPS:g}fps clips. Median used: {ms:.1f} ms/frame.")
    return ms


# -- Pipelines (same methodology as tabular_models.py) ------------

def build_models():
    models = {
        "Logistic Regression": Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("fallback", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=K_FEATS)),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("fallback", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=K_FEATS)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                           class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("fallback", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=K_FEATS)),
            ("clf", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42)),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("fallback", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=K_FEATS)),
            ("clf", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)),
        ])
    return models


# -- Step 4: timeline -----------------------------------------------

def build_timeline(seq: pd.DataFrame, sub: pd.DataFrame, top_features: list,
                   ms_per_frame: float = MS_PER_FRAME) -> pd.DataFrame:
    """Frame-by-frame Mann-Whitney U between Natural and Crossed.

    Benjamini-Hochberg FDR is applied across every test in the table at once
    (base variables x frames). `significant_fdr` is the column to trust: the
    uncorrected one is kept only for comparison, and with 20 frames it produces
    several hits per run by chance alone.

    Base variables are DEDUPLICATED first. The per-clip features map many-to-one
    onto the frame-level variables (contact_kick_knee_angle and
    late_max_kick_knee_angle both map to kick_knee_angle), so a top-N list that
    contains two features sharing a base used to generate the same test twice,
    with the same p-value. That inflates the family size, double-counts every
    surviving hit, and breaks the independence BH assumes.
    """
    seq = seq.merge(sub[["clip_id", "shot_type"]], on="clip_id", how="inner")

    bases = list(dict.fromkeys(base_variable(f) for f in top_features))
    dropped = len(top_features) - len(bases)
    if dropped:
        print(f"  {len(top_features)} selected features -> {len(bases)} distinct "
              f"base variables ({dropped} duplicate test(s) removed before FDR)")

    rows = []
    for base in bases:
        if base not in seq.columns:
            continue
        for frame in range(20):
            fr = seq[seq["frame"] == frame]
            nat = fr.loc[fr["shot_type"] == "Natural", base].dropna()
            cru = fr.loc[fr["shot_type"] == "Crossed", base].dropna()
            if len(nat) < 5 or len(cru) < 5:
                continue
            try:
                u_stat, p_val = mannwhitneyu(nat, cru, alternative="two-sided")
            except ValueError:
                continue
            rows.append({
                "base_variable": base,
                "frame": frame,
                "ms_before_contact": (CONTACT_FRAME - frame) * ms_per_frame,
                "u_stat": u_stat,
                "p_value": p_val,
                "significant_p05": p_val < 0.05,
                "mean_natural": nat.mean(),
                "mean_crossed": cru.mean(),
            })

    tl = pd.DataFrame(rows)
    if tl.empty:
        return tl

    reject, q = benjamini_hochberg(tl["p_value"].values, alpha=0.05)
    tl["q_value"] = np.round(q, 4)
    tl["significant_fdr"] = reject

    n_raw, n_fdr = int(tl["significant_p05"].sum()), int(reject.sum())
    print(f"  {len(tl)} distinct frame-level tests  |  p<0.05 uncorrected: {n_raw} "
          f"(expected by chance: {len(tl) * 0.05:.1f})  |  surviving BH-FDR: {n_fdr}")
    return tl


# -- Decision Tree in plain language -----------------------------

def describe_feature(feat: str) -> str:
    """Translates a tabular feature into plain language, preserving the
    temporal context of the prefix (at contact / change / initial-mid-final phase)."""
    base = base_variable(feat)
    phrase = BASE_VAR_DESC.get(base, f"the {base.replace('_', ' ')}")

    if feat.startswith("contact_"):
        return f"{phrase} at the moment of contact with the ball"
    if feat.startswith("delta_"):
        return f"the change in {phrase} over the course of the kick"
    for phase_key, phase_label in PHASE_LABELS.items():
        for stat_key, stat_label in STAT_LABELS.items():
            if feat.startswith(f"{phase_key}_{stat_key}_"):
                if stat_key == "std":
                    return f"the variability of {phrase} in the {phase_label} phase of the approach"
                return f"the {stat_label} of {phrase} in the {phase_label} phase of the approach"
    return phrase


def dt_path_to_rule(tree: DecisionTreeClassifier, feature_names: list, target_class: int = 1):
    """Walks the tree and returns the path to the purest leaf for the target
    class with at least 10 samples, as a list of (feature, operator, threshold)."""
    tree_ = tree.tree_
    best_leaf, best_purity, best_path = None, -1, None

    def recurse(node, path):
        nonlocal best_leaf, best_purity, best_path  # accumulator shared across recursive calls
        if tree_.feature[node] == -2:  # leaf
            values = tree_.value[node][0]           # weighted count per class (class_weight) -- not normalized
            n = tree_.n_node_samples[node]           # real sample count in the leaf
            purity = values[target_class] / values.sum()   # real proportion of the target class in the leaf
            if n >= 10 and purity > best_purity:
                best_purity, best_leaf, best_path = purity, node, list(path)
            return
        feat = feature_names[tree_.feature[node]]
        thr = tree_.threshold[node]
        recurse(tree_.children_left[node], path + [(feat, "<=", thr)])
        recurse(tree_.children_right[node], path + [(feat, ">", thr)])

    recurse(0, [])
    return best_path, best_purity


# -- Out-of-fold validation of the extracted rule --------------------

def out_of_fold_rule(X, y, target_class=1, n_splits=5, n_repeats=5, seed=42):
    """Build the tree on the training part, then apply its own rule to clips
    it has never seen.

    Leaf purity from a tree fit on all the data answers "how cleanly did the
    tree carve up what it was shown?". This answers "how often is the rule
    right on a new penalty?" -- which is what a practical report is claiming.
    """
    precisions, coverages = [], []
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(X, y):
            pipe = Pipeline([
                ("imp",   KNNImputer(n_neighbors=5)),
                ("fallback", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
                ("sel",   SelectKBest(f_classif, k=min(30, X.shape[1]))),
                ("clf",   DecisionTreeClassifier(max_depth=3, min_samples_leaf=15,
                                                 class_weight="balanced",
                                                 random_state=42)),
            ])
            pipe.fit(X.iloc[tr], y[tr])
            kept = [X.columns[i] for i in pipe["sel"].get_support(indices=True)]
            path, _ = dt_path_to_rule(pipe["clf"], kept, target_class=target_class)
            if not path:
                continue

            # The tree split on standardized values -> push the test rows
            # through the same imputer + scaler before applying the rule.
            Z = pipe["scale"].transform(
                pipe["fallback"].transform(pipe["imp"].transform(X.iloc[te])))
            Z = pd.DataFrame(Z[:, pipe["sel"].get_support(indices=True)], columns=kept)

            mask = np.ones(len(Z), dtype=bool)
            for feat, op, thr in path:
                col = Z[feat].values
                mask &= (col <= thr) if op == "<=" else (col > thr)
            if mask.sum() == 0:
                continue
            precisions.append(float((y[te][mask] == target_class).mean()))
            coverages.append(float(mask.mean()))

    if not precisions:
        return None
    return {
        "precision": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "coverage": float(np.mean(coverages)),
        "n_folds": len(precisions),
    }


# -- Main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Natural vs Crossed analysis: models + timeline + practical report.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--master', default='clips_master.csv', help='Path to clips_master.csv')
    ap.add_argument('--frames', default='biomechanics_dataset_complete.csv',
                    help='Path to biomechanics_dataset_complete.csv')
    ap.add_argument('--no-perm', action='store_true',
                    help='skip the label-permutation test (slow)')
    ap.add_argument('--n-perm', type=int, default=200,
                    help='number of label shuffles (default 200)')
    args = ap.parse_args()

    print(f"Loading {args.master} ...")
    # Comma-separated, like every other file the pipeline produces. The old
    # sep=";" decimal="," made this script crash on its own input.
    df  = pd.read_csv(args.master)
    sub = build_label(df)

    n_nat = int((sub["shot_type"] == "Natural").sum())
    n_cru = int((sub["shot_type"] == "Crossed").sum())
    y     = sub["y"].values

    pr_baseline  = y.mean()                       # PR-AUC chance level
    maj_baseline = max(y.mean(), 1 - y.mean())    # always predict the majority class
    maj_label    = "Crossed" if y.mean() > 0.5 else "Natural"
    gk_accuracy  = sub["gk_guessed"].mean()

    print(f"  n={len(sub)}  |  Natural={n_nat}  Crossed={n_cru}")
    print(f"  Chance ROC-AUC 0.500  |  chance PR-AUC {pr_baseline:.3f}")
    print(f"  Majority-class accuracy (always '{maj_label}'): {maj_baseline:.3f}"
          f"  <- the number any model has to beat")
    print(f"  GK's real-time accuracy on these kicks: {gk_accuracy:.3f}")

    features = feature_columns(df)
    X = sub[[f for f in features if f in sub.columns]].copy()
    groups = sub["video_name"].values if "video_name" in sub.columns else None

    # -- Step 2 --------------------------------------------------
    print(f"\n{'='*68}\nStep 2 -- LR, RF, SVM, XGBoost (repeated + grouped CV)\n{'='*68}")
    results = []
    for name, pipe in build_models().items():
        pipe.set_params(sel__k=min(K_FEATS, X.shape[1]))
        scores = repeated_cv(X, y, pipe)
        row = {"model": name}
        row.update(summarize(scores))
        row["pr_auc_baseline"]   = round(float(pr_baseline), 4)
        row["majority_accuracy"] = round(float(maj_baseline), 4)
        row["gk_accuracy"]       = round(float(gk_accuracy), 4)
        row["roc_beats_chance"]  = beats_chance(scores)
        row["acc_over_majority"] = round(row["accuracy_mean"] - maj_baseline, 4)
        if groups is not None:
            row.update(summarize(grouped_cv(X, y, groups, pipe), prefix="grouped_"))
        results.append(row)
        print(f"  {name:22s}  ROC-AUC={row['roc_auc_mean']:.3f} "
              f"[{row['roc_auc_lo']:.3f},{row['roc_auc_hi']:.3f}]  "
              f"Acc={row['accuracy_mean']:.3f} "
              f"({row['acc_over_majority']:+.3f} vs majority class)")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[OK] {RESULTS_PATH}")

    # Best model by ROC-AUC, not PR-AUC: on this class split PR-AUC drifts
    # above its own baseline under fold noise.
    best_row        = results_df.loc[results_df["roc_auc_mean"].idxmax()]
    best_model_name = best_row["model"]
    best_acc        = float(best_row["accuracy_mean"])
    print(f"\n  Best model: {best_model_name}  "
          f"(ROC-AUC={best_row['roc_auc_mean']:.3f}, accuracy={best_acc:.3f})")
    if groups is not None:
        # The fold range, not the bare mean. This project's own rule is that a
        # range including 0.500 is not a result; reporting the grouped estimate
        # as a single number exempted it from that rule. Note also that the
        # grouped figure coming out ABOVE the random-split one is not evidence
        # of robustness -- with 28 groups, one of them holding 80 clips, the
        # grouped partitions are very uneven and the spread is wide. The claim
        # the number supports is "grouping does not degrade it", nothing more.
        print(f"  Same model grouped by source video: "
              f"ROC-AUC={best_row['grouped_roc_auc_mean']:.3f} "
              f"[{best_row['grouped_roc_auc_lo']:.3f}, "
              f"{best_row['grouped_roc_auc_hi']:.3f}]")

    best_pipe = build_models()[best_model_name]

    phase_rows, foot_rows = [], []

    # -- Step 2b: phase-restricted models -------------------------
    #
    # THE ANTICIPATION TEST. Every frame-level signal that survives FDR
    # correction sits at frame 18-19, i.e. 0-40 ms before contact, and the
    # per-clip features that survive Bonferroni are measured at the contact
    # frame. The simplest explanation consistent with that is that the model is
    # reading how the foot struck the ball, at the moment it struck it -- which
    # is close to tautological, since Natural/Crossed IS the striking technique.
    #
    # Restricting the feature set to the early and mid phases (frames 0-13,
    # >=240 ms before contact) separates the two readings. At chance, the
    # finding is "the contact frame shows the strike". Above chance, the
    # project's own "no usable anticipation" conclusion is understated.
    print(f"\n{'='*68}\nStep 2b -- Phase-restricted models ({best_model_name})\n{'='*68}")
    for label, prefixes in [("early only",      ("early_",)),
                            ("early + mid",     ("early_", "mid_")),
                            ("all (reference)", None)]:
        cols = ([c for c in X.columns if c.startswith(prefixes)] if prefixes
                else list(X.columns))
        if len(cols) < 2:
            print(f"  {label:18s} -- only {len(cols)} feature(s); skipped")
            continue
        pipe = clone(best_pipe).set_params(sel__k=min(K_FEATS, len(cols)))
        s = repeated_cv(X[cols], y, pipe)
        r = summarize(s)
        earliest_ms = (CONTACT_FRAME - 13) * MS_PER_FRAME if prefixes else 0
        phase_rows.append({"phase": label, "n_features": len(cols),
                           "earliest_ms_before_contact": earliest_ms, **r,
                           "roc_beats_chance": beats_chance(s)})
        print(f"  {label:18s} {len(cols):3d} feats  "
              f"ROC-AUC {r['roc_auc_mean']:.3f} "
              f"[{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}]  "
              f"{'[OK]' if beats_chance(s) else '--'}")
    if phase_rows:
        pd.DataFrame(phase_rows).to_csv(PHASE_PATH, index=False)
        print(f"[OK] {PHASE_PATH}")
        em = next((r for r in phase_rows if r["phase"] == "early + mid"), None)
        if em is not None:
            if em["roc_beats_chance"]:
                print("  -> the signal is present before the strike: the fold range "
                      "of the early+mid model clears chance.")
            else:
                print("  -> no signal before the strike: the early+mid fold range "
                      "includes 0.500, so what the full model reads is the "
                      "contact frame itself.")

    # -- Step 2c: per-foot models ---------------------------------
    #
    # The label is an XOR of the kicking foot with the target side, so a
    # sceptic's first question is whether the model is recovering the foot and
    # exploiting the XOR rather than measuring a body-relative action. Within a
    # single foot the XOR is constant, so if the signal is what the project
    # claims it is, it has to survive here.
    print(f"\n{'='*68}\nStep 2c -- Per-foot models ({best_model_name})\n{'='*68}")
    if "foot_right" in sub.columns:
        for label, mask in [("right-footed", sub["foot_right"] == 1),
                            ("left-footed",  sub["foot_right"] == 0)]:
            m = mask.values
            if m.sum() < 40 or len(np.unique(y[m])) < 2:
                print(f"  {label:14s} n={int(m.sum()):3d} -- too few clips; skipped")
                continue
            pipe = clone(best_pipe).set_params(sel__k=min(K_FEATS, X.shape[1]))
            s = repeated_cv(X[m], y[m], pipe)
            r = summarize(s)
            foot_rows.append({"subset": label, "n": int(m.sum()), **r,
                              "roc_beats_chance": beats_chance(s)})
            print(f"  {label:14s} n={int(m.sum()):3d}  "
                  f"ROC-AUC {r['roc_auc_mean']:.3f} "
                  f"[{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}]  "
                  f"{'[OK]' if beats_chance(s) else '--'}")
        if foot_rows:
            pd.DataFrame(foot_rows).to_csv(PERFOOT_PATH, index=False)
            print(f"[OK] {PERFOOT_PATH}")
            print("  (the left-footed subset is small; a null there is as likely to "
                  "be a power problem as a real asymmetry)")

    # -- Step 2d: grouped label-permutation test ------------------
    #
    # There was no permutation test on this target at all. It is also run
    # GROUPED: shuffling labels globally and scoring with a plain
    # StratifiedKFold builds a null that assumes clips are independent, which
    # is the assumption the grouped CV exists to relax.
    if not args.no_perm:
        print(f"\n{'='*68}\nStep 2d -- Label-permutation test ({best_model_name})\n{'='*68}")
        pipe = clone(best_pipe).set_params(sel__k=min(K_FEATS, X.shape[1]))
        perm = permutation_test(X, y, pipe, n_perm=args.n_perm, groups=groups)
        print(f"  {'grouped by source video' if perm['grouped'] else 'ungrouped'}, "
              f"{perm['n_perm']} shuffles")
        print(f"  observed ROC-AUC {perm['observed']:.3f}  |  null mean "
              f"{perm['null_mean']:.3f}, 95th pct {perm['null_p95']:.3f}  |  "
              f"p = {perm['p_value']:.4f}")
        if perm["p_value"] >= 0.05:
            print("  -> not distinguishable from chance under label permutation.")
        results_df.loc[results_df["model"] == best_model_name,
                       "permutation_p"] = perm["p_value"]
        results_df.loc[results_df["model"] == best_model_name,
                       "permutation_grouped"] = perm["grouped"]
        results_df.to_csv(RESULTS_PATH, index=False)

    # -- Step 3: Decision Tree, with out-of-fold rule validation ---
    print(f"\n{'='*68}\nStep 3 -- Decision Tree (depth=3, min_leaf=15)\n{'='*68}")
    tree_pipe = Pipeline([
        ("imp",   KNNImputer(n_neighbors=5)),
        ("fallback", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("sel",   SelectKBest(f_classif, k=min(30, X.shape[1]))),
        ("clf",   DecisionTreeClassifier(max_depth=3, min_samples_leaf=15,
                                         class_weight="balanced", random_state=42)),
    ])
    dt_scores = repeated_cv(X, y, tree_pipe)
    dt_summ   = summarize(dt_scores)
    print(f"  Decision Tree ROC-AUC: {dt_summ['roc_auc_mean']:.3f} "
          f"[{dt_summ['roc_auc_lo']:.3f}, {dt_summ['roc_auc_hi']:.3f}]")

    tree_pipe.fit(X, y)
    top30 = [X.columns[i] for i in tree_pipe["sel"].get_support(indices=True)]
    dt    = tree_pipe["clf"]
    print("  (thresholds below are on standardized units)")
    print(export_text(dt, feature_names=list(top30), max_depth=3))

    path, purity = dt_path_to_rule(dt, top30, target_class=1)
    oof = out_of_fold_rule(X, y, target_class=1)
    if path:
        print(f"  Purest leaf -- in-sample purity: {purity:.1%}")
        if oof:
            print(f"  Same rule out of fold: {oof['precision']:.1%} "
                  f"+/- {oof['precision_std']:.1%} precision on unseen clips "
                  f"(covering {oof['coverage']:.1%} of them)")
            print(f"  -> in-sample purity overstates the rule by "
                  f"{purity - oof['precision']:+.1%}")

    # -- Step 4: timeline ------------------------------------------
    print(f"\n{'='*68}\nStep 4 -- Frame-by-frame timeline (Mann-Whitney U + BH-FDR)\n{'='*68}")
    sel_scores = pd.Series(
        tree_pipe["sel"].scores_[tree_pipe["sel"].get_support(indices=True)],
        index=top30)
    top5 = sel_scores.sort_values(ascending=False).head(TOP_N_TIMELINE).index.tolist()
    print(f"  Top {TOP_N_TIMELINE} features (f_classif): {top5}")

    seq = pd.read_csv(args.frames)
    ms_frame = _ms_per_frame(seq)
    timeline = build_timeline(seq, sub, top5, ms_per_frame=ms_frame)

    # The timeline is keyed by base variable, not by tabular feature: several
    # features map onto the same frame-level variable and testing it once per
    # feature duplicated rows inside the FDR family.
    first_sig = {}
    if is_empty(timeline, "frame-level timeline"):
        timeline = pd.DataFrame(columns=["base_variable", "frame", "p_value",
                                         "q_value", "significant_fdr",
                                         "mean_natural", "mean_crossed"])
    else:
        timeline.to_csv(TIMELINE_PATH, index=False)
        print(f"[OK] {TIMELINE_PATH}")

        # "First reliable frame" uses the FDR-corrected column, not the raw one.
        for base in timeline["base_variable"].unique():
            hits = timeline[(timeline["base_variable"] == base)
                            & (timeline["significant_fdr"])]
            first_sig[base] = int(hits["frame"].min()) if not hits.empty else None
            fr = first_sig[base]
            if fr is not None:
                print(f"    {base:40s}  FDR-significant from frame {fr} "
                      f"(~{(CONTACT_FRAME - fr) * ms_frame:.0f}ms before contact)")
            else:
                print(f"    {base:40s}  never survives FDR at any single frame")

    # -- Step 5: practical report ---------------------------------
    print(f"\n{'='*68}\nStep 5 -- Practical report\n{'='*68}")
    lines = []
    lines.append("NATURAL VS CROSSED ANALYSIS -- PRACTICAL REPORT")
    lines.append("=" * 68)
    lines.append("")
    lines.append(f"Sample: {len(sub)} clips ({n_nat} Natural, {n_cru} Crossed)")
    lines.append("")
    lines.append("1. HOW GOOD IS THE MODEL, REALLY")
    lines.append("-" * 68)
    lines.append(f"Best model: {best_model_name}")
    lines.append(f"  ROC-AUC (repeated 5-fold CV): {best_row['roc_auc_mean']:.3f} "
                 f"[95% fold range {best_row['roc_auc_lo']:.3f}, {best_row['roc_auc_hi']:.3f}]"
                 f"  -- chance is 0.500")
    if groups is not None:
        lines.append(f"  ROC-AUC grouped by source video: "
                     f"{best_row['grouped_roc_auc_mean']:.3f} "
                     f"[95% fold range {best_row['grouped_roc_auc_lo']:.3f}, "
                     f"{best_row['grouped_roc_auc_hi']:.3f}]")
    lines.append(f"  Accuracy: {best_acc*100:.1f}%")
    lines.append("")
    lines.append(f"  Reference points, in the order that matters:")
    lines.append(f"    - Always predicting '{maj_label}': {maj_baseline*100:.1f}%  "
                 f"<- the real baseline")
    lines.append(f"    - This model:                      {best_acc*100:.1f}%  "
                 f"({(best_acc - maj_baseline)*100:+.1f} pp over that baseline)")
    lines.append(f"    - GK's real-time guess:            {gk_accuracy*100:.1f}%")
    lines.append("")
    lines.append(f"  The model is {(best_acc - gk_accuracy)*100:+.1f} pp relative to the "
                 f"goalkeeper, but that comparison is not the headline it looks like:")
    lines.append(f"  a constant prediction of '{maj_label}' already scores "
                 f"{(maj_baseline - gk_accuracy)*100:+.1f} pp against the GK without "
                 f"looking at the kicker at all.")
    lines.append(f"  The honest statement of what the biomechanics add is "
                 f"{(best_acc - maj_baseline)*100:+.1f} pp over the majority class.")
    if not bool(best_row["roc_beats_chance"]):
        lines.append("")
        lines.append("  CAVEAT: the 95% fold range of the ROC-AUC includes 0.500, so on "
                     "this sample the result is not separable from chance.")
    lines.append("")
    lines.append("1b. IS THE SIGNAL THERE BEFORE THE STRIKE?")
    lines.append("-" * 68)
    if phase_rows:
        for r in phase_rows:
            note = ""
            if r["phase"] != "all (reference)":
                note = f"  (nothing after ~{r['earliest_ms_before_contact']:.0f}ms before contact)"
            lines.append(f"  {r['phase']:18s} {r['n_features']:3d} feats  "
                         f"ROC-AUC {r['roc_auc_mean']:.3f} "
                         f"[{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}]"
                         f"{'  <- clears chance' if r['roc_beats_chance'] else ''}{note}")
        em = next((r for r in phase_rows if r["phase"] == "early + mid"), None)
        if em is not None:
            lines.append("")
            if em["roc_beats_chance"]:
                lines.append("  The early+mid model uses no information from the last six "
                             "frames, and still clears chance. Whatever separates the two "
                             "kick types is visible before the strike.")
            else:
                lines.append("  The early+mid model uses no information from the last six "
                             "frames, and its fold range includes 0.500. On this dataset "
                             "the separation comes from the contact frame itself -- the "
                             "model is reading how the foot struck the ball, not "
                             "anticipating it.")
    else:
        lines.append("  Not computed.")
    lines.append("")
    lines.append("1c. DOES IT HOLD WITHIN A SINGLE KICKING FOOT?")
    lines.append("-" * 68)
    if foot_rows:
        for r in foot_rows:
            lines.append(f"  {r['subset']:14s} n={r['n']:3d}  "
                         f"ROC-AUC {r['roc_auc_mean']:.3f} "
                         f"[{r['roc_auc_lo']:.3f}, {r['roc_auc_hi']:.3f}]"
                         f"{'  <- clears chance' if r['roc_beats_chance'] else ''}")
        lines.append("")
        lines.append("  The label is an XOR of the kicking foot with the target side, so a "
                     "model could in principle score by recovering the foot rather than by "
                     "measuring the action. Within one foot the XOR is constant, so a "
                     "result that survives here is a result about the body.")
    else:
        lines.append("  Not computed.")
    lines.append("")
    lines.append("2. THE MOST DISCRIMINATIVE BIOMECHANICAL SIGNALS")
    lines.append("-" * 68)
    lines.append("(frame-level significance is FDR-corrected across all "
                 "distinct base variables x frames tested)")
    lines.append("")
    top_bases = list(dict.fromkeys(base_variable(f) for f in top5))[:3]
    for i, base in enumerate(top_bases, start=1):
        desc    = describe_feature(base)
        row_sig = timeline[timeline["base_variable"] == base] if len(timeline) else timeline
        if len(row_sig) == 0:
            continue
        direction = ("higher" if row_sig["mean_crossed"].mean() > row_sig["mean_natural"].mean()
                     else "lower")
        fr = first_sig.get(base)
        if fr is not None:
            ms = (CONTACT_FRAME - fr) * ms_frame
            timing = (f"survives FDR correction from frame {fr} of 19 onward, "
                      f"i.e. around {ms:.0f}ms before contact")
        else:
            timing = ("never survives FDR correction at any single frame -- "
                      "it separates the groups on aggregate, not at any one moment")
        lines.append(f"{i}. {desc.capitalize()}")
        lines.append(f"   When {desc} is {direction} than average, the kick tends to be CROSSED.")
        lines.append(f"   This signal {timing}.")
        lines.append("")
    lines.append("3. WHEN THE BODY 'DECIDES'")
    lines.append("-" * 68)
    valid = [f for f in first_sig.values() if f is not None]
    if valid:
        earliest = min(valid)
        ms = (CONTACT_FRAME - earliest) * ms_frame
        lines.append(f"Earliest FDR-significant frame among the top variables: "
                     f"frame {earliest}, about {ms:.0f}ms before contact.")
        if ms < REACTION_TIME_MS:
            lines.append(f"That is inside the ~{REACTION_TIME_MS}ms a goalkeeper needs to "
                         f"initiate a dive, so this is useful for video analysis and "
                         f"scouting, not for a real-time reaction.")
        else:
            lines.append(f"That is outside the ~{REACTION_TIME_MS}ms reaction floor, so "
                         f"in principle there is lead time -- subject to the signal being "
                         f"readable by a human in real time, which this analysis does not test.")
    else:
        lines.append("No feature survives FDR correction at any individual frame. "
                     "There is no evidence here for a specific moment at which the "
                     "body 'gives away' the kick.")
    lines.append("")
    lines.append("4. SIMPLE RULE FOR THE GK")
    lines.append("-" * 68)
    if path:
        conds = []
        for feat, op, thr in path:
            conds.append(f"{describe_feature(feat)} is "
                         f"{'above' if op == '>' else 'below'} {thr:.1f} (standardized)")
        lines.append("IF " + " AND ".join(f"({c})" for c in conds))
        lines.append("THEN the kick tends to be CROSSED.")
        lines.append("")
        if oof:
            lines.append(f"  Out-of-fold precision of this rule: {oof['precision']:.1%} "
                         f"+/- {oof['precision_std']:.1%}, covering {oof['coverage']:.1%} "
                         f"of unseen clips.")
            lines.append(f"  (In-sample leaf purity is {purity:.1%}. That number describes "
                         f"how cleanly the tree carved up the data it was shown, not how "
                         f"often the rule is right on a new penalty -- the gap is "
                         f"{purity - oof['precision']:+.1%}.)")
        else:
            lines.append("  The rule could not be validated out of fold -- treat it as "
                         "descriptive only.")
    else:
        lines.append("The Decision Tree did not find a leaf pure/robust enough "
                     "to generate a simple rule with confidence.")
    lines.append("")
    lines.append("Methodological notes")
    lines.append("-" * 68)
    lines.append("- Top features selected via f_classif inside the pipeline, refit per fold.")
    lines.append("- ROC-AUC is the primary metric (chance = 0.500). On this class split "
                 "PR-AUC sits near the prevalence and drifts above it under fold noise.")
    lines.append("- Frame-level p-values are Benjamini-Hochberg corrected across all "
                 "features x frames tested.")
    lines.append("- Rule confidence is reported out of fold; in-sample leaf purity is "
                 "shown only for contrast.")
    lines.append("- Natural = right foot -> Right, or left foot -> Left, with Left/Right "
                 "as seen by the kicker facing the goal. Crossed is the leg swinging "
                 "across the body midline.")
    lines.append("- The label-permutation test permutes within source video, so the null "
                 "does not assume clips from one broadcast are independent.")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] {REPORT_PATH}")
    print("\n[OK] Done!")


if __name__ == "__main__":
    main()
