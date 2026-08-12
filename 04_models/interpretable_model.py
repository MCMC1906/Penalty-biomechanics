"""
interpretable_model.py
======================
Interpretable models to extract actionable biomechanical rules.

Two levels of interpretability:

  1. Decision Tree (depth 3) -- direct rules in natural language
     "IF speed > X AND angle < Y THEN Left (Z% confidence)"
     Useful for coaches and GKs -- no technical knowledge required

  2. SHAP values on the Random Forest -- importance of each feature in each
     prediction. Useful for researchers -- more precise than the average
     feature_importances_

Two corrections relative to the earlier version
-----------------------------------------------
  1. Feature selection and imputation used to be fit on the FULL dataset
     before the CV loop, so the reported PR-AUC was optimistic. They now live
     inside the pipeline and are refit per fold.

  2. Rule confidence used to be read off a leaf of a tree trained on all the
     data ("94% confidence, 28 cases"). That is in-sample purity: a depth-3
     tree will always find a clean-looking corner of the data it was shown.
     Rules are now also scored OUT OF FOLD -- the tree is built on the
     training part and its rule is applied to clips it has never seen. The
     out-of-fold number is the one worth reporting.

Reads:  clips_master.csv
Output:
  - decision_tree_rules.txt    -- rules in natural language, with OOF support
  - shap_summary_<target>.png  -- SHAP importance chart (one per target)
  - interpretable_results.csv

Usage:
  python interpretable_model.py
  python interpretable_model.py --data clips_master.csv
"""

import argparse
import warnings
# Silence the noisy-but-harmless deprecation churn only. Convergence failures,
# degenerate folds and constant-column warnings are exactly the diagnostics
# that matter at this sample size, so they are left switched on.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                                      # windowless backend -- avoids the tkinter error
import matplotlib.pyplot as plt

from sklearn.base              import clone
from sklearn.tree              import DecisionTreeClassifier, export_text
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline          import Pipeline
from sklearn.impute            import KNNImputer
from sklearn.model_selection   import StratifiedKFold

from eval_utils import feature_columns, repeated_cv, summarize, beats_chance

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print('[WARNING] SHAP not installed -- skipping SHAP analysis. pip install shap')

MIN_SAMPLES_LEAF = 15   # minimum clips per leaf -- avoids overfitting on small samples
MAX_DEPTH        = 3    # maximum depth -- more than 3 loses interpretability
K_SELECT         = 30   # features handed to the tree, chosen inside each fold


def build_tree_pipeline(min_leaf=MIN_SAMPLES_LEAF, k=K_SELECT):
    """Imputation + selection + tree, all refit per fold."""
    return Pipeline([
        ('imp',   KNNImputer(n_neighbors=5)),
        ('scale', StandardScaler()),
        ('sel',   SelectKBest(f_classif, k=k)),
        ('clf',   DecisionTreeClassifier(max_depth=MAX_DEPTH,
                                         min_samples_leaf=min_leaf,
                                         class_weight='balanced',
                                         random_state=42)),
    ])


# -- Out-of-fold validation of the extracted rule ---------------------

def purest_leaf(tree, feature_names, target_class=1, min_n=10):
    """Path to the purest leaf for target_class with at least min_n samples."""
    t = tree.tree_
    best = {'purity': -1, 'path': None, 'n': 0, 'node': None}

    def recurse(node, path):
        if t.feature[node] == -2:                          # leaf
            values = t.value[node][0]                      # weighted counts per class
            n      = t.n_node_samples[node]                # real sample count
            purity = values[target_class] / values.sum()
            if n >= min_n and purity > best['purity']:
                best.update(purity=purity, path=list(path), n=int(n), node=node)
            return
        feat = feature_names[t.feature[node]]
        thr  = t.threshold[node]
        recurse(t.children_left[node],  path + [(feat, '<=', thr)])
        recurse(t.children_right[node], path + [(feat, '>',  thr)])

    recurse(0, [])
    return best


def rule_mask(df, path):
    """Boolean mask of the clips that satisfy every condition in the path."""
    mask = np.ones(len(df), dtype=bool)
    for feat, op, thr in path:
        col = df[feat].values
        mask &= (col <= thr) if op == '<=' else (col > thr)
    return mask & ~df[[f for f, _, _ in path]].isna().any(axis=1).values


def out_of_fold_rule_support(X, y, min_leaf, n_splits=5, n_repeats=5, seed=42,
                             target_class=1):
    """Build the tree on the training part, apply its own rule to the held-out
    part, and record how often the rule holds there.

    This is the number that answers "if I use this rule on a new penalty, how
    often am I right?" -- unlike leaf purity, which only answers "how cleanly
    did the tree carve up the data it was shown?".
    """
    precisions, coverages = [], []
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(X, y):
            pipe = build_tree_pipeline(min_leaf)
            pipe.fit(X.iloc[tr], y[tr])

            # Names of the features the selector kept, in tree column order
            kept = [X.columns[i] for i in pipe['sel'].get_support(indices=True)]
            leaf = purest_leaf(pipe['clf'], kept, target_class=target_class)
            if leaf['path'] is None:
                continue

            # The tree splits on SCALED values, so the test data must go
            # through the same imputer + scaler before the rule is applied.
            X_te_t = pd.DataFrame(
                pipe['scale'].transform(pipe['imp'].transform(X.iloc[te]))[
                    :, pipe['sel'].get_support(indices=True)],
                columns=kept)

            m = rule_mask(X_te_t, leaf['path'])
            if m.sum() == 0:
                continue
            precisions.append(float((y[te][m] == target_class).mean()))
            coverages.append(float(m.mean()))

    if not precisions:
        return None
    return {
        'oof_precision_mean': round(float(np.mean(precisions)), 4),
        'oof_precision_std' : round(float(np.std(precisions)), 4),
        'oof_coverage_mean' : round(float(np.mean(coverages)), 4),
        'n_folds_with_rule' : len(precisions),
    }


# -- Decision Tree ----------------------------------------------------

def run_decision_tree(X, y, target_name, min_leaf=MIN_SAMPLES_LEAF):
    """CV score, then a final tree on all the data purely to print readable rules."""
    print(f"\n  Decision Tree (depth={MAX_DEPTH}, min_leaf={min_leaf}, "
          f"selection inside the CV)")

    pipe   = build_tree_pipeline(min_leaf, k=min(K_SELECT, X.shape[1]))
    scores = repeated_cv(X, y, pipe)
    summ   = summarize(scores)
    print(f"  ROC-AUC: {summ['roc_auc_mean']:.3f} "
          f"[{summ['roc_auc_lo']:.3f}, {summ['roc_auc_hi']:.3f}]  |  "
          f"PR-AUC: {summ['pr_auc_mean']:.3f}")
    if not beats_chance(scores):
        print("  (fold range includes chance -- the rules below are descriptive, "
              "not predictive)")

    # Final fit on everything, only to render human-readable rules
    final = build_tree_pipeline(min_leaf, k=min(K_SELECT, X.shape[1]))
    final.fit(X, y)
    kept  = [X.columns[i] for i in final['sel'].get_support(indices=True)]
    rules = export_text(final['clf'], feature_names=list(kept), max_depth=MAX_DEPTH)

    print(f"\n  Decision tree ({target_name}) -- thresholds are on STANDARDIZED units:")
    print(rules)

    leaf = purest_leaf(final['clf'], kept)
    oof  = out_of_fold_rule_support(X, y, min_leaf)
    if leaf['path'] is not None:
        print(f"  Purest leaf: in-sample purity {leaf['purity']:.1%} on {leaf['n']} clips")
        if oof:
            print(f"  Same rule, out of fold: {oof['oof_precision_mean']:.1%} "
                  f"+/- {oof['oof_precision_std']:.1%} precision, "
                  f"covering {oof['oof_coverage_mean']:.1%} of unseen clips")
            gap = leaf['purity'] - oof['oof_precision_mean']
            print(f"  -> in-sample purity overstates the rule by {gap:+.1%}")

    # Imputed+scaled frame reused by the SHAP step
    X_used = pd.DataFrame(
        final['scale'].transform(final['imp'].transform(X))[
            :, final['sel'].get_support(indices=True)],
        columns=kept)

    return summ, rules, leaf, oof, X_used, kept


# -- SHAP -------------------------------------------------------------

def run_shap(X_used, y, feature_names, target_name):
    """SHAP values for the Random Forest -- importance per feature and per prediction.

    Note: this model is fit on all the data, so these importances are
    descriptive of the fitted model, not an out-of-sample claim.
    """
    if not HAS_SHAP:
        return None

    print(f"\n  SHAP values (Random Forest, fit on all data -- descriptive only)")

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_used, y)

    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_used)

    # SHAP returns different shapes depending on the installed version:
    # - list [class0, class1]   -- older versions
    # - 3D array (n, feats, 2)  -- recent versions
    # - 2D array (n, feats)     -- general case
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    top_idx       = np.argsort(mean_abs_shap)[::-1][:15]
    top_features  = [feature_names[i] for i in top_idx.tolist()]
    top_vals      = mean_abs_shap[top_idx]

    print(f"\n  Top 15 features by SHAP importance ({target_name}):")
    for feat, val in zip(top_features, top_vals):
        print(f"    {feat:45s}  SHAP={val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv[:, top_idx], X_used.iloc[:, top_idx],
                      feature_names=top_features, show=False, plot_size=None)
    plt.title(f'SHAP Summary -- {target_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    safe_name = (target_name.replace(' ', '_').replace('/', '_')
                 .replace('(', '').replace(')', ''))
    out_path  = f'shap_summary_{safe_name}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] {out_path}")

    return dict(zip(top_features, top_vals.tolist()))


# -- Main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Interpretable models: Decision Tree + SHAP.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--data', '-d', default='clips_master.csv')
    ap.add_argument('--min-leaf', type=int, default=MIN_SAMPLES_LEAF,
                    help=f'Min samples per leaf (default={MIN_SAMPLES_LEAF})')
    args = ap.parse_args()

    print(f'Loading {args.data} ...')
    df = pd.read_csv(args.data)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    features = feature_columns(df)

    targets = [
        ('macro_zone', 'Left', 'Right', 'P2 -- Direction (Left vs Right)'),
        ('gk_guessed', 1, 0, 'P1 -- Predictability (gk_guessed)'),
    ]

    all_results, all_rules = [], []

    for target_col, group_a, group_b, target_name in targets:
        print(f"\n{'='*60}")
        print(target_name)
        print('='*60)

        sub = df[df[target_col].isin([group_a, group_b])].copy().reset_index(drop=True)
        y   = (sub[target_col] == group_a).astype(int).values
        X   = sub[[f for f in features if f in sub.columns]].copy()
        print(f"  n={len(sub)}  |  chance ROC-AUC 0.500  |  "
              f"chance PR-AUC {y.mean():.3f}")

        summ, rules, leaf, oof, X_used, kept = run_decision_tree(
            X, y, target_name, args.min_leaf)

        block = [f"\n{'='*60}\n{target_name}\n{'='*60}",
                 "Thresholds are on standardized (z-scored) units.",
                 rules]
        if leaf['path'] is not None:
            block.append(f"Purest leaf -- in-sample purity: {leaf['purity']:.1%} "
                         f"on {leaf['n']} clips")
            if oof:
                block.append(
                    f"Same rule applied out of fold: "
                    f"{oof['oof_precision_mean']:.1%} +/- {oof['oof_precision_std']:.1%} "
                    f"precision on unseen clips, covering "
                    f"{oof['oof_coverage_mean']:.1%} of them "
                    f"({oof['n_folds_with_rule']} folds). "
                    f"Report this number, not the in-sample purity.")
        all_rules.append('\n'.join(block))

        row = {'model': 'Decision Tree', 'target': target_name, 'n': len(sub)}
        row.update(summ)
        # Explicit flag: a fold range that includes 0.500 is not a result.
        # Leaving it absent made downstream NaN read as truthy.
        row['roc_beats_chance'] = bool(summ['roc_auc_lo'] > 0.5)
        row['pr_auc_baseline'] = round(float(y.mean()), 4)
        row['leaf_in_sample_purity'] = (round(float(leaf['purity']), 4)
                                        if leaf['path'] is not None else None)
        if oof:
            row.update(oof)
        all_results.append(row)

        run_shap(X_used, y, list(kept), target_name)

    with open('decision_tree_rules.txt', 'w', encoding='utf-8') as f:
        f.write("DECISION TREE RULES\n")
        f.write("Each leaf shows: predicted class [n samples / proportion]\n")
        f.write("In-sample leaf purity is NOT a prediction accuracy -- the "
                "out-of-fold line below each tree is.\n")
        f.write('\n'.join(all_rules))
    print(f'\n[OK] decision_tree_rules.txt')

    pd.DataFrame(all_results).to_csv('interpretable_results.csv', index=False)
    print('[OK] interpretable_results.csv')
    print('[OK] Done!')


if __name__ == '__main__':
    main()
