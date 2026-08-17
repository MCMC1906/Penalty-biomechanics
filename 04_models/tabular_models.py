"""
tabular_models.py
=================
Tabular models for three targets: kick direction (P2), whether the kick went
down the middle rather than to a corner (P4), and predictability by the GK (P1).

Models tested:
  - Logistic Regression (L2)   -- linear baseline
  - Random Forest              -- ensemble of trees, robust to outliers
  - XGBoost                    -- gradient boosting, usually the best tabular model
  - SVM (RBF kernel)           -- good with normalized features and small datasets

Evaluation
----------
  - Primary metric: ROC-AUC (chance = 0.500). Both targets are close to
    balanced, so PR-AUC -- whose chance level is the class prevalence and
    which drifts upward under fold noise -- is misleading here. PR-AUC is
    still reported for continuity with the earlier results.
  - 5-fold stratified CV repeated over 5 seeds (25 fits), not a single split.
  - A second, stricter estimate with StratifiedGroupKFold on video_name:
    clips from the same source video never straddle the split.
  - A label-permutation test for the best model of each target.

Methodology: imputation, scaling and feature selection all live INSIDE the
pipeline, so they are refit per fold -- no leakage.

Reads:  clips_master.csv
Output: tabular_results.csv

Usage:
  python tabular_models.py
  python tabular_models.py --data clips_master.csv --no-perm
"""

import argparse
import warnings
# Silence the noisy-but-harmless deprecation churn only. Convergence failures,
# degenerate folds and constant-column warnings from f_classif are exactly the
# diagnostics that matter at n=356 with 40 selected features, so they are left
# switched on.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.svm               import SVC
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline          import Pipeline
from sklearn.impute            import KNNImputer

from eval_utils import (feature_columns, repeated_cv, grouped_cv,
                        summarize, beats_chance, permutation_test, EXCLUDE)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('[WARNING] XGBoost not installed -- skipping. pip install xgboost')

K_FEATS = 40                                               # top features via SelectKBest


def build_models():
    """Pipelines -- imputation, normalization and selection inside the CV."""
    def base(clf):
        return Pipeline([
            ('imp',   KNNImputer(n_neighbors=5)),          # uses similar clips instead of the global median
            ('scale', StandardScaler()),
            ('sel',   SelectKBest(f_classif, k=K_FEATS)),
            ('clf',   clf),
        ])

    models = {
        'Logistic Regression': base(LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
        'Random Forest': base(RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1)),
        'SVM (RBF)': base(SVC(
            kernel='rbf', C=1.0, probability=True,         # probability=True for PR-AUC
            class_weight='balanced', random_state=42)),
    }
    if HAS_XGB:
        # SelectKBest/f_classif requires data without NaN -- imputation is
        # mandatory beforehand, even though XGBoost handles NaN natively
        models['XGBoost'] = base(XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1,
            eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0))
    return models


def run_target(df, features, target_col, group_a, group_b, target_name,
               run_perm=True, args=None, extra_features=()):
    print(f"\n{'='*72}")
    print(target_name)
    print('='*72)

    sub = df[df[target_col].isin([group_a, group_b])].copy().reset_index(drop=True)
    y   = (sub[target_col] == group_a).astype(int).values   # group_a = positive class (1)
    cols = [f for f in features if f in sub.columns]
    cols += [f for f in extra_features if f in sub.columns and f not in cols]
    X   = sub[cols].copy()
    groups = sub['video_name'].values if 'video_name' in sub.columns else None

    pr_baseline  = y.mean()                                 # PR-AUC chance level = prevalence
    maj_baseline = max(y.mean(), 1 - y.mean())              # accuracy of always predicting the majority class

    print(f"  n={len(sub)}  |  class 1: {y.sum()}  class 0: {(1 - y).sum()}")
    print(f"  {X.shape[1]} features  |  chance ROC-AUC 0.500  |  "
          f"chance PR-AUC {pr_baseline:.3f}  |  majority-class accuracy {maj_baseline:.3f}")
    if groups is not None:
        vc = pd.Series(groups).value_counts()
        print(f"  {len(vc)} source videos (largest contributes {vc.iloc[0]} clips)")

    print(f"\n  {'Model':22s}  {'ROC-AUC [95% fold range]':<28s}{'PR-AUC':<18s}>chance")
    print(f"  {'-'*22}  {'-'*28}{'-'*18}-------")

    results = []
    for name, pipe in build_models().items():
        pipe.set_params(sel__k=min(K_FEATS, X.shape[1]))    # never ask for more features than exist

        scores = repeated_cv(X, y, pipe)
        row = {'model': name, 'target': target_name, 'n': len(sub)}
        row.update(summarize(scores))
        row['pr_auc_baseline']   = round(float(pr_baseline), 4)
        row['majority_accuracy'] = round(float(maj_baseline), 4)
        row['roc_beats_chance']  = beats_chance(scores)

        if groups is not None:
            row.update(summarize(grouped_cv(X, y, groups, pipe), prefix='grouped_'))

        roc_txt = (f"{row['roc_auc_mean']:.3f} "
                   f"[{row['roc_auc_lo']:.3f}, {row['roc_auc_hi']:.3f}]")
        pr_txt  = f"{row['pr_auc_mean']:.3f}+/-{row['pr_auc_std']:.3f}"
        flag    = '[OK]' if row['roc_beats_chance'] else '--'
        print(f"  {name:22s}  {roc_txt:<28s}{pr_txt:<18s}{flag}")
        results.append(row)

    # Grouped-CV comparison: the drop between the two is the size of the
    # video-level effect the random split was quietly rewarding.
    if groups is not None:
        print(f"\n  Grouped by source video (StratifiedGroupKFold):")
        for r in results:
            delta = r['grouped_roc_auc_mean'] - r['roc_auc_mean']
            print(f"    {r['model']:22s} ROC-AUC {r['grouped_roc_auc_mean']:.3f} "
                  f"({delta:+.3f} vs random split)")

    # Permutation test on the best model by grouped ROC-AUC
    if run_perm:
        key  = 'grouped_roc_auc_mean' if groups is not None else 'roc_auc_mean'
        best = max(results, key=lambda r: r[key])
        pipe = build_models()[best['model']]
        pipe.set_params(sel__k=min(K_FEATS, X.shape[1]))
        print(f"\n  Label-permutation test -- {best['model']} "
              f"({args.n_perm} shuffles, permuted within source video):")
        perm = permutation_test(X, y, pipe, n_perm=args.n_perm, groups=groups)
        print(f"    observed ROC-AUC {perm['observed']:.3f}  |  "
              f"null mean {perm['null_mean']:.3f}, 95th pct {perm['null_p95']:.3f}  |  "
              f"p = {perm['p_value']:.4f}")
        if perm['p_value'] >= 0.05:
            print("    -> not distinguishable from chance on this dataset.")
        for r in results:
            if r['model'] == best['model']:
                r['permutation_p'] = perm['p_value']

    return results


def main():
    ap = argparse.ArgumentParser(
        description='Tabular models: LR, RF, XGBoost, SVM.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--data', '-d', default='clips_master.csv')
    ap.add_argument('--no-perm', action='store_true',
                    help='skip the label-permutation test (slow)')
    ap.add_argument('--n-perm', type=int, default=200,
                    help='number of label shuffles (default 200)')
    args = ap.parse_args()

    print(f'Loading {args.data} ...')
    df = pd.read_csv(args.data)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    features = feature_columns(df)

    all_results = []
    all_results += run_target(df, features, 'macro_zone', 'Left', 'Right',
                              'P2 -- Direction (Left vs Right)',
                              run_perm=not args.no_perm, args=args)

    # P4: centre vs cornered.
    #
    # The 45 kicks aimed at the middle column of the goal (zones 2, 5, 8) sit
    # outside every other direction analysis in this project -- macro_zone
    # covers only the corner and side zones, so those clips are dropped before
    # P2 and before Natural vs Crossed. This target puts them back in.
    #
    # It is the better-posed of the direction questions in one respect: it does
    # not depend on the kicking foot, so there is no XOR, no interaction to
    # unfold, and none of the representation problem that made Left/Right look
    # null. Going down the middle is a squarer, less rotated action, so the
    # mechanism is specific enough to check against the feature screen.
    #
    # READ IT WITH THE CLASS BALANCE IN VIEW. 45 positives against 356 is
    # roughly 1:7, so the fold ranges will be wide whatever happens, and a null
    # here does not distinguish "no signal" from "not enough clips". Only a
    # clear positive would mean anything. This is also the fifth target tested
    # on this dataset, so a permutation p-value should be read against 0.01
    # rather than 0.05.
    df['zone_group'] = df['macro_zone'].map(
        {'Center': 'Centre', 'Left': 'Cornered', 'Right': 'Cornered'})
    all_results += run_target(df, features, 'zone_group', 'Centre', 'Cornered',
                              'P4 -- Centre vs cornered',
                              run_perm=not args.no_perm, args=args)

    all_results += run_target(df, features, 'gk_guessed', 1, 0,
                              'P1 -- Predictability (gk_guessed)',
                              run_perm=not args.no_perm, args=args)

    pd.DataFrame(all_results).to_csv('tabular_results.csv', index=False)
    print(f'\n[OK] tabular_results.csv')
    print('[OK] Done!')


if __name__ == '__main__':
    main()
