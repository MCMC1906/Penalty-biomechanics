"""
statistical_tests_fase2.py
==========================
Phase 2 - Analysis by time phase.

Trains 3 separate models for P2 (Left vs Right):
  - Model A: only early_mean_* features  (frames 0-6)
  - Model B: only mid_mean_* features    (frames 7-13)
  - Model C: only late_mean_* features   (frames 14-19)

Central question: at what point in the run-up is the information about the kick
direction present? If Model A approx Model C, the intent is detectable from the
start, giving the GK time to react.

Also repeats for P1 (gk_guessed) and for Natural vs Crossed -- which is the
target where there is a signal to locate in time, and which this script did not
previously cover.

Corrections
-----------
  1. PR-AUC was the metric and 0.5 was drawn as its baseline. On these class
     splits PR-AUC sits near the prevalence and drifts above it under fold
     noise, so a chance-level model looked like it beat the line. ROC-AUC is
     now primary (chance = 0.500 by construction); PR-AUC is kept in the table
     next to its own prevalence baseline.

  2. KNNImputer was fit on the whole dataset before the CV loop, so every fold
     saw test-fold values through the imputation. It is now inside the
     pipeline, refit per fold, like everywhere else in the project.

  3. A single 5-fold split with random_state=42 decided every comparison. Now
     5 folds x 5 seeds.

  4. Phases were ranked with a hardcoded 0.02 threshold for "equivalent", which
     declared winners well inside the noise. The threshold is now the observed
     fold-to-fold spread, and when no phase clears chance the script says so
     instead of naming a best one.

Output:
  - phase2_results.csv
  - phase2_comparison.png (ROC-AUC bar chart by phase, chance line at 0.500)

Usage:
  python statistical_tests_fase2.py
"""

import warnings
# Deprecation churn only -- convergence failures and degenerate folds
# are diagnostics worth seeing at this sample size.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                                      # windowless backend -- avoids the tkinter error on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.base             import clone
from sklearn.ensemble         import RandomForestClassifier
from sklearn.preprocessing    import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline         import Pipeline
from sklearn.impute            import KNNImputer
from sklearn.model_selection  import RepeatedStratifiedKFold
from sklearn.metrics          import average_precision_score, roc_auc_score

from stats_utils import feature_columns, is_empty, build_shot_type

sns.set_theme(style='darkgrid', font_scale=1.05)

# -- Configuration --------------------------------------------------

CV_FOLDS   = 5
N_REPEATS  = 5     # 25 fits per phase -- a single split ranked phases on noise
SEED       = 42
K_FEATURES = 15    # top features per phase -- fewer than the full model because each phase has fewer features

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.


def build_pipeline(k=K_FEATURES):
    return Pipeline([
        ('imp',   KNNImputer(n_neighbors=5)),
        ('scale', StandardScaler()),
        ('sel',   SelectKBest(f_classif, k=k)),
        ('clf',   RandomForestClassifier(
            n_estimators=200, max_depth=4,
            min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1)),
    ])


def cv_scores(X, y, pipeline, cv):
    """Repeated CV. Returns per-fold ROC-AUC and PR-AUC.

    ROC-AUC is primary: its chance level is 0.500 whatever the class balance.
    PR-AUC is reported alongside, but only ever against its own baseline (the
    prevalence), never against 0.5.
    """
    roc, pr = [], []
    for tr, te in cv.split(X, y):
        m = clone(pipeline)
        m.fit(X.iloc[tr], y[tr])
        proba = m.predict_proba(X.iloc[te])[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        roc.append(roc_auc_score(y[te], proba))
        pr.append(average_precision_score(y[te], proba))
    return np.array(roc), np.array(pr)


def run_phase_analysis(df, target_col, group_a, group_b, label_a, label_b, task_name):
    """Trains one model per time phase and compares them on ROC-AUC."""
    mask = df[target_col].isin([group_a, group_b])
    df_f = df[mask].copy().reset_index(drop=True)
    y    = (df_f[target_col] == group_a).astype(int).values

    if len(np.unique(y)) < 2 or len(df_f) < 2 * CV_FOLDS:
        print(f"  Not enough clips for {task_name} (n={len(df_f)}); skipped.")
        return []

    cv = RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    prevalence = float(y.mean())
    results = []

    phases = {
        'Early (frames 0-6)'  : 'early_',
        'Mid   (frames 7-13)' : 'mid_',
        'Late  (frames 14-19)': 'late_',
        'All features'        : None,
    }

    print(f"  n={len(df_f)}  |  chance ROC-AUC 0.500  |  chance PR-AUC {prevalence:.3f}")

    all_feats = feature_columns(df_f, verbose=False)
    for phase_name, prefix in phases.items():
        cols = ([c for c in all_feats if c.startswith(prefix)] if prefix
                else list(all_feats))
        cols = [c for c in cols if df_f[c].notna().any()]
        if len(cols) < 2:
            continue

        # Imputation, scaling and selection all live inside the pipeline, so
        # they are refit on each training fold. The imputer used to be fit on
        # the full dataset before this loop, which let every fold see test-fold
        # values through the imputed cells.
        pipe = build_pipeline(k=min(K_FEATURES, len(cols)))
        roc, pr = cv_scores(df_f[cols], y, pipe, cv)
        if len(roc) == 0:
            continue

        lo, hi = np.percentile(roc, [2.5, 97.5])
        results.append({
            'task'          : task_name,
            'phase'         : phase_name,
            'n_features'    : len(cols),
            'ROC_AUC_mean'  : round(float(roc.mean()), 4),
            'ROC_AUC_std'   : round(float(roc.std()), 4),
            'ROC_AUC_lo'    : round(float(lo), 4),
            'ROC_AUC_hi'    : round(float(hi), 4),
            'beats_chance'  : bool(lo > 0.5),
            'PR_AUC_mean'   : round(float(pr.mean()), 4),
            'PR_AUC_baseline': round(prevalence, 4),
            'group_a'       : label_a,
            'group_b'       : label_b,
        })

        print(f"  {phase_name:25s} n_feat={len(cols):3d}  "
              f"ROC-AUC = {roc.mean():.3f} [{lo:.3f}, {hi:.3f}]  "
              f"{'[OK]' if lo > 0.5 else '--'}")

    return results


def interpret(res_df):
    """Rank phases, but only where the ranking is bigger than the fold noise.

    The old version used a hardcoded 0.02 gap to call two phases 'equivalent'
    and otherwise named a winner -- on differences far inside the fold-to-fold
    spread, and on models that did not clear chance at all.
    """
    for task in res_df['task'].unique():
        sub = res_df[res_df['task'] == task]
        print(f"\n  {task}")

        clears = sub[sub['beats_chance']]
        if clears.empty:
            print("    No phase clears chance: every fold range includes 0.500.")
            print("    Nothing here supports a claim about WHEN the signal appears,")
            print("    because there is no signal to locate.")
            continue

        e = sub[sub['phase'].str.startswith('Early')]
        l = sub[sub['phase'].str.startswith('Late')]
        if e.empty or l.empty:
            continue
        e_mean, l_mean = float(e.iloc[0]['ROC_AUC_mean']), float(l.iloc[0]['ROC_AUC_mean'])
        # The threshold is the observed spread, not a constant.
        noise = float(max(e.iloc[0]['ROC_AUC_std'], l.iloc[0]['ROC_AUC_std']))
        diff  = l_mean - e_mean
        print(f"    Early={e_mean:.3f}  Late={l_mean:.3f}  "
              f"(fold-to-fold sd {noise:.3f})")
        if abs(diff) <= noise:
            print(f"    Delta Late-Early = {diff:+.3f}, within the fold spread: "
                  f"the phases are indistinguishable.")
        elif diff > 0:
            print(f"    Delta Late-Early = {diff:+.3f}, larger than the fold spread: "
                  f"the signal concentrates near contact.")
        else:
            print(f"    Delta Late-Early = {diff:+.3f}, larger than the fold spread: "
                  f"the signal is stronger during the approach.")


def plot_comparison(all_results, out_path):
    df     = pd.DataFrame(all_results)
    tasks  = df['task'].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(7*len(tasks), 6), sharey=False)
    if len(tasks) == 1: axes = [axes]

    palette    = sns.color_palette('deep', 4)
    phase_order = ['Early (frames 0-6)', 'Mid   (frames 7-13)',
                   'Late  (frames 14-19)', 'All features']

    for ax, task in zip(axes, tasks):
        sub  = df[df['task']==task].set_index('phase').reindex(phase_order).dropna()
        bars = ax.bar(range(len(sub)), sub['ROC_AUC_mean'],
                      color=palette[:len(sub)], alpha=0.85, width=0.6)
        ax.errorbar(range(len(sub)), sub['ROC_AUC_mean'], yerr=sub['ROC_AUC_std'],
                    fmt='none', color='black', capsize=5, lw=1.5)
        # 0.500 is the chance level for ROC-AUC by construction. The previous
        # version drew this line at 0.5 for PR-AUC, whose chance level is the
        # class prevalence -- so bars sitting at noise appeared to clear it.
        ax.axhline(0.5, color='red', ls='--', lw=1.2, alpha=0.6, label='Chance (0.500)')
        for bar, val in zip(bars, sub['ROC_AUC_mean']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='500')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([p.strip() for p in sub.index], rotation=15, ha='right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_ylabel('ROC-AUC (5 folds x 5 seeds)', fontsize=11)
        ax.set_title(task, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    fig.suptitle('Phase 2 -- Information by Time Phase\nAt what point in the run-up is there the most biomechanical signal?',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] {out_path}")


def main():
    print('Loading clips_master.csv ...')
    df = pd.read_csv('clips_master.csv')
    print(f'  {df.shape[0]} clips\n')

    all_results = []

    # -- P2: Left vs Right ----------------------------------
    print("=" * 60)
    print("P2 -- Direction: Left vs Right")
    print("=" * 60)
    all_results.extend(run_phase_analysis(
        df, 'macro_zone', 'Left', 'Right', 'Left', 'Right',
        'P2 -- Direction (Left vs Right)'))

    # -- P1: gk_guessed ------------------------------------------
    print("\n" + "=" * 60)
    print("P1 -- Predictability: Guessed vs Not Guessed")
    print("=" * 60)
    all_results.extend(run_phase_analysis(
        df, 'gk_guessed', 1, 0, 'Guessed', 'NotGuessed',
        'P1 -- Predictability (gk_guessed)'))

    # -- Natural vs Crossed --------------------------------------
    #
    # This is the target with a signal, so it is the one where "at what point
    # in the run-up?" is a real question rather than a hypothetical. The script
    # did not previously cover it. Read the Early and Mid rows against the Late
    # one: the per-clip features that survive correction on this target are
    # measured at the contact frame, so a Late-only advantage means the model is
    # reading the strike rather than anticipating it.
    print("\n" + "=" * 60)
    print("Natural vs Crossed")
    print("=" * 60)
    nc = build_shot_type(df)
    all_results.extend(run_phase_analysis(
        nc, 'shot_type', 'Crossed', 'Natural', 'Crossed', 'Natural',
        'Natural vs Crossed'))

    if not all_results:
        print("\nNo task had enough clips to compare phases.")
        return

    # -- Save and plot -----------------------------------------
    res_df = pd.DataFrame(all_results)
    res_df.to_csv('phase2_results.csv', index=False)
    print(f"\n[OK] phase2_results.csv")
    plot_comparison(all_results, 'phase2_comparison.png')

    # -- Interpretation --------------------------------------------
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    interpret(res_df)

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
