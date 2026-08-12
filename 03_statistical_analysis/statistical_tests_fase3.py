"""
statistical_tests_fase3.py
==========================
Phase 3 -- Does the GK behave like a computer?

Answers P3 in 3 layers:

  1. Spearman correlation between P1 and P2 feature rankings
     Do the two models value the same features?

  2. Conditional analysis
     When the P2 model gets the zone right, did the GK also guess more?

  3. Concordance analysis
     Do kicks where both get it right have different biomechanics?

Output:
  - phase3_results.txt
  - phase3_spearman.png
  - phase3_conditional.png

Usage:
  python statistical_tests_fase3.py
"""

import warnings
# Deprecation churn only -- convergence failures and degenerate folds
# are diagnostics worth seeing at this sample size.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd

from stats_utils import feature_columns, is_empty
import matplotlib
matplotlib.use('Agg')                                      # windowless backend -- avoids the tkinter error on Windows
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats               import spearmanr, chi2_contingency, mannwhitneyu
from sklearn.base              import clone
from sklearn.ensemble          import RandomForestClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline          import Pipeline
from sklearn.impute             import SimpleImputer
from sklearn.model_selection   import StratifiedKFold, cross_val_predict
from sklearn.metrics           import average_precision_score

sns.set_theme(style='darkgrid', font_scale=1.05)

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.
K  = 40                                                    # top features for the model (more than fase2 -- uses every phase)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def build_pipe(k=K):
    return Pipeline([
        ('imp',   SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('sel',   SelectKBest(f_classif, k=min(k, 60))),
        ('clf',   RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced',
            random_state=42, n_jobs=-1)),
    ])


def get_feature_importance(X, y, pipe):
    """Trains the pipeline and extracts the importances mapped back to the original features."""
    p = clone(pipe); p.fit(X, y)
    sel_mask   = p.named_steps['sel'].get_support()       # mask of the features selected by SelectKBest
    feat_names = np.array(X.columns)[sel_mask]
    imps       = p.named_steps['clf'].feature_importances_
    return pd.Series(imps, index=feat_names).sort_values(ascending=False)


def prepare_X(df):                                         # selects and imputes numeric features
    cols  = feature_columns(df, verbose=False)
    valid = [c for c in cols if df[c].notna().any()]
    imp   = SimpleImputer(strategy='median')
    return pd.DataFrame(imp.fit_transform(df[valid]), columns=valid)


# -- Layer 1 -- Spearman correlation between P1 and P2 rankings ----

def layer1_spearman(df, report):
    print("\n" + "="*60)
    print("LAYER 1 -- Spearman correlation between P1 and P2 rankings")
    print("="*60)

    # P2
    df_p2  = df[df['macro_zone'].isin(['Left','Right'])].copy().reset_index(drop=True)
    X_p2   = prepare_X(df_p2)
    y_p2   = (df_p2['macro_zone'] == 'Left').astype(int).values
    imp_p2 = get_feature_importance(X_p2, y_p2, build_pipe())

    # P1
    df_p1  = df.dropna(subset=['gk_guessed']).copy().reset_index(drop=True)
    X_p1   = prepare_X(df_p1)
    y_p1   = df_p1['gk_guessed'].astype(int).values
    imp_p1 = get_feature_importance(X_p1, y_p1, build_pipe())

    common = list(set(imp_p2.index) & set(imp_p1.index))  # features present in both models
    print(f"\n  Features common to both models: {len(common)}")

    rank_p2 = imp_p2[common].rank(ascending=False)
    rank_p1 = imp_p1[common].rank(ascending=False)

    rho, p_rho = spearmanr(rank_p2, rank_p1)
    print(f"  Spearman rho = {rho:.4f}  |  p = {p_rho:.4f}")
    sig = "[OK] Significant correlation" if p_rho < 0.05 else "[NOK] No significant correlation"
    print(f"  {sig}")

    report.append(f"\nLAYER 1 -- Spearman between P1 and P2 rankings")
    report.append(f"  Common features: {len(common)}")
    report.append(f"  rho = {rho:.4f}  p = {p_rho:.4f}  -> {sig}")

    print(f"\n  Top 10 features P2 (direction):")
    for i, (feat, val) in enumerate(imp_p2.head(10).items(), 1):
        rank_in_p1 = int(rank_p1.get(feat, -1)) if feat in rank_p1.index else '--'
        print(f"    {i:2d}. {feat:45s} imp={val:.4f}  rank_P1={rank_in_p1}")

    print(f"\n  Top 10 features P1 (gk_guessed):")
    for i, (feat, val) in enumerate(imp_p1.head(10).items(), 1):
        rank_in_p2 = int(rank_p2.get(feat, -1)) if feat in rank_p2.index else '--'
        print(f"    {i:2d}. {feat:45s} imp={val:.4f}  rank_P2={rank_in_p2}")

    # Ranking scatter
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(rank_p2, rank_p1, alpha=0.5, color='#1f77b4', s=40)

    # Highlight the top 5 of each model in red
    highlight = (set(imp_p2.head(5).index) | set(imp_p1.head(5).index)) & set(common)
    for feat in highlight:
        if feat in rank_p2.index and feat in rank_p1.index:
            ax.scatter(rank_p2[feat], rank_p1[feat], color='#d62728', s=80, zorder=3)
            short = feat.replace('early_mean_','e:').replace('late_mean_','l:').replace('mid_mean_','m:').replace('delta_','d:')
            ax.annotate(short, (rank_p2[feat], rank_p1[feat]),
                        xytext=(5,5), textcoords='offset points', fontsize=7.5, color='#d62728')

    ax.set_xlabel('Ranking in the P2 model (Direction)', fontsize=11)
    ax.set_ylabel('Ranking in the P1 model (gk_guessed)', fontsize=11)
    ax.set_title(f'Spearman Correlation between Rankings\nrho = {rho:.3f}  p = {p_rho:.4f}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig('phase3_spearman.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] phase3_spearman.png")

    return imp_p2, imp_p1, rho, p_rho


# -- Layer 2 -- Conditional analysis ------------------------------

def layer2_conditional(df, report):
    print("\n" + "="*60)
    print("LAYER 2 -- Conditional Analysis")
    print("When the model gets the zone right, did the GK also guess more?")
    print("="*60)

    df_p2 = df[df['macro_zone'].isin(['Left','Right'])].copy().reset_index(drop=True)
    X_p2  = prepare_X(df_p2)
    y_p2  = (df_p2['macro_zone'] == 'Left').astype(int).values

    # cross_val_predict generates out-of-fold predictions -- no data leakage
    preds = cross_val_predict(build_pipe(), X_p2, y_p2, cv=CV, method='predict')
    df_p2['model_correct'] = (preds == y_p2).astype(int)
    df_p2 = df_p2.dropna(subset=['gk_guessed'])

    rate_correct = df_p2[df_p2['model_correct']==1]['gk_guessed'].mean()
    rate_wrong   = df_p2[df_p2['model_correct']==0]['gk_guessed'].mean()
    n_correct    = (df_p2['model_correct']==1).sum()
    n_wrong      = (df_p2['model_correct']==0).sum()

    ct = pd.crosstab(df_p2['model_correct'], df_p2['gk_guessed'].astype(int))
    chi2, p_chi, dof, _ = chi2_contingency(ct)

    print(f"\n  When the model was RIGHT (n={n_correct}):  GK guessed {rate_correct*100:.1f}%")
    print(f"  When the model was WRONG (n={n_wrong}):  GK guessed {rate_wrong*100:.1f}%")
    print(f"\n  Chi-squared: chi2={chi2:.3f}  p={p_chi:.4f}  df={dof}")
    sig = "[OK] Significant association" if p_chi < 0.05 else "[NOK] No significant association"
    print(f"  {sig}")

    report.append(f"\nLAYER 2 -- Conditional Analysis")
    report.append(f"  GK guessed when the model was right:  {rate_correct*100:.1f}%  (n={n_correct})")
    report.append(f"  GK guessed when the model was wrong:  {rate_wrong*100:.1f}%  (n={n_wrong})")
    report.append(f"  chi2={chi2:.3f}  p={p_chi:.4f}  -> {sig}")

    # Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(['Model\nCorrect','Model\nWrong'],
                  [rate_correct*100, rate_wrong*100],
                  color=['#2ca02c','#d62728'], alpha=0.85, width=0.5)
    for bar, v in zip(bars, [rate_correct*100, rate_wrong*100]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.1f}%', ha='center', fontsize=12, fontweight='500')
    ax.axhline(df_p2['gk_guessed'].mean()*100, color='grey', ls='--', lw=1.5,
               label=f'Overall average ({df_p2["gk_guessed"].mean()*100:.1f}%)')
    ax.set_ylim(0, 80)
    ax.set_ylabel('% GK Guessed', fontsize=11)
    ax.set_title(f'GK rate by model correctness\nchi2={chi2:.2f}  p={p_chi:.3f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[1]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.index = ['Model Wrong','Model Correct']
    ct_pct.columns = ['GK Not Guessed','GK Guessed']
    sns.heatmap(ct_pct, annot=True, fmt='.1f', cmap='Blues',
                ax=ax, cbar_kws={'label':'%'}, linewidths=0.5)
    ax.set_title('Contingency Table (%)\nModel vs GK', fontsize=11, fontweight='bold')

    fig.suptitle('Layer 2 -- Does the GK guess more when the kick is biomechanically "readable"?',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig('phase3_conditional.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] phase3_conditional.png")

    return rate_correct, rate_wrong, p_chi


# -- Layer 3 -- Concordance: both correct vs the rest -------------

def layer3_concordance(df, report):
    print("\n" + "="*60)
    print("LAYER 3 -- Concordance")
    print("Do kicks where both (model + GK) are correct have distinct biomechanics?")
    print("="*60)

    df_p2 = df[df['macro_zone'].isin(['Left','Right'])].dropna(subset=['gk_guessed']).copy().reset_index(drop=True)
    X_p2  = prepare_X(df_p2)
    y_p2  = (df_p2['macro_zone'] == 'Left').astype(int).values

    preds = cross_val_predict(build_pipe(), X_p2, y_p2, cv=CV, method='predict')
    df_p2['model_correct'] = (preds == y_p2).astype(int)
    df_p2['gk_correct']    = df_p2['gk_guessed'].astype(int)
    df_p2['both_correct']  = ((df_p2['model_correct']==1) & (df_p2['gk_correct']==1)).astype(int)

    n_both = df_p2['both_correct'].sum()
    n_rest = len(df_p2) - n_both
    print(f"\n  Both correct: n={n_both} ({n_both/len(df_p2)*100:.1f}%)")
    print(f"  Rest:         n={n_rest}")

    key_features = ['delta_torso_torsion_angle','late_mean_kick_hip_angle',
                    'early_mean_trunk_inclination','late_std_kick_knee_angle',
                    'delta_running_speed_kmh']
    key_features = [f for f in key_features if f in df_p2.columns]

    print(f"\n  Biomechanical features in both groups:")
    results = []
    for feat in key_features:
        ga = df_p2[df_p2['both_correct']==1][feat].dropna()
        gb = df_p2[df_p2['both_correct']==0][feat].dropna()
        if len(ga) < 3 or len(gb) < 3: continue
        U, p = mannwhitneyu(ga, gb, alternative='two-sided')
        results.append({'feature':feat, 'median_both':round(float(ga.median()),3),
                        'median_rest':round(float(gb.median()),3), 'p_value':round(p,4)})
        sig = '*' if p < 0.05 else ''
        print(f"    {feat:45s}  both={ga.median():.2f}  rest={gb.median():.2f}  p={p:.4f} {sig}")

    report.append(f"\nLAYER 3 -- Concordance (model + GK both correct)")
    report.append(f"  n both correct: {n_both} ({n_both/len(df_p2)*100:.1f}%)")
    for r in results:
        report.append(f"  {r['feature']}: p={r['p_value']}")


# -- Main ---------------------------------------------------------

def main():
    print('Loading clips_master.csv ...')
    df = pd.read_csv('clips_master.csv')
    print(f'  {df.shape[0]} clips\n')

    report = ["PHASE 3 -- P3: Does the GK behave like a computer?", "="*60]

    imp_p2, imp_p1, rho, p_rho = layer1_spearman(df, report)
    rate_correct, rate_wrong, p_chi = layer2_conditional(df, report)
    layer3_concordance(df, report)

    # Automatic conclusion based on the results of the 3 layers
    report.append(f"\nCONCLUSION P3")
    report.append(f"  Spearman rho={rho:.3f} p={p_rho:.4f}")
    report.append(f"  GK correct/wrong rate: {rate_correct*100:.1f}% vs {rate_wrong*100:.1f}%  p={p_chi:.4f}")

    if   p_rho < 0.05 and p_chi < 0.05: conc = "STRONG: The GK and the model read the same signals AND get the same kicks right."
    elif p_chi < 0.05:                   conc = "PARTIAL: The GK guesses more when the kick is biomechanically readable, but uses different signals than the model."
    elif p_rho < 0.05:                   conc = "PARTIAL: The GK values the same features but doesn't guess more on those kicks."
    else:                                conc = "NEGATIVE: The GK and the model operate independently."

    report.append(f"  -> {conc}")
    with open('phase3_results.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\n{'='*60}")
    print(f"CONCLUSION P3: {conc}")
    print(f"{'='*60}")
    print(f"\n[OK] phase3_results.txt")
    print("[OK] Done!")


if __name__ == '__main__':
    main()
