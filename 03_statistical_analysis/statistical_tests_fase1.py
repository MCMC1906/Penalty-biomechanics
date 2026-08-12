"""
statistical_tests_fase1.py
==========================
Tests all biomechanical features against three targets:
  - P1: gk_guessed  (0 vs 1)        - Mann-Whitney U
  - P2: macro_zone  (Left vs Right) - Mann-Whitney U  (excludes Center and Out)
  - P3: outcome     (goal vs not)   - Mann-Whitney U

For each feature computes:
  - U statistic and p-value (Mann-Whitney)
  - Cohen's d (effect size)
  - Median of each group
  - Three significance flags: uncorrected p < 0.05, Bonferroni, and
    Benjamini-Hochberg FDR (q-value)

On multiple testing
-------------------
250 features x 3 targets is 750 tests. At alpha = 0.05, 250 tests produce
about 12.5 "significant" results by chance alone, so an uncorrected count is
close to meaningless on its own -- for P3 the observed count has been BELOW
the chance expectation, which means there is nothing there to explain. Every
results table now prints the expected-by-chance count next to the observed
one, and both corrections next to that.

Output:
  - results_P1_gk_guessed.csv
  - results_P2_macro_zone.csv
  - results_P3_outcome.csv
  - phase1_top_features_P1.png (top 15 by Cohen's d, colored by phase)
  - phase1_top_features_P2.png

Usage:
  python statistical_tests_fase1.py
  python statistical_tests_fase1.py --data clips_master.csv
"""

import argparse
import numpy as np
import pandas as pd
from stats_utils import (benjamini_hochberg, multiplicity_summary,
                         print_multiplicity_summary, feature_columns, is_empty)
import matplotlib
matplotlib.use('Agg')                                      # windowless backend -- avoids the tkinter error on Windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

sns.set_theme(style='darkgrid', font_scale=1.05)

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.

PHASE_COLORS = {'Early': '#1f77b4', 'Mid': '#ff7f0e', 'Late': '#2ca02c', 'Other': '#7f7f7f'}

def feature_phase(feat):                                   # infers the phase from the feature prefix
    if feat.startswith('early_'): return 'Early'
    if feat.startswith('mid_'):   return 'Mid'
    if feat.startswith('late_'):  return 'Late'
    return 'Other'                                         # contact_/delta_/others without a phase prefix


def cohens_d(g1, g2):                                      # effect size between two groups
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0


def effect_label(d):                                       # Cohen's classification for the effect size
    a = abs(d)
    if a < 0.2: return 'negligible'
    if a < 0.5: return 'small'
    if a < 0.8: return 'medium'
    return 'large'


def run_mannwhitney(df, features, group_col, group_a, group_b, label_a, label_b):
    """Runs Mann-Whitney for every feature between two groups. Returns a DataFrame sorted by p-value."""
    ga = df[df[group_col] == group_a]
    gb = df[df[group_col] == group_b]
    n_feats = len(features)                                # used for the Bonferroni threshold -- total features attempted (before the n>=5 per-group filter), not only the ones successfully tested

    rows = []
    for feat in features:
        va = ga[feat].dropna()
        vb = gb[feat].dropna()
        if len(va) < 5 or len(vb) < 5: continue          # minimum of 5 observations per group
        try:
            U, p = mannwhitneyu(va, vb, alternative='two-sided')
        except Exception:
            continue
        d   = cohens_d(va, vb)
        rows.append({
            'feature'            : feat,
            f'n_{label_a}'       : len(va),
            f'n_{label_b}'       : len(vb),
            f'median_{label_a}'  : round(float(va.median()), 4),
            f'median_{label_b}'  : round(float(vb.median()), 4),
            'median_diff'        : round(float(va.median() - vb.median()), 4),
            'U'                  : round(U, 1),
            'p_value'            : round(p, 6),
            'cohens_d'           : round(d, 4),
            'effect'             : effect_label(d),
            'sig_p05'            : 'Yes' if p < 0.05 else 'No',
            'sig_bonferroni'     : 'Yes' if p < (0.05 / n_feats) else 'No',  # controls ANY false positive
        })

    rows_df = pd.DataFrame(rows)
    if is_empty(rows_df, "per-feature Mann-Whitney"):
        # Every group fell below the minimum of 5 observations, so no test ran.
        # This used to reach sort_values() and raise KeyError, which told the
        # reader nothing about why.
        return rows_df
    out = rows_df.sort_values('p_value').reset_index(drop=True)

    # Benjamini-Hochberg: controls the expected PROPORTION of false positives
    # among those called significant. Less strict than Bonferroni and the more
    # appropriate correction when screening many features.
    reject, q = benjamini_hochberg(out['p_value'].values, alpha=0.05)
    out['q_value'] = np.round(q, 5)
    out['sig_fdr'] = np.where(reject, 'Yes', 'No')

    out.index += 1                                         # start at 1 instead of 0
    return out


def print_results(res, label_a, label_b, title):
    summary = multiplicity_summary(res['p_value'].values, alpha=0.05, label=title)
    print()
    print_multiplicity_summary(summary)
    print(f"\n  TOP 10 features (lowest p-value):")
    print(res[['feature', 'p_value', 'q_value', 'cohens_d', 'effect',
               f'median_{label_a}', f'median_{label_b}']].head(10).to_string())
    max_d = res['cohens_d'].abs().max()
    if summary['n_sig_fdr_bh'] == 0:
        print(f"\n  Verdict: no feature survives correction. Largest effect size "
              f"|d| = {max_d:.2f}"
              + ("  (negligible)" if max_d < 0.2 else
                 "  (small)" if max_d < 0.5 else ""))
        print("  The honest reading is a null result for this target, not a "
              "ranked list of findings.")
    return summary


def plot_top_features(res, out_path, title):
    """Top 15 features by |Cohen's d|, horizontal bars colored by phase, * = survives Bonferroni."""
    top = res.reindex(res['cohens_d'].abs().sort_values(ascending=False).index).head(15).iloc[::-1]
    colors = top['feature'].apply(feature_phase).map(PHASE_COLORS)
    labels = [f'{f}  *' if b == 'Yes' else f for f, b in zip(top['feature'], top['sig_bonferroni'])]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top)), top['cohens_d'], color=colors, alpha=0.85)
    for bar, val in zip(bars, top['cohens_d']):
        ax.text(val + (0.02 if val >= 0 else -0.02), bar.get_y()+bar.get_height()/2,
                f'{val:+.2f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title(f'{title}\nTop 15 features by effect size  (* = survives Bonferroni)',
                 fontsize=12, fontweight='bold')
    handles = [plt.Rectangle((0,0),1,1, color=c, alpha=0.85) for c in PHASE_COLORS.values()]
    ax.legend(handles, PHASE_COLORS.keys(), fontsize=9, loc='lower right', title='Phase')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='clips_master.csv')
    args = ap.parse_args()

    print(f'Loading {args.data} ...')
    df = pd.read_csv(args.data)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    features = feature_columns(df)
    print(f'  {len(features)} features to test '
          f'(~{len(features) * 0.05:.0f} will look significant by chance alone)\n')

    # -- P1: gk_guessed -------------------------------------------
    print("=" * 60)
    print("P1 -- gk_guessed: Guessed (1) vs Not Guessed (0)")
    print("=" * 60)
    res_p1 = run_mannwhitney(df, features, 'gk_guessed', 1, 0, 'Guessed', 'NotGuessed')
    print_results(res_p1, 'Guessed', 'NotGuessed', 'P1')
    res_p1.to_csv('results_P1_gk_guessed.csv', index=True)
    print(f"\n[OK] results_P1_gk_guessed.csv")
    plot_top_features(res_p1, 'phase1_top_features_P1.png', 'P1 -- Predictability (gk_guessed)')

    # -- P2: macro_zone -------------------------------------------
    print("\n" + "=" * 60)
    print("P2 -- macro_zone: Left vs Right (excludes Center and Out)")
    print("=" * 60)
    res_p2 = run_mannwhitney(df, features, 'macro_zone', 'Left', 'Right', 'Left', 'Right')
    print_results(res_p2, 'Left', 'Right', 'P2')
    res_p2.to_csv('results_P2_macro_zone.csv', index=True)
    print(f"\n[OK] results_P2_macro_zone.csv")
    plot_top_features(res_p2, 'phase1_top_features_P2.png', 'P2 -- Direction (Left vs Right)')

    # -- P3: outcome (goal vs no goal) ---------------------------
    print("\n" + "=" * 60)
    print("P3 -- outcome: Goal vs No Goal")
    print("=" * 60)
    df_out = df.dropna(subset=['outcome']).copy()
    df_out['is_goal'] = (df_out['outcome'] == 'goal').astype(int)  # binary column for the test
    res_p3 = run_mannwhitney(df_out, features, 'is_goal', 1, 0, 'Goal', 'NoGoal')
    print(f"  Goal:   {(df_out['is_goal']==1).sum()}")
    print(f"  NoGoal: {(df_out['is_goal']==0).sum()}")
    print_results(res_p3, 'Goal', 'NoGoal', 'P3')
    res_p3.to_csv('results_P3_outcome.csv', index=True)
    print(f"\n[OK] results_P3_outcome.csv")

    # -- Final summary ---------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = []
    for label, res in [('P1 (gk_guessed)', res_p1),
                       ('P2 (Left vs Right)', res_p2),
                       ('P3 (goal vs not)', res_p3)]:
        summ = multiplicity_summary(res['p_value'].values, alpha=0.05, label=label)
        rows.append(summ)
        print(f"\n  {label}:")
        print(f"    p < 0.05 uncorrected : {summ['n_sig_uncorrected']}"
              f"   (expected by chance: {summ['expected_by_chance']},"
              f" excess {summ['excess_over_chance']:+})")
        print(f"    survives Bonferroni  : {summ['n_sig_bonferroni']}")
        print(f"    survives BH-FDR      : {summ['n_sig_fdr_bh']}"
              f"   (smallest q = {summ['min_q_value']})")
        surv = res[res['sig_fdr'] == 'Yes']
        if len(surv):
            best = surv.iloc[0]
            print(f"    Best: {best['feature']}  (p={best['p_value']:.5f}  "
                  f"q={best['q_value']:.4f}  d={best['cohens_d']:.3f})")
        else:
            print(f"    Nothing survives correction -- null result for this target.")

    pd.DataFrame(rows).to_csv('multiplicity_summary.csv', index=False)
    print("\n[OK] multiplicity_summary.csv")

    if all(r['n_sig_fdr_bh'] == 0 for r in rows):
        print("\n  Overall: no individual biomechanical feature separates any of the")
        print("  three targets once multiple testing is accounted for. That is a")
        print("  result, and it is the one this analysis supports.")

    print("\n[OK] Done!")


if __name__ == '__main__':
    main()
