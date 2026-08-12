"""
bio_analysis.py
===============
In-depth biomechanical analysis in 4 parts:

  1. By dominant foot -- splits right vs left and repeats P2
  2. Correlation between top features -- avoids counting the same signal twice
  3. Profile of the unpredictable kicker -- biomechanics of kicks that are hard to guess
  4. Kruskal-Wallis for the 9 zones -- are there differences between individual zones?

Reads: clips_master.csv

Output:
  - kruskal_9_zones.csv
  - biomedical_correlation_features.png (correlation heatmap of the top 20 P2 features)
  - biomedical_unpredictable_profile.png (unpredictable vs predictable biomechanical profile)

Usage:
  python bio_analysis.py
  python bio_analysis.py --data clips_master.csv
"""

import argparse
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
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
from stats_utils import (feature_columns, is_empty,
                         benjamini_hochberg, multiplicity_summary,
                         print_multiplicity_summary)

sns.set_theme(style='darkgrid', font_scale=1.05)

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.

TOP_N = 20   # top features to use in the correlation and the profile


def cohens_d(a, b):                                        # effect size between two groups
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def mannwhitney_all(df, features, group_col, group_a, group_b):
    ga = df[df[group_col] == group_a]
    gb = df[df[group_col] == group_b]
    rows = []
    for f in features:
        va = ga[f].dropna(); vb = gb[f].dropna()
        if len(va) < 5 or len(vb) < 5: continue
        try: _, p = mannwhitneyu(va, vb, alternative='two-sided')
        except: continue
        rows.append({'feature': f, 'p_value': round(p,6),
                     'cohens_d': round(cohens_d(va,vb),4),
                     'med_a': round(float(va.median()),3),
                     'med_b': round(float(vb.median()),3)})
    rows_df = pd.DataFrame(rows)
    if is_empty(rows_df, "per-feature tests"):
        return rows_df
    return rows_df.sort_values('p_value').reset_index(drop=True)


def section(title):
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}')


def plot_correlation(corr_p2, out_path):
    """Spearman correlation heatmap between the top 20 P2 features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_p2, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={'label': 'Spearman'}, linewidths=0.3)
    ax.set_title('Correlation between Top 20 Features (P2 -- Direction)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def plot_unpredictable_profile(df_gk, sig, out_path, top_n=8):
    """Biomechanical profile -- standardized median (z-score) of unpredictable vs predictable, top N by p-value."""
    top = sig.head(top_n)
    rows = []
    for _, r in top.iterrows():
        vals = df_gk[r['feature']].dropna()
        mu, sd = vals.mean(), vals.std()
        if sd == 0: continue
        rows.append((r['feature'], (r['med_a']-mu)/sd, (r['med_b']-mu)/sd))
    feats, z_unp, z_prd = zip(*rows)
    x, w = np.arange(len(feats)), 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, z_unp, width=w, color='#d62728', alpha=0.85, label='Unpredictable (GK did not guess)')
    ax.bar(x + w/2, z_prd, width=w, color='#2ca02c', alpha=0.85, label='Predictable (GK guessed)')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Standardized median (z-score)', fontsize=11)
    ax.set_title('Biomechanical Profile -- Unpredictable vs Predictable Kicker', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


# -- 1. By dominant foot ------------------------------------------

def analysis_by_foot(df, features):
    section('1. ANALYSIS BY DOMINANT FOOT')
    print('  Repeats P2 (Left vs Right) split by kicking foot')
    print('  -- Natural: right foot kicks to the left (or left foot to the right)')
    print('  -- Crossed: right foot kicks to the right (or left foot to the left)\n')

    for foot, foot_label in [('right','Right Foot'), ('left','Left Foot')]:
        sub = df[df['foot_used'] == foot].copy()
        sub = sub[sub['macro_zone'].isin(['Left','Right'])]
        if len(sub) < 20:
            print(f'  {foot_label}: n={len(sub)} insufficient -- skipping')
            continue

        n_left  = (sub['macro_zone']=='Left').sum()
        n_right = (sub['macro_zone']=='Right').sum()
        print(f'  {foot_label} (n={len(sub)}): Left={n_left}  Right={n_right}')

        res = mannwhitney_all(sub, features, 'macro_zone', 'Left', 'Right')
        sig = res[res['p_value'] < 0.05]
        print(f'    Significant p<0.05: {len(sig)}/{len(res)}')
        if len(sig) > 0:
            print(f'    Top 3:')
            for _, r in sig.head(3).iterrows():
                print(f'      {r["feature"]:40s}  p={r["p_value"]:.5f}  d={r["cohens_d"]:+.3f}')
        print()


# -- 2. Correlation between top features -----------------------------

def correlation_top_features(df, features):
    section('2. CORRELATION BETWEEN TOP FEATURES')
    print('  Identifies redundant features in the P1 and P2 results')

    # Get top features from P1 and P2
    res_p1 = mannwhitney_all(df, features, 'gk_guessed', 1, 0)
    res_p2 = mannwhitney_all(df[df['macro_zone'].isin(['Left','Right'])],
                             features, 'macro_zone', 'Left', 'Right')

    top_p1 = res_p1.head(TOP_N)['feature'].tolist()
    top_p2 = res_p2.head(TOP_N)['feature'].tolist()
    top_all = list(dict.fromkeys(top_p1 + top_p2))   # dedup while keeping order

    # Spearman correlation between the top features
    corr = df[top_all].corr(method='spearman')
    plot_correlation(corr.loc[top_p2, top_p2], 'biomedical_correlation_features.png')

    # Pairs with high correlation (possible redundancy)
    print(f'\n  Pairs with |Spearman| > 0.85 (potentially redundant):')
    found = False
    pairs = set()
    for i, c1 in enumerate(top_all):
        for c2 in top_all[i+1:]:
            if c1 not in corr.columns or c2 not in corr.columns: continue
            r = corr.loc[c1, c2]
            if abs(r) > 0.85:
                key = tuple(sorted([c1,c2]))
                if key not in pairs:
                    pairs.add(key)
                    print(f'    r={r:+.3f}  {c1}')
                    print(f'           {c2}')
                    found = True
    if not found:
        print('No highly correlated pairs -- independent features')

    # Top 5 most independent features (low average correlation with the rest)
    print(f'\n  Top 5 most independent features (P1):')
    for feat in top_p1[:5]:
        if feat not in corr.columns: continue
        others = [f for f in top_p1[:10] if f != feat and f in corr.columns]
        mean_r = corr.loc[feat, others].abs().mean() if others else 0
        print(f'    {feat:45s}  average correlation w/ other top10: {mean_r:.3f}')


# -- 3. Profile of the unpredictable kicker ----------------------------

def unpredictable_kicker_profile(df, features):
    section('3. PROFILE OF THE UNPREDICTABLE KICKER')
    print('  Biomechanics of the kicks the GK did NOT guess')
    print('  (compared with the ones they guessed)\n')

    df_gk = df.dropna(subset=['gk_guessed']).copy()
    not_guessed = df_gk[df_gk['gk_guessed'] == 0]
    guessed     = df_gk[df_gk['gk_guessed'] == 1]

    print(f'  Not guessed: n={len(not_guessed)}  |  Guessed: n={len(guessed)}')

    res = mannwhitney_all(df_gk, features, 'gk_guessed', 0, 1)  # 0=unpredictable vs 1=predictable
    sig = res[res['p_value'] < 0.05]

    if len(sig) > 0:
        plot_unpredictable_profile(df_gk, sig, 'biomedical_unpredictable_profile.png')

    print(f'\n  Features that distinguish unpredictable kicks (p<0.05): {len(sig)}')
    print(f'\n  Profile of the UNPREDICTABLE kick (vs predictable):')
    for _, r in sig.head(10).iterrows():
        direction = 'LOWER' if r['cohens_d'] < 0 else 'HIGHER'
        print(f'    {r["feature"]:45s}  {direction}  (p={r["p_value"]:.5f}  d={r["cohens_d"]:+.3f})')
        print(f'      Unpredictable={r["med_a"]:.2f}  Predictable={r["med_b"]:.2f}')


# -- 4. Kruskal-Wallis for the 9 zones ------------------------------

def kruskal_9_zones(df, features):
    section('4. KRUSKAL-WALLIS -- 9 INDIVIDUAL ZONES')
    print('  Are there biomechanical differences between individual zones?')
    print('  (beyond Left vs Right)\n')

    # Zones with at least 10 clips
    zone_counts = df['target_zone'].value_counts()
    valid_zones = zone_counts[zone_counts >= 10].index.tolist()
    print(f'  Zones with n>=10: {sorted(valid_zones)}')

    df_z = df[df['target_zone'].isin(valid_zones)].copy()

    rows = []
    for feat in features:
        groups = [df_z[df_z['target_zone']==z][feat].dropna().values
                  for z in valid_zones]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 3: continue
        try:
            stat, p = kruskal(*groups)
        except: continue
        rows.append({'feature': feat, 'H': round(stat,3), 'p_value': round(p,6)})

    res = pd.DataFrame(rows)
    if is_empty(res, "correlation screen"):
        return res
    res = res.sort_values('p_value').reset_index(drop=True)
    reject, q = benjamini_hochberg(res['p_value'].values, alpha=0.05)
    res['q_value'] = np.round(q, 5)
    res['sig_fdr'] = reject
    sig = res[res['sig_fdr']]

    print()
    print_multiplicity_summary(multiplicity_summary(
        res['p_value'].values, label='Kruskal-Wallis, 9 zones'))
    print(f'\n  Top 10 (Kruskal-Wallis H):')
    for _, r in res.head(10).iterrows():
        flag = ' *' if r['sig_fdr'] else ''
        print(f'    {r["feature"]:45s}  H={r["H"]:7.2f}  p={r["p_value"]:.5f}  '
              f'q={r["q_value"]:.4f}{flag}')

    if len(sig) > 0:
        # For the most discriminative feature, show medians per zone
        best = res.iloc[0]['feature']
        print(f'\n  Medians per zone ({best}):')
        for z in sorted(valid_zones, key=lambda x: str(x)):
            vals = df_z[df_z['target_zone']==z][best].dropna()
            print(f'    Zone {z:3s}: median={vals.median():.2f}  n={len(vals)}')

    res.to_csv('kruskal_9_zones.csv', index=False)
    print(f'\n[OK] kruskal_9_zones.csv')


# -- Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='In-depth biomechanical analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--data', '-d', default='clips_master.csv')
    args = ap.parse_args()

    print(f'Loading {args.data} ...')
    df = pd.read_csv(args.data)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    features = feature_columns(df)
    print()

    analysis_by_foot(df, features)
    correlation_top_features(df, features)
    unpredictable_kicker_profile(df, features)
    kruskal_9_zones(df, features)

    print('\n[OK] Done!')


if __name__ == '__main__':
    main()
