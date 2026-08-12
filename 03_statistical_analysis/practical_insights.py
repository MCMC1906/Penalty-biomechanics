"""
practical_insights.py
=====================
Practical analysis in 3 parts:

  1. Natural vs Crossed
     Goal rate and GK anticipation rate for each kick type.
     Natural = right foot to the right (or left foot to the left)
     Crossed = right foot to the left (or left foot to the right)

  2. Anticipation frame
     At what frame does the biomechanical information become reliable?
     Uses biomechanics_dataset_complete.csv (per frame).

  3. Early vs Mid vs Late
     What time phase holds the signal for P1 and P2?
     Practical guide for GK training.

Reads:
  clips_master.csv
  biomechanics_dataset_complete.csv  (for the anticipation frame)

Output:
  - natural_vs_crossed.csv
  - frame_anticipation.csv
  - practical_frame_anticipation.png (PR-AUC by frame + baseline + 1st significant frame)
  - practical_natural_crossed.png (goal rate and GK anticipation rate, Natural vs Crossed)

Usage:
  python practical_insights.py
  python practical_insights.py --master clips_master.csv --frames biomechanics_dataset_complete.csv
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
from scipy.stats import mannwhitneyu, chi2_contingency
from stats_utils import (benjamini_hochberg, multiplicity_summary,
                         print_multiplicity_summary, feature_columns,
                         is_empty, shot_type_series)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

sns.set_theme(style='darkgrid', font_scale=1.05)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.


def section(title):
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}')


def cohens_d(a, b):                                        # effect size between two groups
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def rf_prauc(X, y, k=15):                                  # PR-AUC with cross-validation
    pipe = Pipeline([('imp', KNNImputer(n_neighbors=5)),
                     ('sc',  StandardScaler()),
                     ('sel', SelectKBest(f_classif, k=min(k,X.shape[1]))),
                     ('clf', RandomForestClassifier(n_estimators=200, max_depth=4,
                             min_samples_leaf=5, class_weight='balanced',
                             random_state=42, n_jobs=-1))])
    scores = []
    for tr, te in CV.split(X, y):
        m = clone(pipe); m.fit(X.iloc[tr], y[tr])
        scores.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:,1]))
    return np.mean(scores), np.std(scores)


def plot_natural_crossed(goal_nat, goal_cru, p_goal, guess_nat, guess_cru, p_gk, out_path):
    """Goal rate and GK anticipation rate, Natural vs Crossed side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (v_nat, v_cru, p, ylabel, title) in zip(axes, [
        (goal_nat*100, goal_cru*100, p_goal, '% Goal', 'Goal Rate'),
        (guess_nat*100, guess_cru*100, p_gk, '% GK Guessed', 'GK Anticipation Rate'),
    ]):
        bars = ax.bar(['Natural','Crossed'], [v_nat, v_cru], color=['#1f77b4','#ff7f0e'], alpha=0.85, width=0.5)
        for bar, v in zip(bars, [v_nat, v_cru]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
                    ha='center', fontsize=12, fontweight='500')
        ax.set_ylim(0, 100); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}\np={p:.3f}', fontsize=11, fontweight='bold')
    fig.suptitle('Natural vs Crossed', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def plot_frame_antecipacao(res, baseline, first_frame, out_path):
    """PR-AUC by frame (0-19) with baseline and marking of the 1st significant frame."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res['frame'], res['pr_auc'], marker='o', color='#1f77b4', lw=2, ms=5)
    ax.axhline(baseline, color='red', ls='--', lw=1.5, alpha=0.7, label=f'Baseline ({baseline:.3f})')
    if first_frame is not None:
        ax.axvline(first_frame, color='#2ca02c', ls=':', lw=1.5, alpha=0.8,
                   label=f'1st significant frame (frame {int(first_frame)})')
    ax.set_xticks(range(0, 20))
    ax.set_xlabel('Frame (0 = start of clip, 19 = contact)', fontsize=11)
    ax.set_ylabel('PR-AUC (5-fold CV)', fontsize=11)
    ax.set_title('Anticipation Frame -- Kick Direction (P2)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


# -- 1. Natural vs Crossed ----------------------------------------

def analyze_natural_crossed(df):
    section('1. NATURAL vs CROSSED')

    # Natural: right foot kicks to the right OR left foot to the left
    # Crossed: the opposite
    # Label from the shared definition (stats_utils), not a local copy.
    df = df.copy()
    df['shot_type'] = shot_type_series(df)
    df = df[df['shot_type'].notna()].copy()

    nat = df[df['shot_type'] == 'Natural']
    cru = df[df['shot_type'] == 'Crossed']
    print(f'\n  Natural: n={len(nat)}  |  Crossed: n={len(cru)}')

    # Goal rate
    if 'outcome' in df.columns:
        goal_nat = (nat['outcome'] == 'goal').mean()
        goal_cru = (cru['outcome'] == 'goal').mean()
        ct_goal  = pd.crosstab(df['shot_type'], df['outcome'] == 'goal')
        try:
            _, p_goal, _, _ = chi2_contingency(ct_goal)
        except: p_goal = 1.0
        print(f'\n  Goal rate:')
        print(f'    Natural : {goal_nat*100:.1f}%')
        print(f'    Crossed : {goal_cru*100:.1f}%')
        print(f'    p={p_goal:.4f}  {"[OK] significant" if p_goal < 0.05 else "not significant"}')

    # GK anticipation rate
    if 'gk_guessed' in df.columns:
        df_gk = df.dropna(subset=['gk_guessed'])
        nat_gk = df_gk[df_gk['shot_type']=='Natural']
        cru_gk = df_gk[df_gk['shot_type']=='Crossed']
        guess_nat = nat_gk['gk_guessed'].astype(int).mean()
        guess_cru = cru_gk['gk_guessed'].astype(int).mean()
        ct_gk = pd.crosstab(df_gk['shot_type'], df_gk['gk_guessed'].astype(int))
        try:
            _, p_gk, _, _ = chi2_contingency(ct_gk)
        except: p_gk = 1.0
        print(f'\n  GK guessed:')
        print(f'    Natural : {guess_nat*100:.1f}%')
        print(f'    Crossed : {guess_cru*100:.1f}%')
        print(f'    p={p_gk:.4f}  {"[OK] significant" if p_gk < 0.05 else "not significant"}')

    if 'outcome' in df.columns and 'gk_guessed' in df.columns:
        plot_natural_crossed(goal_nat, goal_cru, p_goal, guess_nat, guess_cru, p_gk,
                             'practical_natural_crossed.png')

    # Features that differentiate Natural vs Crossed
    feats = feature_columns(df)
    rows  = []
    for f in feats:
        va = nat[f].dropna(); vb = cru[f].dropna()
        if len(va) < 5 or len(vb) < 5: continue
        try: _, p = mannwhitneyu(va, vb, alternative='two-sided')
        except: continue
        rows.append({'feature': f, 'p': round(p,6), 'd': round(cohens_d(va,vb),4),
                     'nat': round(float(va.median()),3), 'cru': round(float(vb.median()),3)})
    res = pd.DataFrame(rows)
    if is_empty(res, "frame-level timeline"):
        return res
    res = res.sort_values('p')
    reject, q = benjamini_hochberg(res['p'].values, alpha=0.05)
    res['q'] = np.round(q, 5)
    res['sig_fdr'] = reject
    print()
    print_multiplicity_summary(multiplicity_summary(
        res['p'].values, label='Natural vs Crossed features'))
    sig = res[res['sig_fdr']]
    if len(sig):
        print(f'  Top 5 surviving FDR:')
        for _, r in sig.head(5).iterrows():
            print(f'    {r["feature"]:45s}  p={r["p"]:.5f}  q={r["q"]:.4f}  '
                  f'd={r["d"]:+.3f}  Nat={r["nat"]:.2f}  Cru={r["cru"]:.2f}')

    res.to_csv('natural_vs_crossed.csv', index=False)
    print(f'\n[OK] natural_vs_crossed.csv')


# -- 2. Anticipation frame --------------------------------------

def frame_antecipacao(df_frames):
    section('2. ANTICIPATION FRAME')
    print('  At what frame does the biomechanics become reliable for predicting the direction?')
    print('  (frame 0 = start of clip  |  frame 19 = ball contact)\n')

    if 'macro_zone' not in df_frames.columns:
        print('  [WARNING] macro_zone not available -- skipping'); return

    df_frames = df_frames[df_frames['macro_zone'].isin(['Left','Right'])].copy()

    bio_cols = ['kick_knee_angle','kick_hip_angle','kick_angular_vel',
                'running_speed_kmh','trunk_inclination','torso_torsion_angle',
                'lateral_trunk_lean','mean_visibility_score']
    bio_cols = [c for c in bio_cols if c in df_frames.columns]

    print(f'  {"Frame":>6s}  {"PR-AUC":>8s}  {"vs baseline":<12s}')
    print(f'  {"-"*6}  {"-"*8}  {"-"*12}')

    baseline = 0.534
    results = []
    for frame in range(20):
        sub = df_frames[df_frames['frame'] == frame].copy()
        if len(sub) < 50: continue
        y = (sub['macro_zone'] == 'Left').astype(int).values
        valid = [c for c in bio_cols if sub[c].notna().any()]
        if not valid: continue
        imp = KNNImputer(n_neighbors=5)
        X   = pd.DataFrame(imp.fit_transform(sub[valid]), columns=valid)
        prauc, _ = rf_prauc(X, y, k=min(8, len(valid)))
        flag = '[OK]' if prauc > baseline else '    '
        print(f'  {frame:>6d}  {prauc:>8.3f}  {flag}')
        results.append({'frame': frame, 'pr_auc': prauc})

    res = pd.DataFrame(results)
    if len(res):
        first_sig = res[res['pr_auc'] > baseline]
        if len(first_sig):
            f_sig = first_sig.iloc[0]['frame']
            fps   = 25.0
            ms    = (19 - f_sig) / fps * 1000
            print(f'\n  First frame above baseline: frame {int(f_sig)}')
            print(f'  Equivalent to {ms:.0f}ms before contact')
        res.to_csv('frame_anticipation.csv', index=False)
        print(f'[OK] frame_anticipation.csv')
        plot_frame_antecipacao(res, baseline, f_sig if len(first_sig) else None,
                               'practical_frame_anticipation.png')


# -- 3. Early vs Mid vs Late (practical guide) -----------------------

def early_mid_late(df):
    section('3. EARLY vs MID vs LATE -- practical guide for GK training')

    phases = {'early': 'early_mean_', 'mid': 'mid_mean_', 'late': 'late_mean_'}
    targets = {
        'P2 Direction'   : ('macro_zone', 'Left', 'Right', 0.534),
        'P1 GK Predict.' : ('gk_guessed', 1, 0, 0.464),
    }

    print(f'\n  {"Target":20s}  {"Phase":8s}  {"PR-AUC":>8s}  {"Sig p<0.05":>10s}')
    print(f'  {"-"*20}  {"-"*8}  {"-"*8}  {"-"*10}')

    for tname, (gcol, ga, gb, baseline) in targets.items():
        sub = df[df[gcol].isin([ga, gb])].copy()
        y   = (sub[gcol] == ga).astype(int).values
        for pname, prefix in phases.items():
            feats = [c for c in feature_columns(df, verbose=False)
                     if c.startswith(prefix)]
            if not feats: continue
            valid = [f for f in feats if sub[f].notna().any()]
            imp   = KNNImputer(n_neighbors=5)
            X     = pd.DataFrame(imp.fit_transform(sub[valid]), columns=valid)
            prauc, _ = rf_prauc(X, y)

            # significant features in this phase, FDR-corrected within the phase
            pvals, names = [], []
            for f in valid:
                va = sub[sub[gcol]==ga][f].dropna()
                vb = sub[sub[gcol]==gb][f].dropna()
                if len(va)<5 or len(vb)<5: continue
                try: _, p = mannwhitneyu(va, vb, alternative='two-sided')
                except: continue
                pvals.append(p); names.append(f)
            n_fdr = int(benjamini_hochberg(pvals, alpha=0.05)[0].sum()) if pvals else 0
            flag = '[OK]' if prauc > baseline else '    '
            print(f'  {tname:20s}  {pname:8s}  {prauc:>8.3f}  {n_fdr:>10d}  {flag}')

    print(f'\n  Reading this table:')
    print(f'    The counts above are features surviving FDR correction within each')
    print(f'    phase, not raw p<0.05 counts. Phase-level PR-AUC differences of a')
    print(f'    few hundredths are within fold noise at n=414 -- do not build a')
    print(f'    training recommendation on which phase scores marginally higher')
    print(f'    unless the gap survives repeated CV (see 04_models/eval_utils.py).')


# -- Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Practical analysis: Natural vs Crossed, anticipation frame, early/mid/late.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--master', '-m', default='clips_master.csv')
    ap.add_argument('--frames', '-f', default='biomechanics_dataset_complete.csv')
    args = ap.parse_args()

    print(f'Loading {args.master} ...')
    df = pd.read_csv(args.master)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    analyze_natural_crossed(df)
    early_mid_late(df)

    # Anticipation frame (requires the per-frame file)
    import os
    if os.path.exists(args.frames):
        print(f'\nLoading {args.frames} ...')
        df_frames = pd.read_csv(args.frames)
        # Merge macro_zone from the master
        meta = df[['clip_id','macro_zone']].drop_duplicates()
        df_frames = df_frames.merge(meta, on='clip_id', how='left')
        for c in ['kick_knee_angle','kick_hip_angle','kick_angular_vel',
                  'running_speed_kmh','trunk_inclination','torso_torsion_angle',
                  'lateral_trunk_lean','mean_visibility_score']:
            if c in df_frames.columns:
                df_frames[c] = pd.to_numeric(df_frames[c], errors='coerce')
        frame_antecipacao(df_frames)
    else:
        print(f'\n[WARNING] {args.frames} not found -- skipping anticipation frame')

    print('\n[OK] Done!')


if __name__ == '__main__':
    main()
