"""
reg_cros.py
===========
Statistically validates whether real biomechanical differences exist between
Natural and Crossed kicks -- complements the ML results in 04_models/
with independent statistical evidence (Mann-Whitney, Cohen's d, chi-squared).

Natural / Crossed (defined once in stats_utils.build_shot_type()):
  Natural : right foot -> Right   OR   left foot -> Left
  Crossed : right foot -> Left    OR   left foot -> Right

Left/Right are goal zones as seen by the kicker facing the goal, so Crossed is
the leg swinging across the body midline.

Tests:
  1. Mann-Whitney U + Cohen's d for every numeric feature
     (same pattern as statistical_tests_fase1.py)
  2. Top 10 most discriminative features, with biomechanical interpretation
  3. Significance timeline by frame (0-19), same pattern as
     practical_insights.py -- at what frame does each feature become significant?
  4. Natural vs Crossed split by dominant foot -- is the pattern symmetric?
  5. Goal rate Natural vs Crossed (chi-squared)

Reads:
  clips_master.csv                 (tests 1, 2, 4, 5)
  biomechanics_dataset_complete.csv   (test 3 -- per-frame timeline)

Output:
  natural_crossed_statistical.csv
  natural_crossed_timeline_stats.csv
  natural_crossed_stats_report.txt
  natural_crossed_timeline.png (p-value by frame, top 5 features, line at p=0.05)

Usage:
  python reg_cros.py
  python reg_cros.py --master clips_master.csv --frames biomechanics_dataset_complete.csv
"""

import argparse
import os
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
from stats_utils import (feature_columns, is_empty, build_shot_type,
                         benjamini_hochberg, multiplicity_summary,
                         print_multiplicity_summary)

sns.set_theme(style='darkgrid', font_scale=1.05)

# Feature selection and the Natural/Crossed label are defined once, in
# stats_utils.py, and imported. Each script in this folder used to carry its own
# copy of EXCLUDE plus an ad-hoc "'visibility' not in c" filter, and they
# drifted: when the signed lateral features were excluded from the models, all
# six kept testing them, which is why the statistical tests screened 235
# features while the model scripts screened 219.

# -- Biomechanical interpretation of the base features -------------------
BASE_INTERP = {
    'kick_knee_angle'        : 'Knee angle of the kicking leg',
    'supp_knee_angle'        : 'Knee angle of the support leg',
    'kick_ankle_angle'       : 'Ankle angle of the kicking leg',
    'supp_ankle_angle'       : 'Ankle angle of the support leg',
    'kick_hip_angle'         : 'Hip angle of the kicking leg',
    'kick_hip_angular_vel'   : 'Angular velocity of the kicking hip',
    'kick_angular_vel'       : 'Angular velocity of the kicking knee',
    'kick_ankle_angular_vel' : 'Angular velocity of the kicking ankle',
    'kick_angular_accel'     : 'Angular acceleration of the kicking knee (joint whip)',
    'kick_foot_speed_ms'     : 'Linear speed of the kicking foot',
    'running_speed_kmh'      : 'Approach run speed',
    'trunk_inclination'      : 'Trunk inclination relative to vertical',
    'lateral_trunk_lean'     : 'Lateral trunk lean',
    'torso_torsion_angle'    : 'Torso torsion (shoulders vs hips)',
    'left_elbow_angle'       : 'Left elbow angle',
    'right_elbow_angle'      : 'Right elbow angle',
    'mean_visibility_score'  : 'Average MediaPipe confidence',
    'knee_stability_index'   : 'Support-knee stability (standard deviation)',
    'elbow_asymmetry'        : 'Asymmetry between both elbows at contact',
    'max_running_speed_kmh'  : 'Maximum running speed in the clip',
    'support_foot_distance'  : 'Distance from the support foot to the ball',
    'support_foot_angle_deg' : 'Angle of the support foot relative to the ball',
    'deception_torsion'      : 'Variation in torso torsion (deception proxy)',
    'deception_lateral'      : 'Variation in lateral trunk lean (deception proxy)',
    'deception_trunk'        : 'Variation in trunk inclination (deception proxy)',
    'deception_score'        : 'Overall deception score (average of the 3 variations above)',
    'peak_hip_vel'           : 'Peak angular velocity of the hip',
    'peak_knee_vel'          : 'Peak angular velocity of the knee',
    'peak_ankle_vel'         : 'Peak angular velocity of the ankle',
    'dt_hip_to_knee'         : 'Time interval between the hip and knee peaks',
    'dt_knee_to_ankle'       : 'Time interval between the knee and ankle peaks',
    'vel_ratio_hip_knee'     : 'Velocity-transfer ratio hip->knee',
    'vel_ratio_knee_ankle'   : 'Velocity-transfer ratio knee->ankle',
}

PHASE_INTERP = {                                           # prefix -- phase description (same phases as build_clips_master.py)
    'contact_'   : 'at the contact frame (T0)',
    'early_mean_': 'on average in the early phase (frames 0-6, approach run)',
    'early_std_' : 'in the standard deviation of the early phase (frames 0-6)',
    'early_max_' : 'at the maximum of the early phase (frames 0-6)',
    'early_min_' : 'at the minimum of the early phase (frames 0-6)',
    'mid_mean_'  : 'on average in the mid phase (frames 7-13, setup/loading)',
    'mid_std_'   : 'in the standard deviation of the mid phase (frames 7-13)',
    'mid_max_'   : 'at the maximum of the mid phase (frames 7-13)',
    'mid_min_'   : 'at the minimum of the mid phase (frames 7-13)',
    'late_mean_' : 'on average in the late phase (frames 14-19, swing and impact)',
    'late_std_'  : 'in the standard deviation of the late phase (frames 14-19)',
    'late_max_'  : 'at the maximum of the late phase (frames 14-19)',
    'late_min_'  : 'at the minimum of the late phase (frames 14-19)',
    'delta_'     : 'in the late-early evolution (how much it changes from start to end of the clip)',
}


def section(title):
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}')


def cohens_d(a, b):                                        # effect size between two groups
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*a.var(ddof=1) + (n2-1)*b.var(ddof=1)) / (n1+n2-2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def effect_label(d):                                       # Cohen's classification for the effect size
    a = abs(d)
    if a < 0.2: return 'negligible'
    if a < 0.5: return 'small'
    if a < 0.8: return 'medium'
    return 'large'


# The Natural/Crossed rule now lives in stats_utils.build_shot_type(), so this
# script, practical_insights.py and 04_models/reg_cross_analysis.py cannot drift
# apart again. They had: for a while 04_models called "Natural" what these two
# called "Crossed", which left every interpretive sentence and every class count
# pointing the wrong way between folders.


def interpret_feature(feat):                               # generates a readable biomechanical description from the feature name
    for prefix, phase_txt in PHASE_INTERP.items():
        if feat.startswith(prefix):
            base = feat[len(prefix):]
            base_txt = BASE_INTERP.get(base, base.replace('_', ' '))
            return f'{base_txt} -- {phase_txt}'
    return BASE_INTERP.get(feat, feat.replace('_', ' '))


# -- 1. Mann-Whitney + Cohen's d for every feature -------------

def run_mannwhitney(df, features, group_col, group_a, group_b):
    """Runs Mann-Whitney for every feature between Natural and Crossed. Returns a DataFrame sorted by p-value."""
    ga = df[df[group_col] == group_a]
    gb = df[df[group_col] == group_b]
    n_feats = len(features)                                # used for the Bonferroni threshold -- total features attempted (before the n>=5 per-group filter)

    rows = []
    for feat in features:
        va = ga[feat].dropna()
        vb = gb[feat].dropna()
        if len(va) < 5 or len(vb) < 5: continue          # minimum of 5 observations per group
        try:
            U, p = mannwhitneyu(va, vb, alternative='two-sided')
        except Exception:
            continue
        d = cohens_d(va, vb)
        rows.append({
            'feature'            : feat,
            'n_natural'          : len(va),
            'n_crossed'          : len(vb),
            'median_natural'     : round(float(va.median()), 4),
            'median_crossed'     : round(float(vb.median()), 4),
            'median_diff'        : round(float(va.median() - vb.median()), 4),
            'U'                  : round(U, 1),
            'p_value'            : round(p, 6),
            'cohens_d'           : round(d, 4),
            'effect'             : effect_label(d),
            'sig_p05'            : 'Yes' if p < 0.05 else 'No',
            'sig_bonferroni'     : 'Yes' if p < (0.05 / n_feats) else 'No',  # controls ANY false positive
        })

    rows_df = pd.DataFrame(rows)
    if is_empty(rows_df, "Natural vs Crossed, per-clip features"):
        # Every group fell below the minimum of 5 observations, so no test ran.
        # This used to reach sort_values() and raise KeyError, which told the
        # reader nothing about why.
        return rows_df
    out = rows_df.sort_values('p_value').reset_index(drop=True)

    reject, q = benjamini_hochberg(out['p_value'].values, alpha=0.05)
    out['q_value'] = np.round(q, 5)
    out['sig_fdr'] = np.where(reject, 'Yes', 'No')

    print_multiplicity_summary(multiplicity_summary(
        out['p_value'].values, label='Natural vs Crossed, per-clip features'))

    out.index += 1                                         # start at 1 instead of 0
    return out


# -- 2. Top 10 with biomechanical interpretation -------------------------

def top10_with_interpretation(res, report):
    """Lists the 10 most discriminative features with biomechanical interpretation."""
    section('2. TOP 10 MOST DISCRIMINATIVE FEATURES')

    report.append('\nTOP 10 FEATURES -- Natural vs Crossed')
    for i, r in res.head(10).iterrows():
        interp    = interpret_feature(r['feature'])
        direction = 'HIGHER in Natural' if r['median_diff'] > 0 else 'HIGHER in Crossed'
        print(f"  {i:2d}. {r['feature']:40s}  p={r['p_value']:.5f}  d={r['cohens_d']:+.3f} ({r['effect']})")
        print(f"      {interp}")
        print(f"      {direction}  (Natural={r['median_natural']:.2f}  Crossed={r['median_crossed']:.2f})")
        report.append(f"  {i}. {r['feature']}  p={r['p_value']:.5f}  d={r['cohens_d']:+.3f} ({r['effect']})")
        report.append(f"     {interp}")
        report.append(f"     {direction}  (Natural={r['median_natural']:.2f}  Crossed={r['median_crossed']:.2f})")


# -- 3. Significance timeline by frame --------------------------

def plot_timeline(res_timeline, out_path, top_n=5):
    """p-value by frame for the top N features (lowest minimum p-value across frames), line at p=0.05."""
    best_p = res_timeline.groupby('feature')['p_value'].min().sort_values()
    top_feats = best_p.head(top_n).index.tolist()
    palette = sns.color_palette('deep', len(top_feats))

    fig, ax = plt.subplots(figsize=(11, 6))
    for feat, color in zip(top_feats, palette):
        sub = res_timeline[res_timeline['feature'] == feat].sort_values('frame')
        ax.plot(sub['frame'], sub['p_value'], marker='o', ms=4, lw=1.8, color=color, label=feat)
    ax.axhline(0.05, color='red', ls='--', lw=1.5, alpha=0.7, label='p = 0.05')
    ax.set_xticks(range(0, 20))
    ax.set_xlabel('Frame (0 = start of clip, 19 = contact)', fontsize=11)
    ax.set_ylabel('p-value (Mann-Whitney)', fontsize=11)
    ax.set_title('Significance Timeline -- Top 5 Features (Natural vs Crossed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def timeline_significance(df_master, df_frames, bio_cols, report):
    """Tests Natural vs Crossed at every frame (0-19) -- at what point does each feature become significant?"""
    section('3. SIGNIFICANCE TIMELINE BY FRAME')
    print('  At what frame does each feature become significant (p<0.05)?')
    print('  (frame 0 = start of clip  |  frame 19 = ball contact)\n')

    meta      = df_master[['clip_id', 'shot_type']].drop_duplicates()
    df_frames = df_frames.merge(meta, on='clip_id', how='left')
    df_frames = df_frames[df_frames['shot_type'].isin(['Natural', 'Crossed'])]

    fps = 25.0
    rows = []
    for feat in bio_cols:
        for frame in range(20):
            sub = df_frames[df_frames['frame'] == frame]
            va  = sub[sub['shot_type'] == 'Natural'][feat].dropna()
            vb  = sub[sub['shot_type'] == 'Crossed'][feat].dropna()
            if len(va) < 5 or len(vb) < 5: continue
            try:
                _, p = mannwhitneyu(va, vb, alternative='two-sided')
            except Exception:
                continue
            d  = cohens_d(va, vb)
            ms = (19 - frame) / fps * 1000                 # ms before contact (T0 = frame 19)
            rows.append({
                'feature'          : feat,
                'frame'            : frame,
                'ms_before_contact': round(ms, 0),
                'n_natural'        : len(va),
                'n_crossed'        : len(vb),
                'p_value'          : round(p, 6),
                'cohens_d'         : round(d, 4),
                'sig_p05'          : 'Yes' if p < 0.05 else 'No',
            })

    res = pd.DataFrame(rows)

    # 16 features x 20 frames is 320 tests -- about 16 of them will clear
    # p<0.05 with no signal at all. The "first significant frame" is read off
    # the FDR-corrected column, never the raw one.
    if len(res):
        reject, q = benjamini_hochberg(res['p_value'].values, alpha=0.05)
        res['q_value'] = np.round(q, 5)
        res['sig_fdr'] = np.where(reject, 'Yes', 'No')
        print_multiplicity_summary(multiplicity_summary(
            res['p_value'].values, label='frame-by-frame timeline'))

    res.to_csv('natural_crossed_timeline_stats.csv', index=False)
    print(f'[OK] natural_crossed_timeline_stats.csv  ({len(res)} rows)')

    # First significant frame per feature -- already sorted by increasing frame.
    # FDR-corrected, because 320 uncorrected tests hand out ~16 free hits.
    if is_empty(res, 'frame-by-frame timeline'):
        return res
    sig_col = 'sig_fdr' if 'sig_fdr' in res.columns else 'sig_p05'
    print(f'\n  First FDR-significant frame per feature:')
    report.append('\nTIMELINE -- first FDR-significant frame per feature')
    early_features = []
    for feat in bio_cols:
        sig = res[(res['feature'] == feat) & (res[sig_col] == 'Yes')]
        if len(sig) == 0:
            print(f'    {feat:25s}  never survives FDR')
            continue
        first = sig.iloc[0]
        flag  = ' <-- before 200ms' if first['ms_before_contact'] >= 200 else ''
        print(f'    {feat:25s}  frame {int(first["frame"]):2d}  ({first["ms_before_contact"]:.0f}ms before){flag}')
        report.append(f'  {feat}: frame {int(first["frame"])} ({first["ms_before_contact"]:.0f}ms before){flag}')
        if first['ms_before_contact'] >= 200:
            early_features.append(feat)

    print(f'\n  Features significant at least 200ms before contact: {len(early_features)}/{len(bio_cols)}')
    if early_features:
        print(f'    {", ".join(early_features)}')
    report.append(f'\n  Significant >=200ms before contact: {", ".join(early_features) if early_features else "none"}')

    return res


# -- 4. Natural vs Crossed by dominant foot --------------------------

def natural_crossed_by_foot(df, features, report):
    """Repeats Natural vs Crossed split by dominant foot -- is the biomechanical pattern symmetric?"""
    section('4. NATURAL vs CROSSED BY DOMINANT FOOT')
    print('  Is the Natural vs Crossed biomechanical pattern symmetric between feet?\n')

    report.append('\nNATURAL vs CROSSED BY DOMINANT FOOT')
    for foot, foot_label in [('right', 'Right Foot'), ('left', 'Left Foot')]:
        sub = df[df['foot_used'] == foot].copy()
        sub = sub[sub['shot_type'].isin(['Natural', 'Crossed'])]
        if len(sub) < 20:
            print(f'  {foot_label}: n={len(sub)} insufficient -- skipping')
            continue

        n_nat = (sub['shot_type'] == 'Natural').sum()
        n_cru = (sub['shot_type'] == 'Crossed').sum()
        print(f'  {foot_label} (n={len(sub)}): Natural={n_nat}  Crossed={n_cru}')

        res = run_mannwhitney(sub, features, 'shot_type', 'Natural', 'Crossed')
        if is_empty(res, f'{foot_label} subset'):
            continue
        sig = res[res['sig_fdr'] == 'Yes']
        print(f'    Surviving FDR: {len(sig)}/{len(res)}'
              f'  (uncorrected p<0.05: {(res["sig_p05"] == "Yes").sum()}, '
              f'~{len(res) * 0.05:.0f} expected by chance)')
        report.append(f'  {foot_label} (n={len(sub)}): {len(sig)}/{len(res)} surviving FDR')
        if len(sig) > 0:
            print(f'    Top 3:')
            for _, r in sig.head(3).iterrows():
                print(f'      {r["feature"]:40s}  p={r["p_value"]:.5f}  d={r["cohens_d"]:+.3f}')
                report.append(f'      {r["feature"]}  p={r["p_value"]:.5f}  d={r["cohens_d"]:+.3f}')
        print()


# -- 5. Goal rate Natural vs Crossed ------------------------------

def goal_rate(df, report):
    """Compares the goal rate between Natural and Crossed (chi-squared)."""
    section('5. GOAL RATE -- NATURAL vs CROSSED')

    df_out = df.dropna(subset=['outcome'])
    nat = df_out[df_out['shot_type'] == 'Natural']
    cru = df_out[df_out['shot_type'] == 'Crossed']

    goal_nat = (nat['outcome'] == 'goal').mean()
    goal_cru = (cru['outcome'] == 'goal').mean()
    ct = pd.crosstab(df_out['shot_type'], df_out['outcome'] == 'goal')
    try:
        chi2, p, dof, _ = chi2_contingency(ct)
    except Exception:
        chi2, p, dof = 0.0, 1.0, 0

    print(f'\n  Natural (n={len(nat)}): {goal_nat*100:.1f}% goal')
    print(f'  Crossed (n={len(cru)}): {goal_cru*100:.1f}% goal')
    print(f'\n  Chi-squared: chi2={chi2:.3f}  p={p:.4f}  df={dof}')
    sig = '[OK] Significant difference' if p < 0.05 else '[NOK] No significant difference'
    print(f'  {sig}')

    report.append('\nGOAL RATE -- Natural vs Crossed')
    report.append(f'  Natural: {goal_nat*100:.1f}%  (n={len(nat)})')
    report.append(f'  Crossed: {goal_cru*100:.1f}%  (n={len(cru)})')
    report.append(f'  chi2={chi2:.3f}  p={p:.4f}  -> {sig}')


# -- Main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Natural vs Crossed statistical tests -- complements the 04_models/ ML.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--master', '-m', default='clips_master.csv')
    ap.add_argument('--frames', '-f', default='biomechanics_dataset_complete.csv')
    args = ap.parse_args()

    print(f'Loading {args.master} ...')
    df = pd.read_csv(args.master)
    print(f'  {df.shape[0]} clips x {df.shape[1]} columns')

    df = build_shot_type(df)
    print(f'  Natural={(df["shot_type"]=="Natural").sum()}  Crossed={(df["shot_type"]=="Crossed").sum()}')

    features = feature_columns(df)
    print(f'  {len(features)} features to test')

    report = ['NATURAL vs CROSSED -- Statistical Tests', '=' * 60]

    # -- 1. Mann-Whitney + Cohen's d -------------------------------
    section("1. MANN-WHITNEY U + COHEN'S D -- ALL FEATURES")
    res = run_mannwhitney(df, features, 'shot_type', 'Natural', 'Crossed')
    if is_empty(res, 'Natural vs Crossed per-clip tests'):
        report.append('\nNo per-clip test had enough observations per group.')
        with open('natural_crossed_stats_report.txt', 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(report))
        print('[OK] natural_crossed_stats_report.txt (nothing to test)')
        return
    sig = res[res['sig_p05'] == 'Yes']
    bon = res[res['sig_bonferroni'] == 'Yes']
    fdr = res[res['sig_fdr'] == 'Yes']
    res.to_csv('natural_crossed_statistical.csv', index=True)
    print(f"\n[OK] natural_crossed_statistical.csv")

    report.append(f'\nMANN-WHITNEY + COHEN D -- {len(features)} features tested')
    report.append(f'  Uncorrected p<0.05: {len(sig)}/{len(res)} '
                  f'(~{len(res) * 0.05:.0f} expected by chance)')
    report.append(f'  Surviving Bonferroni: {len(bon)}/{len(res)}')
    report.append(f'  Surviving BH-FDR: {len(fdr)}/{len(res)}')

    # -- 2. Top 10 with interpretation --------------------------------
    top10_with_interpretation(res, report)

    # -- 3. Timeline by frame ---------------------------------------
    if os.path.exists(args.frames):
        print(f'\nLoading {args.frames} ...')
        df_frames = pd.read_csv(args.frames)
        bio_cols  = [c for c in BASE_INTERP if c in df_frames.columns]  # only the base biomechanical columns, available per frame
        for c in bio_cols:
            df_frames[c] = pd.to_numeric(df_frames[c], errors='coerce')
        res_timeline = timeline_significance(df, df_frames, bio_cols, report)
        plot_timeline(res_timeline, 'natural_crossed_timeline.png')
    else:
        print(f'\n[WARNING] {args.frames} not found -- skipping timeline')

    # -- 4. By dominant foot ------------------------------------------
    natural_crossed_by_foot(df, features, report)

    # -- 5. Goal rate ------------------------------------------------
    goal_rate(df, report)

    with open('natural_crossed_stats_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'\n[OK] natural_crossed_stats_report.txt')

    print('\n[OK] Done!')


if __name__ == '__main__':
    main()
