"""
model_comparison.py
===================
Aggregates the results from every model and produces the final table and charts.

Reads the output CSVs from the other scripts:
  - tabular_results.csv        (tabular_models.py)
  - dl_results.csv             (deep_learning_models.py)
  - interpretable_results.csv  (interpretable_model.py)

Output:
  - comparison_table.csv                        -- full table of every model
  - comparison_P2__Direction_Left_vs_Right.png   -- chart for P2 (direction)
  - comparison_P1__Predictability_gk_guessed.png -- chart for P1 (gk_guessed)

Usage:
  python model_comparison.py
  python model_comparison.py --dir "D:/output/models/"
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
import seaborn as sns
from pathlib import Path

sns.set_theme(style='darkgrid', font_scale=1.1)

# Model order in the chart -- from simplest to most complex
MODEL_ORDER = [
    'Baseline',
    'Logistic Regression',
    'Random Forest',
    'SVM (RBF)',
    'XGBoost',
    'Decision Tree',
    'CNN 1D',
    'LSTM',
]

# Colors per model category
MODEL_COLORS = {
    'Baseline'           : '#aaaaaa',
    'Logistic Regression': '#1f77b4',
    'Random Forest'      : '#2ca02c',
    'SVM (RBF)'          : '#9467bd',
    'XGBoost'            : '#ff7f0e',
    'Decision Tree'      : '#8c564b',
    'CNN 1D'             : '#e377c2',
    'LSTM'               : '#d62728',
}


def load_results(data_dir):
    """Reads every results CSV and concatenates them."""
    all_dfs = []
    for fname in ['tabular_results.csv', 'dl_results.csv', 'interpretable_results.csv']:
        path = Path(data_dir) / fname
        if path.exists():
            df = pd.read_csv(path)
            # Normalize column names between scripts (dl_results.csv uses 'task'/'roc_auc' instead of 'target'/'roc_auc_mean')
            if 'task' in df.columns:
                df = df.rename(columns={'task':'target', 'roc_auc':'roc_auc_mean'})
            all_dfs.append(df)
            print(f'  [OK] {fname}  ({len(df)} results)')
        else:
            print(f'  [WARNING] {fname} not found -- skipping')

    if not all_dfs:
        print('[ERROR] No results file found.'); return None
    return pd.concat(all_dfs, ignore_index=True)


def add_baseline(df):
    """Adds a chance row for each target.

    ROC-AUC has a fixed chance level of 0.500 regardless of class balance,
    which is exactly why it is now the metric being plotted: the old PR-AUC
    chart used the class prevalence as its reference line, and on these
    near-balanced targets a model sitting at chance drifts above that line
    under fold noise and looks like a result.
    """
    rows = []
    for target in df['target'].unique():
        rows.append({'model': 'Chance', 'target': target,
                     'roc_auc_mean': 0.5, 'roc_auc_std': 0.0,
                     'roc_beats_chance': False})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def plot_target(df, target_name, out_path, baseline=0.5):
    """ROC-AUC bar chart by model, with the 95% fold range as error bars.

    A bar whose lower whisker crosses 0.500 is not a result, and the chart
    should make that visible rather than hide it behind a mean.
    """
    sub = df[df['target']==target_name].copy()

    # Sort by the defined model order -- models not listed go at the end
    order = [m for m in MODEL_ORDER if m in sub['model'].values]
    order += [m for m in sub['model'].values if m not in order]
    sub['_order'] = sub['model'].map({m:i for i,m in enumerate(order)})
    sub = sub.sort_values('_order')

    colors = [MODEL_COLORS.get(m,'#333333') for m in sub['model']]

    fig, ax = plt.subplots(figsize=(10, 6))
    vals = sub['roc_auc_mean'].fillna(0.5)
    bars = ax.bar(range(len(sub)), vals, color=colors, alpha=0.85, width=0.6)

    # Error bars: prefer the 95% fold range, fall back to the std
    for i, (_, row) in enumerate(sub.iterrows()):
        m = row.get('roc_auc_mean')
        if pd.isna(m):
            continue
        lo, hi = row.get('roc_auc_lo'), row.get('roc_auc_hi')
        if pd.notna(lo) and pd.notna(hi):
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]],
                        fmt='none', color='black', capsize=5, lw=1.5)
        elif row.get('roc_auc_std', 0) > 0:
            ax.errorbar(i, m, yerr=row['roc_auc_std'],
                        fmt='none', color='black', capsize=5, lw=1.5)

    ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.8, label='Chance (0.500)')

    for bar, (_, row) in zip(bars, sub.iterrows()):
        if pd.isna(row.get('roc_auc_mean')):
            continue
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{row["roc_auc_mean"]:.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='500')

    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub['model'], rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0.35, min(1.0, float(vals.max()) + 0.15))
    ax.set_ylabel('ROC-AUC (error bars = 95% fold range)', fontsize=11)
    ax.set_title(f'Model Comparison\n{target_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Category legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Linear (LR)'),
        Patch(facecolor='#2ca02c', label='Ensemble (RF, XGB)'),
        Patch(facecolor='#9467bd', label='Kernel (SVM)'),
        Patch(facecolor='#8c564b', label='Tree (DT)'),
        Patch(facecolor='#d62728', label='Deep Learning (LSTM, CNN)'),
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', ls='--',
                                                    label='Chance (0.500)')],
              fontsize=9, loc='lower right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out_path}')


def print_table(df):
    """Prints a formatted table of every model."""
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE -- ALL MODELS")
    print(f"{'='*70}")
    for target in df['target'].unique():
        sub = df[df['target']==target].copy()
        sub['_order'] = sub['model'].map({m:i for i,m in enumerate(MODEL_ORDER)}).fillna(99)
        sub = sub.sort_values('_order')
        print(f"\n  {target}   (chance ROC-AUC = 0.500)")
        print(f"  {'Model':22s}  {'ROC-AUC [95% fold range]':<28s}"
              f"{'grouped':<10s}{'PR-AUC':<10s}>chance")
        print(f"  {'-'*22}  {'-'*28}{'-'*10}{'-'*10}-------")
        for _, row in sub.iterrows():
            m = row.get('roc_auc_mean')
            if pd.isna(m):
                continue
            lo, hi = row.get('roc_auc_lo'), row.get('roc_auc_hi')
            roc = (f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
                   if pd.notna(lo) and pd.notna(hi) else f"{m:.3f}")
            grp = (f"{row['grouped_roc_auc_mean']:.3f}"
                   if pd.notna(row.get('grouped_roc_auc_mean')) else '--')
            pr  = (f"{row['pr_auc_mean']:.3f}"
                   if pd.notna(row.get('pr_auc_mean')) else '--')
            # Derive from the fold range when the flag is absent: a missing
            # value read as NaN, and NaN is truthy, which flagged
            # chance-level models as results.
            beats = row.get('roc_beats_chance')
            if pd.isna(beats):
                lo = row.get('roc_auc_lo')
                beats = bool(pd.notna(lo) and lo > 0.5)
            flag = '[OK]' if bool(beats) else '--'
            print(f"  {row['model']:22s}  {roc:<28s}{grp:<10s}{pr:<10s}{flag}")

        real = sub[sub['model'] != 'Chance']
        lo_ok = real.get('roc_auc_lo', pd.Series(dtype=float)) > 0.5
        flags = real['roc_beats_chance'].fillna(False).astype(bool) | lo_ok.reindex(real.index).fillna(False)
        if len(real) and not flags.any():
            print("    -> no model's fold range clears chance on this target.")


def main():
    ap = argparse.ArgumentParser(
        description='Aggregates and compares the results from every model.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--dir', '-d', default='.',
                    help='Folder with the results CSVs (default: current folder)')
    args = ap.parse_args()

    print('Loading results...')
    df = load_results(args.dir)
    if df is None: return

    df = add_baseline(df)                                  # adds a baseline row for visual reference

    print_table(df)

    # Charts per target
    for target in df['target'].unique():
        safe = (target.replace(' ', '_').replace('/', '_').replace('(', '')
                      .replace(')', '').replace('--', ''))   # sanitizes the target name for the file name
        plot_target(df, target, f'comparison_{safe.strip("_")}.png')

    # Save the full table
    df.drop(columns=['_order'], errors='ignore').to_csv('comparison_table.csv', index=False)
    print(f'[OK] comparison_table.csv')
    print('[OK] Done!')


if __name__ == '__main__':
    main()
