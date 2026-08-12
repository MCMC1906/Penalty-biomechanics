"""
deep_learning_models.py
========================
LSTM and CNN 1D on the 20-frame biomechanical sequences.

Input : sequence of 20 frames x 16 biomechanical features per clip
Target: macro_zone (Left=1 vs Right=0)  and  gk_guessed (1 vs 0)

Evaluation
----------
Primary metric is ROC-AUC (chance = 0.500). Both targets are close to
balanced, so PR-AUC -- whose chance level is the class prevalence -- makes a
model sitting at chance look like it is beating a baseline. PR-AUC is still
reported alongside it.

Early stopping
--------------
An earlier version selected the best epoch by scoring the TEST fold every
epoch, keeping the best-scoring weights, and then reporting that same fold.
That reports the maximum of ~120 noisy test evaluations and inflates the
result by a large margin on folds of ~80 samples. Fixed: an inner validation
split (20% of the training fold, stratified) drives early stopping and weight
selection, and the test fold is evaluated exactly once, after training ends.

The tabular benchmark is read from tabular_results.csv when present, instead
of being hardcoded -- the old hardcoded values (0.657 / 0.492) had drifted
away from what the tabular script actually produces.

Install:
  pip install torch

Usage:
  python deep_learning_models.py
  python deep_learning_models.py --data biomechanics_dataset_complete.csv
  python deep_learning_models.py --repeats 3     # average over 3 seeds
"""

import argparse, sys, time
import warnings
# Silence the noisy-but-harmless deprecation churn only. Convergence failures,
# degenerate folds and constant-column warnings are exactly the diagnostics
# that matter at this sample size, so they are left switched on.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection  import (StratifiedKFold, StratifiedGroupKFold,
                                      train_test_split)
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import average_precision_score, roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("PyTorch not installed. Run: pip install torch")
    sys.exit(1)

sns.set_theme(style='darkgrid', font_scale=1.05)

# -- Configuration ---------------------------------------------------

SEQ_LEN    = 20

# Input width is derived from the data at runtime, not hardcoded. It changed
# when lateral_trunk_lean was dropped from BIO_COLS (16 -> 15), and a constant
# here silently disagreed with the tensors until the LSTM raised a shape error.
# load_sequences() sets this before any model is constructed.
N_FEATURES = None          # set by load_sequences() from the data
CV_FOLDS   = 5
EPOCHS     = 120
BATCH_SIZE = 32
LR         = 1e-3
DROPOUT    = 0.4
PATIENCE   = 15        # early stopping
SEED       = 42        # fixes weight init / batch shuffling so LSTM and CNN 1D results are reproducible run to run
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)               # no-op if there's no GPU
torch.backends.cudnn.deterministic = True       # forces deterministic cuDNN kernels (a bit slower, but stable results)
torch.backends.cudnn.benchmark = False

BIO_COLS = [
    'kick_knee_angle', 'supp_knee_angle',
    'kick_ankle_angle', 'supp_ankle_angle',
    'kick_hip_angle',
    'kick_angular_vel', 'kick_hip_angular_vel', 'kick_ankle_angular_vel',
    'kick_angular_accel',
    'kick_foot_speed_ms', 'running_speed_kmh',
    'trunk_inclination', 'torso_torsion_angle',   # lateral_trunk_lean dropped: sign is camera-dependent
    'left_elbow_angle', 'right_elbow_angle',
]

print(f"  Device: {DEVICE}")


# -- Data preparation --------------------------------------------

def load_sequences(data_path, target='direction'):
    """
    target='direction' -> macro_zone Left(1) vs Right(0)
    target='gk'        -> gk_guessed 1(guessed) vs 0(not guessed)

    Returns X (n, 20, 16), y, the PR-AUC chance level, and the source-video
    label of each clip (used to group the CV folds).
    """
    df = pd.read_csv(data_path)

    for c in BIO_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['target_zone'] = df['target_zone'].astype(str)
    df['macro_zone']  = df['target_zone'].apply(
        lambda z: 'Left' if z in {'1','4','7'}
        else 'Right' if z in {'3','6','9'} else None)

    if target == 'direction':
        df = df[df['macro_zone'].notna()].copy()
        label_fn = lambda clip: 1 if clip['macro_zone'].iloc[0] == 'Left' else 0
    else:  # gk
        df['gk_guessed'] = pd.to_numeric(df['gk_guessed'], errors='coerce')
        df = df.dropna(subset=['gk_guessed']).copy()
        label_fn = lambda clip: int(clip['gk_guessed'].iloc[0])

    cols = [c for c in BIO_COLS if c in df.columns]

    def source_video(clip_id, clip):
        """Source video of a clip. clip_id looks like
        crop_<video>__event_<id>__t0_<frame>, so the video name can be
        recovered from it when the column is absent."""
        if 'video_name' in clip.columns:
            return str(clip['video_name'].iloc[0])
        cid = str(clip_id)
        return cid.split('__event_')[0].replace('crop_', '') if '__event_' in cid else cid

    seqs, labels, clip_ids, groups = [], [], [], []
    for clip_id, clip in df.groupby('clip_id'):
        clip = clip.sort_values('frame')
        arr  = clip[cols].values.astype(np.float32)

        if len(arr) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(arr), len(cols)), dtype=np.float32)
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:SEQ_LEN]  # in practice the clips already have exactly 20 frames -- this is just a precaution

        arr = np.nan_to_num(arr, nan=0.0)
        seqs.append(arr)
        labels.append(label_fn(clip))
        clip_ids.append(clip_id)
        groups.append(source_video(clip_id, clip))

    X = np.stack(seqs)
    y = np.array(labels)
    groups = np.array(groups)

    # Model constructors default to N_FEATURES; bind it to what the data
    # actually has so the two can never drift apart again.
    global N_FEATURES
    N_FEATURES = X.shape[2]

    # PR-AUC chance level is the prevalence in THIS dataset -- computed, not
    # hardcoded, so it stays correct if the dataset changes.
    pr_baseline = float(y.mean())

    print(f"  Sequences: {X.shape}  |  class 1: {y.sum()}  class 0: {(1-y).sum()}")
    print(f"  Source videos: {len(np.unique(groups))}")
    print(f"  Chance ROC-AUC 0.500  |  chance PR-AUC {pr_baseline:.3f}")
    return X, y, pr_baseline, groups


def normalize_fold(X_tr, X_te):
    """Normalizes feature-wise (train mean/std applied to the test set)."""
    n_tr, seq, feat = X_tr.shape
    scaler = StandardScaler()
    X_tr_r = scaler.fit_transform(X_tr.reshape(-1, feat)).reshape(n_tr, seq, feat)
    X_te_r = scaler.transform(X_te.reshape(-1, feat)).reshape(X_te.shape[0], seq, feat)
    return X_tr_r, X_te_r


# -- Models ----------------------------------------------------------

class LSTMClassifier(nn.Module):
    """
    Simple LSTM for short sequences.
    Uses the last hidden state as the clip representation.
    """
    def __init__(self, input_size=None, hidden_size=32,
                 num_layers=1, dropout=DROPOUT):
        super().__init__()
        input_size   = N_FEATURES if input_size is None else input_size
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq=20, features=16)
        out, _ = self.lstm(x)
        out    = out[:, -1, :]          # last hidden state
        return self.head(self.dropout(out)).squeeze(-1)


class CNN1DClassifier(nn.Module):
    """
    CNN 1D: treats the 20-frame sequence as a time signal.
    Global Average Pooling at the end — robust to the short sequence.
    """
    def __init__(self, input_size=None, dropout=DROPOUT):
        super().__init__()
        # Transposed input: (batch, features=16, seq=20)
        self.conv = nn.Sequential(
            nn.Conv1d(N_FEATURES if input_size is None else input_size,
                      32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),    # Global Average Pooling -> (batch, 64, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq=20, features=16) -> transpose -> (batch, features, seq)
        x   = x.transpose(1, 2)
        out = self.conv(x).squeeze(-1)  # (batch, 64)
        return self.head(out).squeeze(-1)


# -- Training loop -----------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for xb, yb in loader:
        logits = model(xb.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(yb.numpy())
    return np.array(all_probs), np.array(all_labels)


def train_model(ModelClass, X_tr, y_tr, X_te, y_te, pos_weight, seed=SEED):
    """Trains a model (LSTM or CNN) and returns the fold's test metrics.

    Early stopping and weight selection use an inner validation split taken
    out of the training fold. The test fold is evaluated exactly once, at
    the end, after the weights are already fixed.
    """
    # Inner split: 20% of the training fold, stratified, held out for early
    # stopping. Small folds make this split noisy, which is why PATIENCE is
    # generous -- but a noisy honest signal beats a clean dishonest one.
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)

    model     = ModelClass().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                           batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va_t, y_va_t),
                           batch_size=BATCH_SIZE)
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t),
                           batch_size=BATCH_SIZE)

    best_score   = -np.inf
    best_weights = None
    patience_cnt = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, tr_loader, optimizer, criterion)

        # Model selection signal comes from the VALIDATION split, which was
        # carved out of the training fold. The test fold is never evaluated
        # inside this loop -- see the note in the header.
        probs_va, labels_va = evaluate(model, va_loader)
        val_score = (roc_auc_score(labels_va, probs_va)
                     if len(np.unique(labels_va)) > 1 else 0.5)

        scheduler.step(1 - val_score)   # ReduceLROnPlateau minimizes -> 1-score to maximize score
        history.append({'epoch': epoch, 'loss': loss, 'val_roc_auc': val_score})

        if val_score > best_score:
            best_score   = val_score
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    # Restore the weights that were best ON VALIDATION
    if best_weights:
        model.load_state_dict(best_weights)

    # Only now is the test fold touched, exactly once.
    probs, labels = evaluate(model, te_loader)
    two_classes   = len(np.unique(labels)) > 1
    final_prauc   = average_precision_score(labels, probs) if two_classes else 0.0
    final_rocauc  = roc_auc_score(labels, probs) if two_classes else 0.5

    return final_prauc, final_rocauc, pd.DataFrame(history), epoch


# -- Cross-validation ---------------------------------------------------

def run_cv(ModelClass, model_name, X, y, groups=None, repeats=1):
    """Stratified CV, optionally grouped by source video and repeated over
    several seeds. Returns mean/std of ROC-AUC and PR-AUC plus the training
    history (validation curves)."""
    print(f"\n  {'='*55}")
    print(f"  {model_name}")
    print(f"  {'='*55}")

    pr_scores, roc_scores, all_history = [], [], []
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))  # class0/class1 ratio -- compensates the imbalance in the loss

    for rep in range(repeats):
        seed = SEED + rep
        if groups is not None:
            # Clips from the same source video share camera pose, lighting and
            # possibly the same kicker -- keep them on one side of the split.
            cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                      random_state=seed)
            splits = cv.split(X, y, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                 random_state=seed)
            splits = cv.split(X, y)

        for fold, (tr_idx, te_idx) in enumerate(splits, 1):
            X_tr, X_te = normalize_fold(X[tr_idx], X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]

            t0 = time.time()
            pr, roc, hist, n_ep = train_model(
                ModelClass, X_tr, y_tr, X_te, y_te, pos_weight, seed=seed)
            elapsed = time.time() - t0

            pr_scores.append(pr)
            roc_scores.append(roc)
            hist['fold'] = fold + rep * CV_FOLDS
            all_history.append(hist)

            tag = f"rep {rep+1} " if repeats > 1 else ""
            print(f"  {tag}Fold {fold}  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  "
                  f"epochs={n_ep}  ({elapsed:.0f}s)")

    mean_roc, std_roc = np.mean(roc_scores), np.std(roc_scores)
    mean_pr,  std_pr  = np.mean(pr_scores),  np.std(pr_scores)
    lo, hi = np.percentile(roc_scores, [2.5, 97.5])

    print(f"\n  -> ROC-AUC : {mean_roc:.3f} +/- {std_roc:.3f}  "
          f"[fold range {lo:.3f}, {hi:.3f}]")
    print(f"  -> PR-AUC  : {mean_pr:.3f} +/- {std_pr:.3f}")
    if lo <= 0.5:
        print("     (fold range includes chance -- not a supported result)")

    return {
        'name'        : model_name,
        'roc_mean'    : mean_roc,
        'roc_std'     : std_roc,
        'roc_lo'      : lo,
        'roc_hi'      : hi,
        'pr_mean'     : mean_pr,
        'pr_std'      : std_pr,
        'beats_chance': bool(lo > 0.5),
        'fold_scores' : roc_scores,
        'history'     : pd.concat(all_history, ignore_index=True),
    }


# -- Comparison charts ------------------------------------------------

def plot_results(lstm_res, cnn_res, out_path='dl_comparison.png',
                 tabular_benchmark=None, title=''):
    """3 panels: ROC-AUC comparison + validation curves for each model.

    Chance is 0.500 for ROC-AUC, which is the line that matters; error bars
    are the 95% range across folds, so a bar whose lower whisker crosses 0.5
    is visibly not a result.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f'LSTM vs CNN 1D — {title}',
                 fontsize=13, fontweight='bold', y=1.02)

    palette = {'LSTM': '#1f77b4', 'CNN 1D': '#ff7f0e', 'Best tabular': '#2ca02c'}

    # -- 1. ROC-AUC comparison -----------------------------------------
    ax = axes[0]
    names  = [lstm_res['name'], cnn_res['name']]
    means  = [lstm_res['roc_mean'], cnn_res['roc_mean']]
    lo_err = [m - r['roc_lo'] for m, r in zip(means, [lstm_res, cnn_res])]
    hi_err = [r['roc_hi'] - m for m, r in zip(means, [lstm_res, cnn_res])]
    colors = [palette['LSTM'], palette['CNN 1D']]

    if tabular_benchmark is not None:
        names.insert(0, 'Best tabular')
        means.insert(0, tabular_benchmark)
        lo_err.insert(0, 0.0)
        hi_err.insert(0, 0.0)
        colors.insert(0, palette['Best tabular'])

    bars = ax.bar(names, means, color=colors, alpha=0.85, width=0.5,
                  yerr=[lo_err, hi_err], capsize=6, error_kw={'lw': 1.5})
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='500')
    ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.8, label='Chance (0.500)')
    ax.set_ylim(0.30, 0.85)
    ax.set_ylabel('ROC-AUC (error bars = 95% fold range)')
    ax.set_title('Model Comparison', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(axis='x', rotation=10)

    # -- 2/3. Validation curves ----------------------------------------
    for ax, res, colour in ((axes[1], lstm_res, '#1f77b4'),
                            (axes[2], cnn_res, '#ff7f0e')):
        hist = res['history']
        for fold in hist['fold'].unique():
            sub = hist[hist['fold'] == fold]
            ax.plot(sub['epoch'], sub['val_roc_auc'], alpha=0.35, lw=1, color=colour)
        avg = hist.groupby('epoch')['val_roc_auc'].mean()
        ax.plot(avg.index, avg.values, color=colour, lw=2.5, label='CV average')
        ax.axhline(0.5, color='red', ls='--', lw=1.2, alpha=0.6, label='Chance')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROC-AUC (inner validation split)')
        ax.set_title(f"{res['name']} — Validation Curves",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[OK] {out_path}")


def load_tabular_benchmark(task_name, path='tabular_results.csv'):
    """Best tabular ROC-AUC for this task, read from tabular_results.csv.

    Replaces the hardcoded 0.657 / 0.492 constants, which no longer matched
    what tabular_models.py produces.
    """
    try:
        t = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [note] {path} not found -- no tabular benchmark shown.")
        return None
    key = 'grouped_roc_auc_mean' if 'grouped_roc_auc_mean' in t.columns else 'roc_auc_mean'
    hit = t[t['target'].astype(str).str.startswith(task_name.split('--')[0].strip())]
    if hit.empty or key not in hit.columns:
        return None
    return float(hit[key].max())


# -- Main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='biomechanics_dataset_complete.csv')
    ap.add_argument('--repeats', type=int, default=3,
                    help='number of CV repeats with different seeds (default 3)')
    ap.add_argument('--no-group', action='store_true',
                    help='use a plain random split instead of grouping by source video')
    args = ap.parse_args()

    all_summaries = []

    for task_name, target in [
        ('P2 -- Direction (Left vs Right)',   'direction'),
        ('P1 -- Predictability (gk_guessed)', 'gk'),
    ]:
        print(f"\n{'='*60}")
        print(task_name)
        print(f"{'='*60}")
        print(f"\n{args.data}")

        X, y, pr_baseline, groups = load_sequences(args.data, target=target)
        if args.no_group:
            groups = None

        benchmark = load_tabular_benchmark(task_name)

        print(f"\nConfiguration")
        print(f"   Sequence  : {SEQ_LEN} frames x {X.shape[2]} features")
        print(f"   CV        : {CV_FOLDS}-fold x {args.repeats} seeds"
              + ("  (grouped by source video)" if groups is not None else ""))
        print(f"   Chance    : ROC-AUC 0.500  |  PR-AUC {pr_baseline:.3f}")
        if benchmark is not None:
            print(f"   Best tabular ROC-AUC: {benchmark:.3f}")

        print(f"\n{'='*60}")
        print("LSTM  (hidden=32, 1 layer, dropout=0.4)")
        lstm_res = run_cv(LSTMClassifier, 'LSTM', X, y,
                          groups=groups, repeats=args.repeats)

        print(f"\n{'='*60}")
        print("CNN 1D  (Conv 16->32->64 + GlobalAvgPool, dropout=0.4)")
        cnn_res = run_cv(CNN1DClassifier, 'CNN 1D', X, y,
                         groups=groups, repeats=args.repeats)

        for res in (lstm_res, cnn_res):
            res['task'] = task_name
            res['pr_baseline'] = pr_baseline
            res['tabular_benchmark'] = benchmark

        # Summary per task
        print(f"\n{'='*60}")
        print(f"SUMMARY — {task_name}")
        print(f"{'Model':20s} {'ROC-AUC':>22s} {'PR-AUC':>10s}  >chance")
        print(f"{'='*62}")
        print(f"{'Chance':20s} {0.5:>22.3f} {pr_baseline:>10.3f}")
        if benchmark is not None:
            print(f"{'Best tabular':20s} {benchmark:>22.3f}")
        for res in (lstm_res, cnn_res):
            rng  = f"{res['roc_mean']:.3f} [{res['roc_lo']:.3f},{res['roc_hi']:.3f}]"
            flag = '[OK]' if res['beats_chance'] else '--'
            print(f"{res['name']:20s} {rng:>22s} {res['pr_mean']:>10.3f}  {flag}")

        plot_results(lstm_res, cnn_res,
                     out_path=f"dl_{target}_comparison.png",
                     tabular_benchmark=benchmark, title=task_name)

        all_summaries.extend([lstm_res, cnn_res])

    # Final CSV
    summary_df = pd.DataFrame([{
        'task': r['task'], 'model': r['name'],
        'roc_auc_mean': r['roc_mean'], 'roc_auc_std': r['roc_std'],
        'roc_auc_lo': r['roc_lo'], 'roc_auc_hi': r['roc_hi'],
        'pr_auc_mean': r['pr_mean'], 'pr_auc_std': r['pr_std'],
        'pr_auc_baseline': r['pr_baseline'],
        'tabular_benchmark_roc': r['tabular_benchmark'],
        'roc_beats_chance': r['beats_chance'],
    } for r in all_summaries])
    summary_df.to_csv('dl_results.csv', index=False)
    print(f"\n[OK] dl_results.csv")
    print("[OK] Done!")


if __name__ == '__main__':
    main()
