"""
build_clips_master.py
=====================
Aggregates biomechanics_dataset_complete.csv (per frame) into a per-clip dataset with all the features needed for analysis and modeling.

Reads:
  biomechanics_dataset_complete.csv  - per frame (n_clips x 20 rows)
  proximal_distal_sequence.csv       - per clip (hip->knee->ankle sequence)

Writes:
  clips_master.csv  - sep=,  decimal=.  (used by 03_statistical_analysis/ and 04_models/)

Features generated per clip:
  - Metadata: clip_id, event_id, outcome, foot_used, target_zone, macro_zone, gk_guessed
  - Contact frame (frame 19): contact_kick_knee_angle, etc.
  - Phase means (early/mid/late): early_mean_kick_knee_angle, etc.
  - Phase std devs: early_std_kick_knee_angle, etc.
  - Phase maxima: late_max_kick_foot_speed_ms, etc.
  - Deltas (late_mean - early_mean): delta_kick_knee_angle, etc.
  - Deception score: variation in trunk posture (deception proxy)
  - Derived features: macro_zone, knee_stability_index, elbow_asymmetry, etc.
  - Proximal-distal: peak_hip_vel, dt_hip_to_knee, etc.
  - Support-foot position: support_foot_x, support_foot_y, etc.

Usage:

  python build_clips_master.py -d "D:/output/" -o "D:/output/"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


# -- Configuration ---------------------------------------------------

PHASES = {
    'early': (0,  6),   # approach run
    'mid'  : (7,  13),  # setup / loading
    'late' : (14, 19),  # swing and impact
}

BIO_COLS = [                                               # base biomechanical metrics (per frame)
    'kick_knee_angle', 'supp_knee_angle',
    'kick_ankle_angle', 'supp_ankle_angle',
    'kick_hip_angle',
    'kick_hip_angular_vel', 'kick_angular_vel', 'kick_ankle_angular_vel',
    'kick_angular_accel',
    'kick_foot_speed_ms', 'running_speed_kmh',
    'trunk_inclination', 'lateral_trunk_lean', 'torso_torsion_angle',
    'left_elbow_angle', 'right_elbow_angle',
    'mean_visibility_score',
]

META_COLS = [                                              # metadata extracted from the first frame
    'clip_id', 'video_name', 'event_id',
    'foot_used', 'kicking_leg', 'outcome',
    'target_zone', 'gk_guessed',
]


# -- Helpers -------------------------------------------------------

def _macro_zone(z):                                        # converts zone 1-9 into Left/Center/Right/Out
    z = str(z)
    if z in {'1','4','7'}: return 'Left'
    if z in {'3','6','9'}: return 'Right'
    if z in {'2','5','8'}: return 'Center'
    return 'Out'

def _gk_label(v):                                          # converts 0/1 into a readable label
    try:
        return 'Guessed' if int(v) == 1 else 'Not Guessed'
    except Exception:
        return None

def _r(val, dec=4):                                        # rounds a float, returns None if NaN or Inf
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, dec)
    except Exception:
        return None


# -- Per-clip features ---------------------------------------------

def build_clip_features(clip_df: pd.DataFrame) -> dict:
    """Takes the 20 rows of a clip and returns a dict of aggregated features."""
    clip_df = clip_df.sort_values('frame').reset_index(drop=True)
    row = {}

    # First-frame metadata (identical across all frames)
    first = clip_df.iloc[0]
    for c in META_COLS:
        if c in clip_df.columns:
            row[c] = first[c]
    row['n_frames'] = len(clip_df)

    # Derived labels
    if 'target_zone' in clip_df.columns:
        row['macro_zone'] = _macro_zone(first.get('target_zone', ''))
        row['zone_name']  = str(first.get('target_zone', ''))
    if 'gk_guessed' in clip_df.columns:
        row['gk_guessed_label'] = _gk_label(first.get('gk_guessed'))
    if 'foot_used' in clip_df.columns:
        fu = str(first.get('foot_used', ''))
        row['foot_used_label'] = 'Right Foot' if fu == 'right' else 'Left Foot'
        row['foot_right']      = int(fu == 'right')

    # Contact frame
    contact = clip_df[clip_df['is_contact_frame'] == 1]
    if len(contact) == 0:
        contact = clip_df.iloc[[-1]]                       # fallback if is_contact_frame isn't set
    for c in BIO_COLS:
        if c in clip_df.columns:
            row[f'contact_{c}'] = _r(contact[c].iloc[0])

    # Means, std devs, maxima and minima per phase
    phase_means = {}
    for phase, (f0, f1) in PHASES.items():
        seg = clip_df[(clip_df['frame'] >= f0) & (clip_df['frame'] <= f1)]
        for c in BIO_COLS:
            if c not in clip_df.columns:
                continue
            vals = seg[c].dropna()
            row[f'{phase}_mean_{c}'] = _r(vals.mean()) if len(vals) else None
            row[f'{phase}_std_{c}']  = _r(vals.std())  if len(vals) else None
            row[f'{phase}_max_{c}']  = _r(vals.max())  if len(vals) else None
            row[f'{phase}_min_{c}']  = _r(vals.min())  if len(vals) else None
            phase_means.setdefault(phase, {})[c] = vals.mean() if len(vals) else np.nan

    # Deltas (late_mean - early_mean) - evolution over the clip
    for c in BIO_COLS:
        e = phase_means.get('early', {}).get(c, np.nan)
        l = phase_means.get('late',  {}).get(c, np.nan)
        row[f'delta_{c}'] = _r(l - e) if not (np.isnan(e) or np.isnan(l)) else None

    # Deception score - change in trunk posture between early and late
    # deception proxy: a kicker who changes posture a lot may be trying to deceive the GK
    def _deception(col):
        e = phase_means.get('early', {}).get(col, np.nan)
        l = phase_means.get('late',  {}).get(col, np.nan)
        return abs(l - e) if not (np.isnan(e) or np.isnan(l)) else np.nan

    d_torsion = _deception('torso_torsion_angle')
    d_lean    = _deception('lateral_trunk_lean')
    d_trunk   = _deception('trunk_inclination')

    row['deception_torsion'] = _r(d_torsion)
    row['deception_lateral'] = _r(d_lean)
    row['deception_trunk']   = _r(d_trunk)

    valid_d = [v for v in [d_torsion, d_lean, d_trunk] if not np.isnan(v)]
    row['deception_score'] = _r(np.mean(valid_d)) if valid_d else None

    # Derived features
    row['knee_stability_index'] = _r(clip_df['supp_knee_angle'].std()) if 'supp_knee_angle' in clip_df.columns else None

    lel = row.get('contact_left_elbow_angle')
    rel = row.get('contact_right_elbow_angle')
    row['elbow_asymmetry'] = _r(abs(lel - rel)) if lel is not None and rel is not None else None

    if 'running_speed_kmh' in clip_df.columns:
        rs = clip_df['running_speed_kmh'].dropna()
        row['max_running_speed_kmh'] = _r(rs.max())

    # Support-foot position, uses the first frame where it's available
    if 'support_foot_x' in clip_df.columns:
        sfx = clip_df['support_foot_x'].dropna()
        sfy = clip_df['support_foot_y'].dropna() if 'support_foot_y' in clip_df.columns else pd.Series()
        if len(sfx):
            x = float(sfx.iloc[0]); y = float(sfy.iloc[0]) if len(sfy) else 0.0
            row['support_foot_x']         = _r(x)
            row['support_foot_y']         = _r(y)
            row['support_foot_distance']  = _r(np.sqrt(x**2 + y**2))
            row['support_foot_angle_deg'] = _r(np.degrees(np.arctan2(y, x)))
            row['support_foot_quadrant']  = (
                'front-left'  if x < 0 and y > 0  else
                'front-right' if x > 0 and y > 0  else
                'back-left'   if x < 0 and y <= 0 else
                'back-right')

    if 'mean_visibility_score' in clip_df.columns:
        row['clip_visibility'] = _r(clip_df['mean_visibility_score'].mean())

    return row


# -- Main pipeline -------------------------------------------------

def build_master(data_dir: str, out_dir: str):
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    complete_path = data_dir / 'biomechanics_dataset_complete.csv'
    proxdist_path = data_dir / 'proximal_distal_sequence.csv'

    if not complete_path.exists():
        print(f'[ERROR] Not found: {complete_path}'); return

    print(f'Loading {complete_path} ...')
    df = pd.read_csv(complete_path)
    print(f'   {len(df)} rows x {len(df.columns)} columns  |  {df["clip_id"].nunique()} clips')

    for c in BIO_COLS:                                     # force numeric types, the csv may contain strings
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Aggregate per clip
    print('\nAggregating features per clip...')
    rows  = []
    clips = df['clip_id'].unique()
    for i, cid in enumerate(clips, 1):
        if i % 50 == 0 or i == len(clips):
            print(f'   {i}/{len(clips)}', end='\r')
        rows.append(build_clip_features(df[df['clip_id'] == cid].copy()))

    print(f'\n   {len(rows)} clips processed')
    master = pd.DataFrame(rows)

    # Merge proximal-distal
    if proxdist_path.exists():
        print(f'\nMerging with proximal_distal_sequence.csv...')
        pd_df   = pd.read_csv(proxdist_path)
        pd_cols = ['clip_id'] + [c for c in pd_df.columns if c != 'clip_id' and c not in master.columns]
        master  = master.merge(pd_df[pd_cols], on='clip_id', how='left')
        print(f'   +{len(pd_cols)-1} features')
    else:
        print(f'WARNING:  proximal_distal_sequence.csv not found - skipping')

    print(f'\n[OK] {len(master)} clips x {len(master.columns)} features')

    # Save -- this is the file used by 03_statistical_analysis/ and 04_models/
    out_path = out_dir / 'clips_master.csv'
    master.to_csv(out_path, index=False, sep=',', decimal='.', float_format='%.4f')
    print(f'{out_path}')

    print(f'\nDone!')
    return master


# -- CLI -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Aggregates biomechanics_dataset_complete.csv into a per-clip dataset.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--data',   '-d', default='.', help='Folder with the input CSVs')
    ap.add_argument('--output', '-o', default=None, help='Output folder (default: same as --data)')
    args = ap.parse_args()
    build_master(args.data, args.output or args.data)


if __name__ == '__main__':
    main()
