"""
kinematics.py
=============
Computes all the biomechanical metrics from the keypoints extracted by MediaPipe and filtered by signal_filter.

Outlier detection on the computed angles:
  1. Anatomical limits: angles outside [2 degrees, 178 degrees] are impossible
  2. Inter-frame jumps: variation > 60 degrees/frame is impossible at 25fps
  Outliers are marked as NaN, they are not interpolated.

Fix for the gradient with NaN:
  Uses linear interpolation before the gradient and restores the NaN afterwards.
"""

import numpy as np
from pose_extractor import KP

ANGLE_MIN    =   2.0   # degrees - below this is anatomically impossible
ANGLE_MAX    = 178.0   # degrees - above this is anatomically impossible
MAX_JUMP_DEG =  60.0   # degrees/frame - above this is an outlier at 25fps
MAX_VEL_DEG_S    = 2000.0    # degrees/s  - real p99 ~1200, above this is an artifact
MAX_ACCEL_DEG_S2 = 35000.0   # degrees/s2 - real p99 ~32500, the limit sits above the p99, max observed 122315 is an artifact


# -- Base vector functions ----------------------------------------

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b between b->a and b->c. Returns degrees [0,180] or NaN."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return np.nan                                      # vectors too short, overlapping keypoints
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def _angle_vecs(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two 2D vectors. Returns degrees [0,180] or NaN."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-4 or n2 < 1e-4:
        return np.nan
    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def _trunk_inclination(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Trunk inclination relative to the vertical axis. 0 degrees = vertical."""
    trunk = shoulder_mid - hip_mid
    norm  = np.linalg.norm(trunk)
    if norm < 1e-4:
        return np.nan
    trunk_n = trunk / norm
    cos_t   = np.clip(np.dot(trunk_n, np.array([0.0, -1.0])), -1.0, 1.0)  # vertical in pixel coordinates = (0,-1)
    return float(np.degrees(np.arccos(cos_t)))


# -- Outlier detection on the angles ------------------------------

def _clean_angle_series(angles: np.ndarray) -> np.ndarray:
    """Marks only physically impossible values as NaN, does not interpolate."""
    out = angles.copy()
    out = np.where((out < ANGLE_MIN) | (out > ANGLE_MAX), np.nan, out)  # anatomical limits
    for i in range(1, len(out)):
        if np.isnan(out[i]) or np.isnan(out[i-1]):
            continue
        if abs(out[i] - out[i-1]) > MAX_JUMP_DEG:
            out[i] = np.nan                                # impossible jump
    return out


def _clean_velocity_series(vel: np.ndarray, max_val: float) -> np.ndarray:
    """Marks velocities and accelerations above the physically possible limit as NaN."""
    return np.where(np.abs(vel) > max_val, np.nan, vel)


# -- NaN-safe gradient ---------------------------------------

def _gradient_nan_safe(series: np.ndarray, dt: float) -> np.ndarray:
    """
    Gradient without artificial spikes at the NaN boundaries.
    Fills NaN via linear interpolation, computes the gradient, restores NaN where the original angle was invalid.
    """
    nan_mask = np.isnan(series)
    if nan_mask.all():
        return np.full_like(series, np.nan)

    filled = series.copy()
    valid  = np.where(~nan_mask)[0]
    for i in np.where(nan_mask)[0]:
        before = valid[valid < i]; after = valid[valid > i]
        if len(before) and len(after):
            f0, f1 = before[-1], after[0]
            t = (i - f0) / (f1 - f0)
            filled[i] = filled[f0] * (1-t) + filled[f1] * t
        elif len(before): filled[i] = filled[before[-1]]
        else:             filled[i] = filled[after[0]]

    grad = np.gradient(filled, dt)
    grad[nan_mask] = np.nan                                # restores NaN where the angle was invalid
    return grad


# -- Automatic kicking-leg detection ------------------------

def detect_kicking_leg(keypoints: np.ndarray) -> str:
    """The kicking foot moves more, compares the total ankle displacement."""      # the metadata should already indicate the kicking foot, but if not this function is used
    l = keypoints[:, KP['left_ankle'],  :]
    r = keypoints[:, KP['right_ankle'], :]

    def _disp(pts):
        d = np.diff(np.nan_to_num(pts), axis=0)
        return float(np.sum(np.linalg.norm(d, axis=1)))

    return 'left' if _disp(l) > _disp(r) else 'right'


# -- Main computation ---------------------------------------------

def compute_kinematics(keypoints: np.ndarray,
                       fps: float,
                       visibility: np.ndarray = None,
                       kicking_leg: str = None) -> dict:
    """
    keypoints  : (n_frames, 33, 2)  in pixels, already filtered by signal_filter
    fps        : frames per second
    visibility : (n_frames, 33) MediaPipe scores [0,1]
    kicking_leg: 'left' | 'right' | None  (auto-detected if None)
    """
    if kicking_leg is None:
        kicking_leg = detect_kicking_leg(keypoints)
    support_leg = 'right' if kicking_leg == 'left' else 'left'

    n  = keypoints.shape[0]
    dt = 1.0 / fps

    def pt(name):
        return keypoints[:, KP[name], :]                  # extracts the time series of a keypoint (n, 2)

    l_sh  = pt('left_shoulder');  r_sh  = pt('right_shoulder')
    l_el  = pt('left_elbow');     r_el  = pt('right_elbow')
    l_wr  = pt('left_wrist');     r_wr  = pt('right_wrist')
    l_hip = pt('left_hip');       r_hip = pt('right_hip')
    l_kn  = pt('left_knee');      r_kn  = pt('right_knee')
    l_an  = pt('left_ankle');     r_an  = pt('right_ankle')
    l_toe = pt('left_toe');       r_toe = pt('right_toe')

    sh_mid  = (l_sh  + r_sh)  / 2                         # shoulder midpoint: torso reference
    hip_mid = (l_hip + r_hip) / 2                         # hip midpoint: center-of-mass proxy

    if kicking_leg == 'left':
        kick_hip, kick_kn, kick_an, kick_toe = l_hip, l_kn, l_an, l_toe
        supp_hip, supp_kn, supp_an, supp_toe = r_hip, r_kn, r_an, r_toe
    else:
        kick_hip, kick_kn, kick_an, kick_toe = r_hip, r_kn, r_an, r_toe
        supp_hip, supp_kn, supp_an, supp_toe = l_hip, l_kn, l_an, l_toe

    # -- A. Joint angles ------------------------------------

    kick_knee_angle  = _clean_angle_series(np.array([_angle(kick_hip[t], kick_kn[t], kick_an[t])  for t in range(n)]))
    supp_knee_angle  = _clean_angle_series(np.array([_angle(supp_hip[t], supp_kn[t], supp_an[t])  for t in range(n)]))
    kick_ankle_angle = _clean_angle_series(np.array([_angle(kick_kn[t],  kick_an[t], kick_toe[t]) for t in range(n)]))
    supp_ankle_angle = _clean_angle_series(np.array([_angle(supp_kn[t],  supp_an[t], supp_toe[t]) for t in range(n)]))
    kick_hip_angle   = _clean_angle_series(np.array([_angle(sh_mid[t],   kick_hip[t], kick_kn[t]) for t in range(n)]))
    left_elbow_angle  = _clean_angle_series(np.array([_angle(l_sh[t], l_el[t], l_wr[t]) for t in range(n)]))
    right_elbow_angle = _clean_angle_series(np.array([_angle(r_sh[t], r_el[t], r_wr[t]) for t in range(n)]))

    # -- B. Angular velocities (NaN-safe gradient) --------

    kick_angular_vel       = _clean_velocity_series(_gradient_nan_safe(kick_knee_angle,  dt), MAX_VEL_DEG_S)
    kick_hip_angular_vel   = _clean_velocity_series(_gradient_nan_safe(kick_hip_angle,   dt), MAX_VEL_DEG_S)
    kick_ankle_angular_vel = _clean_velocity_series(_gradient_nan_safe(kick_ankle_angle, dt), MAX_VEL_DEG_S)
    kick_angular_accel     = _clean_velocity_series(_gradient_nan_safe(kick_angular_vel, dt), MAX_ACCEL_DEG_S2)  # acceleration = joint whip

    # -- B. Proximal-distal sequence (hip -> knee -> ankle) --

    _abs_hip   = np.abs(np.nan_to_num(kick_hip_angular_vel))
    _abs_knee  = np.abs(np.nan_to_num(kick_angular_vel))
    _abs_ankle = np.abs(np.nan_to_num(kick_ankle_angular_vel))

    peak_hip_frame   = int(np.argmax(_abs_hip))
    peak_knee_frame  = int(np.argmax(_abs_knee))
    peak_ankle_frame = int(np.argmax(_abs_ankle))

    peak_hip_vel   = float(_abs_hip[peak_hip_frame])
    peak_knee_vel  = float(_abs_knee[peak_knee_frame])
    peak_ankle_vel = float(_abs_ankle[peak_ankle_frame])

    dt_hip_to_knee   = (peak_knee_frame  - peak_hip_frame)  / fps  # positive = correct order
    dt_knee_to_ankle = (peak_ankle_frame - peak_knee_frame) / fps
    proximal_distal_valid = bool(peak_hip_frame <= peak_knee_frame <= peak_ankle_frame)

    # -- Pixel -> meter scale --------------------------------------
    # Assumes average human torso = 0.50m (shoulder -> hip) - limitation: +-10% depending on height
    torso_px     = np.nanmean(np.linalg.norm(sh_mid - hip_mid, axis=1))
    px_per_meter = torso_px / 0.50 if torso_px > 1.0 else 1.0

    # -- B. Running speed (proxy: hip displacement) ----
    hip_disp_px       = np.linalg.norm(np.diff(np.nan_to_num(hip_mid), axis=0), axis=1)
    hip_speed         = (hip_disp_px / px_per_meter) * fps            # m/s
    running_speed_kmh = np.concatenate([[hip_speed[0]], hip_speed]) * 3.6

    # -- B. Kicking-foot linear speed ----------------------
    toe_disp_px        = np.linalg.norm(np.diff(np.nan_to_num(kick_toe), axis=0), axis=1)
    kick_foot_speed_ms = np.concatenate([[toe_disp_px[0] * fps / px_per_meter],
                                          toe_disp_px * fps / px_per_meter])

    # -- C. Trunk posture --------------------------------------

    trunk_incl = np.array([_trunk_inclination(sh_mid[t], hip_mid[t]) for t in range(n)])

    trunk_vec = sh_mid - hip_mid
    denom = np.where(np.abs(trunk_vec[:, 1]) > 1e-4, -trunk_vec[:, 1], 1e-4)
    lateral_trunk_lean = np.degrees(np.arctan2(trunk_vec[:, 0], denom))  # positive = lean right

    shoulder_line = l_sh - r_sh                                        # vector between shoulders (n, 2)
    hip_line      = l_hip - r_hip                                      # vector between hips (n, 2)
    torso_torsion_angle = np.array([_angle_vecs(shoulder_line[t], hip_line[t]) for t in range(n)])

    # -- D. Average MediaPipe confidence ---------------------------
    mean_vis = np.mean(visibility, axis=1) if visibility is not None else np.ones(n)

    # -- Support-foot position ------------------------------------
    supp_ankle_kp = KP[f'{support_leg}_ankle']
    kick_ankle_kp = KP[f'{kicking_leg}_ankle']
    supp_pos_last = keypoints[-2, supp_ankle_kp, :]                    # second-to-last frame: foot already planted
    ball_est      = keypoints[ 0, kick_ankle_kp, :]                    # ball proxy = kicking ankle at frame 0

    return {
        'kicking_leg'              : kicking_leg,
        'support_leg'              : support_leg,
        'kick_knee_angle'          : kick_knee_angle,
        'supp_knee_angle'          : supp_knee_angle,
        'kick_ankle_angle'         : kick_ankle_angle,
        'supp_ankle_angle'         : supp_ankle_angle,
        'kick_hip_angle'           : kick_hip_angle,
        'left_elbow_angle'         : left_elbow_angle,
        'right_elbow_angle'        : right_elbow_angle,
        'kick_hip_angular_vel'     : kick_hip_angular_vel,
        'kick_angular_vel'         : kick_angular_vel,
        'kick_ankle_angular_vel'   : kick_ankle_angular_vel,
        'kick_angular_accel'       : kick_angular_accel,
        'kick_foot_speed_ms'       : kick_foot_speed_ms,
        'running_speed_kmh'        : running_speed_kmh,
        'trunk_inclination'        : trunk_incl,
        'lateral_trunk_lean'       : lateral_trunk_lean,
        'torso_torsion_angle'      : torso_torsion_angle,
        'mean_visibility_score'    : mean_vis,
        'peak_hip_vel'             : peak_hip_vel,
        'peak_knee_vel'            : peak_knee_vel,
        'peak_ankle_vel'           : peak_ankle_vel,
        'peak_hip_frame'           : peak_hip_frame,
        'peak_knee_frame'          : peak_knee_frame,
        'peak_ankle_frame'         : peak_ankle_frame,
        'dt_hip_to_knee'           : dt_hip_to_knee,
        'dt_knee_to_ankle'         : dt_knee_to_ankle,
        'proximal_distal_valid'    : proximal_distal_valid,
        'support_foot_x'           : float(np.nan_to_num(supp_pos_last[0] - ball_est[0])),
        'support_foot_y'           : float(np.nan_to_num(ball_est[1] - supp_pos_last[1])),
        'timestamps'               : np.arange(n) / fps,
    }
