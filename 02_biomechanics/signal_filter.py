"""
signal_filter.py
================
Cleans up the keypoints extracted by MediaPipe.

Philosophy: trust MediaPipe, intervene as little as possible.
  1. Interpolates ONLY frames with near-zero confidence (< vis_threshold) - values below this indicate MediaPipe detected nothing.
  2. Light Savitzky-Golay (window=3) to remove sub-pixel jitter without destroying the angular-velocity peaks in frames 14-19.
"""

import numpy as np
from scipy.signal import savgol_filter


class SignalFilter:
    """
    vis_threshold : minimum MediaPipe confidence to accept a keypoint, below this the frame is interpolated from its neighbors.
    window        : Savitzky-Golay window — the larger it is the more it smooths (3 = minimum useful, preserves velocity peaks)
    poly          : SG polynomial degree — 2 is enough for biomechanical signal.
    """

    def __init__(self, vis_threshold: float = 0.25,
                 window: int = 3, poly: int = 2):
        self.vis_threshold = vis_threshold
        self.window        = window
        self.poly          = poly

    # -- Linear interpolation where visibility is low -------------------------

    def _interpolate(self, keypoints: np.ndarray,
                     visibility: np.ndarray) -> np.ndarray:
        """Interpolates only frames with confidence below the threshold or NaN."""
        result = keypoints.copy()
        n_kps  = keypoints.shape[1]

        for k in range(n_kps):
            bad = (visibility[:, k] < self.vis_threshold) | \
np.any(np.isnan(keypoints[:, k, :]), axis=1)                                                # low confidence OR NaN coordinates
            valid_idx = np.where(~bad)[0]              # indices of the frames with good confidence

            if len(valid_idx) == 0:
                result[:, k, :] = 0.0                  # keypoint never detected — fill with 0
                continue                                # will be ignored by kinematics via NaN/visibility

            if not np.any(bad):
                continue                                # nothing to interpolate for this keypoint

            for coord in range(2):                     # interpolates x and y separately
                series = result[:, k, coord].copy()
                for i in np.where(bad)[0]:
                    before = valid_idx[valid_idx < i]
                    after  = valid_idx[valid_idx > i]
                    if len(before) and len(after):
                        f0, f1 = before[-1], after[0]
                        t = (i - f0) / (f1 - f0)      # relative position between the two valid frames
                        series[i] = series[f0] * (1-t) + series[f1] * t  # linear interpolation
                    elif len(before):
                        series[i] = np.nan
                    else:
                        series[i] = np.nan
                result[:, k, coord] = series

        return result

    # -- Light Savitzky-Golay -----------------------------------------------

    def _savgol(self, keypoints: np.ndarray) -> np.ndarray:
        n = keypoints.shape[0]
        w = self.window

        if n < w:                                                    # ensure a valid, odd window
            w = n if n % 2 == 1 else max(n - 1, 3)                      # shrinks if the clip has fewer frames than the window
        w = max(w, self.poly + 2)                                # minimum window = polynomial degree
        if w % 2 == 0:
            w = max(w - 1, 3)                                                   # Savitzky-Golay requires an odd window

        result = keypoints.copy()

        for k in range(keypoints.shape[1]):
            for c in range(2):                                              # processes x and y separately
                series   = keypoints[:, k, c].copy()
                nan_mask = np.isnan(series)                                             # records where the NaNs are so they can be restored afterward

                if nan_mask.all():
                    result[:, k, c] = 0.0              # keypoint completely absent
                    continue

                # Fill NaN via linear interpolation before the SG
                if nan_mask.any():
                    valid = np.where(~nan_mask)[0]
                    for idx in np.where(nan_mask)[0]:
                        b = valid[valid < idx]
                        a = valid[valid > idx]
                        if len(b) and len(a):
                            f0, f1 = b[-1], a[0]
                            t = (idx - f0) / (f1 - f0)
                            series[idx] = series[f0] * (1-t) + series[f1] * t
                        elif len(b):
                            series[idx] = series[b[-1]]                 # temporary, will be restored to NaN after the SG
                        else:
                            series[idx] = series[a[0]]                      # temporary, will be restored to NaN after the SG

                smoothed = savgol_filter(series, w, self.poly)

                smoothed[nan_mask] = np.nan                             # restores NaN where the keypoint was invalid
                result[:, k, c] = smoothed

        return result

    def process(self, keypoints: np.ndarray,
                visibility: np.ndarray) -> np.ndarray:
        """
        keypoints  : (n_frames, 33, 2)
        visibility : (n_frames, 33)
        """
        clean    = self._interpolate(keypoints, visibility)  # step 1: interpolate low-confidence frames
        filtered = self._savgol(clean)                       # step 2: light smoothing
        return filtered
