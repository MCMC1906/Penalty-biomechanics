"""
pose_extractor.py
=================
Body keypoint extraction with MediaPipe Pose (Tasks API >= 0.10.13).

For each video frame, extracts 33 keypoints (x, y in pixels) and the
confidence score of each one. This is the first step of the biomechanics pipeline.

Available models (downloaded automatically on first run):
  0 — Lite  (~9 MB)  — fast, less accurate
  1 — Full  (~20 MB) — balanced
  2 — Heavy (~29 MB) — most accurate, recommended for back-facing clips

process_video() output:
  keypoints  : (n_frames, 33, 2)  — x,y coordinates in pixels
  visibility : (n_frames, 33)     — MediaPipe confidence [0, 1]
  timestamps : (n_frames,)        — time in seconds
  fps        : float
  frame_size : (width, height)

Usage:
  extractor = PoseExtractor(model_complexity=2)
  data = extractor.process_video("clip.mp4", "clip_id")
  extractor.close()
"""

import cv2 # type: ignore
import numpy as np    # type: ignore
import urllib.request
import os

# -- Model URLs ----------------------------------------------
_MODELS = {                                                                    # available models, index matches model_complexity
    0: ("pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"),
    1: ("pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"),
    2: ("pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"),
}

def _ensure_model(complexity: int = 2, dest_dir: str = ".") -> str:
    filename, url = _MODELS[complexity]
    path = os.path.join(dest_dir, filename)
    if not os.path.exists(path):                                           # only downloads on the 1st run
        print(f"Downloading MediaPipe model ({filename}) ...")
        urllib.request.urlretrieve(url, path)                     # download to disk
        print(f"Download complete → {path}")
    return path

# -- Indices of the 33 keypoints ---------------------------------
KP = {
    'nose'          : 0,
    'left_shoulder' : 11, 'right_shoulder' : 12,
    'left_elbow'    : 13, 'right_elbow'    : 14,
    'left_wrist'    : 15, 'right_wrist'    : 16,
    'left_hip'      : 23, 'right_hip'      : 24,
    'left_knee'     : 25, 'right_knee'     : 26,
    'left_ankle'    : 27, 'right_ankle'    : 28,
    'left_heel'     : 29, 'right_heel'     : 30,
    'left_toe'      : 31, 'right_toe'      : 32,
}
N_KPS = 33

# -- Skeleton connections for visualization ------------------------
_SKELETON = [
    (11, 12),                          # shoulders
    (11, 13), (13, 15),                # left arm
    (12, 14), (14, 16),                # right arm
    (11, 23), (12, 24), (23, 24),      # torso
    (23, 25), (25, 27),                # left thigh + leg
    (27, 29), (27, 31), (29, 31),      # left foot
    (24, 26), (26, 28),                # right thigh + leg
    (28, 30), (28, 32), (30, 32),      # right foot
]

# Colors per segment (BGR)
_C_LEFT  = (255, 180,  50)   # orange - left side
_C_RIGHT = ( 50, 180, 255)   # yellow - right side
_C_TORSO = (100, 255, 100)   # green  - torso
_C_KP    = (  0,   0, 255)   # red    - keypoints
_C_KP_LO = (120, 120, 120)   # gray   - low confidence

_SEGMENT_COLORS = {
    (11,12): _C_TORSO,
    (11,13): _C_LEFT,  (13,15): _C_LEFT,
    (12,14): _C_RIGHT, (14,16): _C_RIGHT,
    (11,23): _C_TORSO, (12,24): _C_TORSO, (23,24): _C_TORSO,
    (23,25): _C_LEFT,  (25,27): _C_LEFT,
    (27,29): _C_LEFT,  (27,31): _C_LEFT,  (29,31): _C_LEFT,
    (24,26): _C_RIGHT, (26,28): _C_RIGHT,
    (28,30): _C_RIGHT, (28,32): _C_RIGHT, (30,32): _C_RIGHT,
}


def _draw_pose(frame: np.ndarray,
               kps: np.ndarray, vis: np.ndarray,
               clip_id: str, frame_idx: int, total: int,
               vis_threshold: float = 0.3) -> np.ndarray:
    disp = frame.copy()                    # copy so the original frame isn't changed

    # Connections
    for (i, j) in _SKELETON:
        if vis[i] > vis_threshold and vis[j] > vis_threshold:                # only draws if both are above the confidence threshold (has no effect on the results, only on the visualization)
            p1 = (int(kps[i, 0]), int(kps[i, 1]))
            p2 = (int(kps[j, 0]), int(kps[j, 1]))
            color = _SEGMENT_COLORS.get((i, j), _C_TORSO)
            cv2.line(disp, p1, p2, color, 2, cv2.LINE_AA)

    # Keypoints
    for k in range(N_KPS):
        if np.isnan(kps[k, 0]):
            continue                                                # keypoint with no detection, skip
        pt    = (int(kps[k, 0]), int(kps[k, 1]))
        color = _C_KP if vis[k] > vis_threshold else _C_KP_LO           # good confidence = red, low = gray
        cv2.circle(disp, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(disp, pt, 4, (255, 255, 255), 1, cv2.LINE_AA)

    # Info bar at the top
    h, w = disp.shape[:2]
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, disp, 0.35, 0, disp)                # semi-transparent bar

    name = clip_id[-55:] if len(clip_id) > 55 else clip_id                  # uses only the last 55 characters if clip_id is too long
    cv2.putText(disp, f"{name}  |  frame {frame_idx + 1}/{total}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (220, 220, 220), 1, cv2.LINE_AA)

    return disp


class PoseExtractor:
    """
    Extracts 33 2D keypoints per frame using the MediaPipe Tasks API.
    Compatible with mediapipe >= 0.10.13.

    model_complexity : 0=lite (fast)  1=full  2=heavy (accurate)
    show_preview      : True to show a window with the skeleton in real time
    """

    def __init__(self, model_complexity: int = 2,
                 min_detection_confidence: float = 0.4,     # minimum confidence to detect a pose
                 min_tracking_confidence: float = 0.4,   # minimum confidence to keep tracking across frames
                 model_dir: str = ".",              # folder where the .task model is stored
                 show_preview: bool = False):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            raise ImportError("Install mediapipe: pip install mediapipe")

        self._mp = mp
        self.show_preview = show_preview
        self._win = "MediaPipe Pose Analysis"       # preview window name
        self._preview_open = False                     # tracks whether the window is open

        model_path = _ensure_model(model_complexity, model_dir)  # downloads the model if needed

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,       # video mode, keeps state across frames for tracking
            num_poses=1,                               # detects only one person
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=0.4,             # minimum confidence
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)  # initializes the model
        self.kp = KP                                 # exposes the keypoint dict for external use
        print(f"MediaPipe Pose ready (complexity={model_complexity})")

        if self.show_preview:
            cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win, 960, 600)
            self._preview_open = True
            print("Preview window open  [Q = disable preview]")

    def process_video(self, video_path: str, clip_id: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {video_path}")

        fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0                         # uses 25fps if the video has no fps defined
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        kps_list, vis_list = [], []                                # accumulate the frame-by-frame results
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret: break                          # end of video or read error

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe requires RGB, while OpenCV reads in BGR
            mp_img = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb,
            )
            ts_ms  = int(frame_idx * 1000 / fps)      # timestamp in ms, required in VIDEO mode
            result = self._landmarker.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]         # detected set of landmarks
                kps = np.array(
                    [[lm.x * frame_w, lm.y * frame_h] for lm in lms],  # normalized coordinates [0,1]
                    dtype=np.float32,
                )
                vis = np.array(
                    [getattr(lm, 'visibility', 1.0) for lm in lms],  # confidence per keypoint
                    dtype=np.float32,
                )
            else:
                kps = np.full((N_KPS, 2), np.nan, dtype=np.float32)  # no detection - nan to signal absence
                vis = np.zeros(N_KPS, dtype=np.float32)               # zero confidence

            kps_list.append(kps)
            vis_list.append(vis)

            if self.show_preview and self._preview_open:
                disp = _draw_pose(frame, kps, vis,
                                  clip_id, frame_idx, total)

                # Rescale for the window
                max_h, max_w = 600, 960
                h, w = disp.shape[:2]
                s = min(max_w / w, max_h / h)
                if s < 1.0:
                    disp = cv2.resize(disp,
                                      (int(w * s), int(h * s)),
                                      interpolation=cv2.INTER_AREA)

                cv2.imshow(self._win, disp)
                key = cv2.waitKey(1) & 0xFF                     # 1ms, doesn't block processing
                if key in (ord('q'), ord('Q')):                     # Q disables the preview without stopping processing
                    cv2.destroyWindow(self._win)
                    self._preview_open = False
                    print("  Preview disabled.")

            frame_idx += 1

        cap.release()

        n = len(kps_list)
        return {
            'clip_id'    : clip_id,
            'keypoints'  : np.stack(kps_list),   # (n_frames, 33, 2) - x,y coordinates in pixels
            'visibility' : np.stack(vis_list),   # (n_frames, 33)    - confidence per keypoint
            'timestamps' : np.arange(n) / fps,   # (n_frames,)       - time in seconds
            'fps'        : fps,
            'frame_size' : (frame_w, frame_h),
        }

    def close(self):
        # Call once after all clips to free the model from memory
        self._landmarker.close()
        if self._preview_open:
            cv2.destroyWindow(self._win)
            self._preview_open = False
