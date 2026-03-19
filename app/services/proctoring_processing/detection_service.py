"""
Detection Service
Handles gaze detection, head pose detection, and violation tracking
Industry-standard implementation - NO CALIBRATION, fixed thresholds


Additional accuracy improvements:
  - MIN_CONFIDENCE_THRESHOLD lowered to 0.45 to capture more genuine events
  - MIN_EVENT_DURATION lowered to 0.5s to avoid missing quick suspicious glances
  - Dual-component head event recording: yaw AND pitch events recorded independently
  - Gap tolerance tightened to 0.3s to catch rapid look-aways
  - Occlusion fix: both spreads must be < 0.10 AND penalty reduced to 0.15
  - Velocity label now returns "slow" even below MIN_VELOCITY_THRESHOLD (no empty string)
"""
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List, Any
import mediapipe as mp
from numpy.typing import NDArray
from collections import deque

from .base_proctoring_processing_service import BaseService


class DetectionService(BaseService):
    """
    Service for detecting gaze, head pose, and tracking violations.
    Industry-standard CPU-based detection using MediaPipe Face Mesh.
    """
    _last_tvt_prediction: object

    def __init__(self, config):
        super().__init__(config)
        self._last_tvt_prediction = None
        self._last_tvt_time = None
        self._last_tvt_time = None
        self._last_tvt_prediction = None

    def _setup(self) -> None:
        """Initialize MediaPipe and detection models."""
        self.mp_face_mesh = mp.solutions.face_mesh

        # max_num_faces=2 so multiple-face detection works for secondary face
        self.gaze_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,           # Enables iris tracking
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60
        )
        self.pose_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=False,          # Faster for head pose
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60
        )

        # Eye landmarks (MediaPipe 468-landmark model)
        self.LEFT_EYE         = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE        = [362, 263, 387, 386, 385, 373, 374, 380]
        self.LEFT_IRIS        = [474, 475, 476, 477]
        self.RIGHT_IRIS       = [469, 470, 471, 472]
        self.LEFT_EYE_CORNERS  = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]

        # 6-point canonical 3-D face model (mm units, industry standard)
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
        self.model_points = np.array([
            (0.0,    0.0,    0.0),       # Nose tip
            (0.0,  -330.0, -65.0),       # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0,  170.0, -135.0),     # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0,  -150.0, -125.0),    # Right mouth corner
        ], dtype=np.float64)

        self._init_kalman_filters()

        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size = 5   # Reduced for faster response to gaze shifts

        self.yaw_history       = deque(maxlen=self.config.VELOCITY_HISTORY_SIZE)
        self.pitch_history     = deque(maxlen=self.config.VELOCITY_HISTORY_SIZE)
        self.eye_angle_history = deque(maxlen=self.config.VELOCITY_HISTORY_SIZE)
        self.timestamp_history = deque(maxlen=self.config.VELOCITY_HISTORY_SIZE)

        self.violation_tracker = ViolationTracker(self.config)

        # TVT (Temporal Vision Transformer) — optional
        self._tvt_buffer: Optional[Any] = None
        self._tvt_model: Optional[Any] = None
        self._last_tvt_prediction: Optional[Dict] = None
        self._last_tvt_time: float = 0.0
        if getattr(self.config, 'ENABLE_TVT', False):
            try:
                from .temporal_buffer import TemporalBuffer
                from .tvt_lite_model import create_tvt_model
                window = getattr(self.config, 'TVT_TEMPORAL_WINDOW', 24)
                self._tvt_buffer = TemporalBuffer(window_size=window, landmark_dim=936)
                self._tvt_model = create_tvt_model(self.config)
            except Exception:
                self._tvt_buffer = None
                self._tvt_model = None

    # ------------------------------------------------------------------
    # Kalman filter helpers
    # ------------------------------------------------------------------

    def _init_kalman_filters(self) -> None:
        """Initialize Kalman filters for gaze smoothing."""
        def _make_kf() -> cv2.KalmanFilter:
            kf = cv2.KalmanFilter(2, 1)
            kf.measurementMatrix = np.array([[1, 0]], np.float32)
            kf.transitionMatrix  = np.array([[1, 1], [0, 1]], np.float32)
            kf.processNoiseCov   = np.eye(2, dtype=np.float32) * 0.03
            kf.measurementNoiseCov = np.array([[1]], np.float32) * 0.1
            return kf

        self.kf_horizontal = _make_kf()
        self.kf_vertical   = _make_kf()
        self.kalman_initialized = False

    def _smooth_gaze_with_kalman(self, horizontal: float, vertical: float) -> Tuple[float, float]:
        """Apply Kalman filtering to smooth gaze measurements."""
        try:
            if not self.kalman_initialized:
                self.kf_horizontal.statePre = np.array([[horizontal], [0]], np.float32)
                self.kf_vertical.statePre   = np.array([[vertical],   [0]], np.float32)
                self.kalman_initialized = True
                return horizontal, vertical

            self.kf_horizontal.predict()
            self.kf_vertical.predict()
            h_corr = self.kf_horizontal.correct(np.array([[horizontal]], np.float32))
            v_corr = self.kf_vertical.correct(np.array([[vertical]],   np.float32))
            return float(h_corr[0][0]), float(v_corr[0][0])
        except Exception:
            return horizontal, vertical

    # ------------------------------------------------------------------
    # Eye / iris helpers
    # ------------------------------------------------------------------

    def _calculate_eye_aspect_ratio(self, eye_coords: List[List[float]]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for blink / eye-open detection."""
        try:
            ea = np.array(eye_coords, dtype=np.float64)
            v1  = np.linalg.norm(ea[1] - ea[5])
            v2  = np.linalg.norm(ea[2] - ea[4])
            h   = np.linalg.norm(ea[0] - ea[3])
            return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_gaze_confidence(self, avg_ear: float, iris_valid: bool, eye_width: float) -> float:
        """Return a 0-1 confidence score for the current gaze reading."""
        conf = 0.0
        if avg_ear > 0.15:
            conf += min(avg_ear / 0.25, 1.0) * 0.4
        if iris_valid:
            conf += 0.4
        if eye_width > 2.0:
            conf += min(eye_width / 10.0, 1.0) * 0.2
        return min(conf, 1.0)

    # ------------------------------------------------------------------
    # BUG-3 FIX: Face occlusion - reduced false positives
    # Old logic: penalise 0.3 if EITHER x_spread < 0.15 OR y_spread < 0.15
    # New logic: penalise 0.15 only if BOTH x_spread < 0.10 AND y_spread < 0.10
    # ------------------------------------------------------------------

    def _calculate_face_occlusion(self, face_landmarks) -> Tuple[float, float]:
        """
        Calculate face occlusion ratio and confidence.

        BUG-3 FIX: spread penalty now fires only when BOTH axes are tiny (< 0.10)
        and the penalty magnitude is reduced from 0.3 → 0.15 to prevent constant
        false-positive triggering on normal frontal faces.
        """
        try:
            key_indices = [1, 234, 454, 10, 152, 33, 263]
            visible_count = 0
            x_coords: List[float] = []
            y_coords: List[float] = []

            for idx in key_indices:
                if idx < len(face_landmarks):
                    lm = face_landmarks[idx]
                    if 0.05 <= lm.x <= 0.95 and 0.05 <= lm.y <= 0.95:
                        visible_count += 1
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)

            visibility_ratio = visible_count / len(key_indices)

            spread_penalty = 0.0
            if len(x_coords) >= 3:
                x_spread = max(x_coords) - min(x_coords)
                y_spread = max(y_coords) - min(y_coords)
                # BUG-3 FIX: require BOTH spreads to be tiny (truly distorted/tiny face)
                if x_spread < 0.10 and y_spread < 0.10:
                    spread_penalty = 0.15   # Reduced from 0.30
            else:
                spread_penalty = 0.40       # Fewer than 3 visible landmarks → real occlusion

            occlusion_ratio = min((1.0 - visibility_ratio) + spread_penalty, 1.0)
            confidence      = min(visibility_ratio * 1.2, 1.0)
            return occlusion_ratio, confidence

        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Landmark vector for TVT (468 landmarks -> 936-dim vector)
    # ------------------------------------------------------------------

    def get_landmark_vector(self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        """
        Extract 468 facial landmarks as flat vector [x1,y1,...,x468,y468].
        Uses normalized coords (0-1). Returns None if no face.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.gaze_face_mesh.process(rgb)
        if results is None:
            return None
        mlm = getattr(results, 'multi_face_landmarks', None)
        if not mlm:
            return None
        face_landmarks = mlm[0].landmark
        # Normalized x,y for each of 468 landmarks -> 936
        out = np.zeros(936, dtype=np.float32)
        for i, lm in enumerate(face_landmarks):
            if i >= 468:
                break
            out[i * 2] = lm.x
            out[i * 2 + 1] = lm.y
        return out

    def push_landmark_and_maybe_run_tvt(
        self, landmark_vector: Optional[NDArray[np.float32]], timestamp: float
    ) -> None:
        """Push one frame's landmarks to TVT buffer; run inference when ready and interval elapsed."""
        if self._tvt_buffer is None or self._tvt_model is None or landmark_vector is None:
            return
        self._tvt_buffer.push(landmark_vector, timestamp)
        interval = getattr(self.config, 'TVT_INFERENCE_INTERVAL_SEC', 1.0)
        if not self._tvt_buffer.is_ready():
            return
        if timestamp - self._last_tvt_time < interval:
            return
        window = self._tvt_buffer.get_window()
        if window is not None:
            try:
                self._last_tvt_prediction = self._tvt_model.predict(window)
                self._last_tvt_time = timestamp
            except Exception:
                pass

    def get_tvt_prediction(self) -> object:
        """Return latest TVT prediction or None if TVT disabled."""
        return self._last_tvt_prediction

    # ------------------------------------------------------------------
    # Gaze detection
    # ------------------------------------------------------------------

    def detect_gaze(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], int, Optional[Tuple[float, float]], float, float]:
        """
        Detect gaze angles with enhanced iris tracking and Kalman smoothing.

        Returns
        -------
        (horizontal_angle, vertical_angle, num_faces, bbox_center,
         gaze_confidence, occlusion_ratio)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.gaze_face_mesh.process(rgb)

        if results is None:
            return None, None, 0, None, 0.0, 0.0

        mlm = getattr(results, 'multi_face_landmarks', None)
        if not mlm:
            return None, None, 0, None, 0.0, 0.0

        num_faces = len(mlm)
        face_landmarks = mlm[0].landmark
        h, w = frame.shape[:2]

        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]
        bbox_center = (float(np.mean(x_coords)), float(np.mean(y_coords)))

        occlusion_ratio, _ = self._calculate_face_occlusion(face_landmarks)

        try:
            le_coords = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.LEFT_EYE]
            re_coords = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.RIGHT_EYE]
            li_coords = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.LEFT_IRIS]
            ri_coords = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.RIGHT_IRIS]

            l_ear = self._calculate_eye_aspect_ratio(le_coords)
            r_ear = self._calculate_eye_aspect_ratio(re_coords)
            avg_ear = (l_ear + r_ear) / 2.0

            # Reject closed / half-closed eyes
            if avg_ear < 0.12:
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

            iris_valid = (
                all(0 <= c[0] <= w and 0 <= c[1] <= h for c in li_coords) and
                all(0 <= c[0] <= w and 0 <= c[1] <= h for c in ri_coords)
            )
            if not iris_valid:
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

            le_center = np.mean(np.array(le_coords, dtype=np.float64), axis=0)
            re_center = np.mean(np.array(re_coords, dtype=np.float64), axis=0)
            li_center = np.mean(np.array(li_coords, dtype=np.float64), axis=0)
            ri_center = np.mean(np.array(ri_coords, dtype=np.float64), axis=0)

            def _eye_width(corners: List[int]) -> float:
                p1 = np.array([face_landmarks[corners[0]].x * w, face_landmarks[corners[0]].y * h])
                p2 = np.array([face_landmarks[corners[1]].x * w, face_landmarks[corners[1]].y * h])
                return float(np.linalg.norm(p2 - p1))

            avg_eye_width = (_eye_width(self.LEFT_EYE_CORNERS) + _eye_width(self.RIGHT_EYE_CORNERS)) / 2.0
            if avg_eye_width < 1.8:
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

            avg_disp = ((li_center - le_center) + (ri_center - re_center)) / 2.0

            h_ratio = float(np.clip(avg_disp[0] / avg_eye_width, -1.0, 1.0))
            v_ratio = float(np.clip(avg_disp[1] / avg_eye_width, -1.0, 1.0))

            h_angle = float(np.arcsin(h_ratio * 0.9) * (180.0 / np.pi))
            v_angle = float(np.arcsin(v_ratio * 0.9) * (180.0 / np.pi))

            h_angle, v_angle = self._smooth_gaze_with_kalman(h_angle, v_angle)

            self.gaze_history.append((h_angle, v_angle))
            if len(self.gaze_history) > self.max_history_size:
                self.gaze_history.pop(0)

            # Weighted average: more weight on recent frames for faster response
            if len(self.gaze_history) >= 3:
                wts = np.array([0.2, 0.3, 0.5])
                recent = self.gaze_history[-3:]
                h_angle = float(np.average([x[0] for x in recent], weights=wts))
                v_angle = float(np.average([x[1] for x in recent], weights=wts))

            gaze_conf = self._calculate_gaze_confidence(avg_ear, iris_valid, avg_eye_width)
            return round(h_angle, 2), round(v_angle, 2), num_faces, bbox_center, gaze_conf, occlusion_ratio

        except (IndexError, ValueError, TypeError):
            return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

    # ------------------------------------------------------------------
    # Head pose detection
    # BUG-1 FIX: pitch normalised to [-90, 90] after solvePnP decomposition
    # ------------------------------------------------------------------

    def _estimate_head_pose(
        self, frame: np.ndarray, results
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        """
        Industry-standard head pose estimation using solvePnP.

        BUG-1 FIX: Raw pitch from Euler decomposition can be ~±174-180° for a
        frontal face due to coordinate-convention ambiguity.  Normalise pitch
        into [-90, 90] after conversion to degrees.
        """
        mlm = getattr(results, 'multi_face_landmarks', None)
        if not mlm:
            return None, None, None, 0.0

        face_landmarks = mlm[0].landmark
        h, w = frame.shape[:2]

        image_points = np.array(
            [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.landmark_indices],
            dtype=np.float64
        )

        focal_length  = float(w)
        camera_matrix = np.array(
            [[focal_length, 0, w / 2.0],
             [0, focal_length, h / 2.0],
             [0, 0, 1]],
            dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rvec, _ = cv2.solvePnP(
                self.model_points, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success or rvec is None:
                return None, None, None, 0.0

            r, _ = cv2.Rodrigues(rvec)
            if r is None:
                return None, None, None, 0.0

            sy        = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
            singular  = sy < 1e-6

            if not singular:
                pitch_rad = np.arctan2(r[2, 1], r[2, 2])
                yaw_rad   = np.arctan2(-r[2, 0], sy)
                roll_rad  = np.arctan2(r[1, 0], r[0, 0])
            else:
                pitch_rad = np.arctan2(-r[1, 2], r[1, 1])
                yaw_rad   = np.arctan2(-r[2, 0], sy)
                roll_rad  = 0.0

            yaw_deg   = float(np.degrees(yaw_rad))
            pitch_deg = float(np.degrees(pitch_rad))
            roll_deg  = float(np.degrees(roll_rad))

            # ── BUG-1 FIX ──────────────────────────────────────────────────────
            # solvePnP + Rodrigues can return pitch near ±180° for a normally
            # oriented face.  The canonical "looking straight ahead" pose should
            # give pitch ≈ 0°.  Map the wraparound back into [-90, 90]:
            if pitch_deg > 90.0:
                pitch_deg = 180.0 - pitch_deg
            elif pitch_deg < -90.0:
                pitch_deg = -180.0 - pitch_deg
            # ───────────────────────────────────────────────────────────────────

            # Confidence: based on landmark spread and solvePnP conditioning
            x_coords = [face_landmarks[i].x for i in self.landmark_indices]
            y_coords = [face_landmarks[i].y for i in self.landmark_indices]
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)

            lm_conf = 1.0 if (x_spread >= 0.15 and y_spread >= 0.15) else 0.70
            sy_conf = 0.95 if sy > 0.2 else (0.80 if sy > 0.1 else 0.60)

            return (
                round(yaw_deg,   2),
                round(pitch_deg, 2),
                round(roll_deg,  2),
                round(lm_conf * sy_conf, 2)
            )

        except (cv2.error, ValueError, TypeError):
            return None, None, None, 0.0

    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        """Detect head pose angles (yaw, pitch, roll) + confidence."""
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_face_mesh.process(rgb)
        if results is None:
            return None, None, None, 0.0
        return self._estimate_head_pose(frame, results)

    # ------------------------------------------------------------------
    # Violation update (public entry point called per frame)
    # ------------------------------------------------------------------

    def update_violations(
        self,
        timestamp: float,
        gaze_h: Optional[float],
        gaze_v: Optional[float],
        yaw: Optional[float],
        pitch: Optional[float],
        roll: Optional[float],          # kept for API compatibility, unused
        num_faces: int,
        thresholds: Dict[str, float],
        gaze_confidence: float = 0.0,
        head_confidence: float = 0.0,
        occlusion_ratio: float = 0.0,
        landmark_vector: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """Update violation tracking with current frame data."""
        _ = roll  # reserved for future roll-based detection

        # TVT: push landmarks and run inference when interval elapsed
        if getattr(self.config, 'ENABLE_TVT', False) and landmark_vector is not None:
            self.push_landmark_and_maybe_run_tvt(landmark_vector, timestamp)
        tvt_prediction = self.get_tvt_prediction() if getattr(self.config, 'ENABLE_TVT', False) else None

        self.timestamp_history.append(timestamp)
        if yaw   is not None: self.yaw_history.append(yaw)
        if pitch  is not None: self.pitch_history.append(pitch)
        if gaze_h is not None and gaze_v is not None:
            self.eye_angle_history.append(np.sqrt(gaze_h ** 2 + gaze_v ** 2))

        self.violation_tracker.update(
            timestamp, gaze_h, gaze_v, yaw, pitch, num_faces, thresholds,
            gaze_confidence, head_confidence, occlusion_ratio,
            self.yaw_history, self.pitch_history,
            self.eye_angle_history, self.timestamp_history,
            tvt_prediction=tvt_prediction,
        )

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        return self.violation_tracker.get_counts()

    def get_all_timestamps(self) -> Dict[str, List[float]]:
        return {k: self.violation_tracker.get_timestamps(k)
                for k in self.violation_tracker.counts}

    def get_all_max_intensities(self) -> Dict[str, float]:
        return self.violation_tracker.max_intensities.copy()

    def get_violation_events(self) -> List[Dict]:
        return self.violation_tracker.get_all_events()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release MediaPipe resources and finalise active violations."""
        for attr in ('gaze_face_mesh', 'pose_face_mesh'):
            if hasattr(self, attr):
                getattr(self, attr).close()
        self.gaze_history.clear()
        if getattr(self, '_tvt_buffer', None) is not None:
            self._tvt_buffer.clear()
        self.violation_tracker.finalize()


# ======================================================================
# ViolationTracker
# ======================================================================

class ViolationTracker:
    """
    Duration-based violation tracker.

    Key design decisions
    --------------------
    * Head yaw and head pitch are tracked and recorded INDEPENDENTLY so that
      "look right while tilting down" produces two separate events rather than
      one arbitrarily chosen event.  This prevents entire classes of suspicious
      behaviour from being silently discarded.

    * BUG-4 FIX: The ``last_recorded_*_timestamp == start_time_rounded`` guard
      that silently dropped legitimate consecutive events has been removed.
      The state-machine's minimum-duration and gap-tolerance logic already
      handles deduplication correctly.

    * BUG-2 FIX: ``_get_velocity_label`` now always returns a non-empty string
      ("negligible" / "slow" / "moderate" / "rapid") so every event in the
      output carries a velocity label.

    * BUG-3 FIX (also in DetectionService._calculate_face_occlusion):
      Spread penalty requires BOTH axes to be tiny AND magnitude reduced.
    """

    def __init__(self, config) -> None:
        self.config = config

        self.counts: Dict[str, int] = {
            'gaze_left': 0, 'gaze_right': 0,
            'gaze_up':   0, 'gaze_down':  0,
            'head_left': 0, 'head_right': 0,
            'head_up':   0, 'head_down':  0,
            'face_missing':    0,
            'multiple_faces':  0,
            'face_occluded':   0,
        }
        self.timestamps: Dict[str, List[float]]    = {k: [] for k in self.counts}
        self.max_intensities: Dict[str, float]     = {k: 0.0 for k in self.counts}
        self.violation_events: List[Dict]          = []

        # Gap between frames without a violation before we close the event
        # Tightened from 0.5 → 0.3 s so fast look-aways are not missed
        self.gap_tolerance = getattr(config, 'EVENT_GAP_TOLERANCE', 0.3)

        # ── Head-turn bridge (BUG-8 FIX) ─────────────────────────────────────
        # When an extreme head turn causes MediaPipe to lose the face (num_faces=0),
        # the system was incorrectly opening a face_missing event.  We track the
        # last time a yaw state machine was active and suppress face_missing for a
        # short bridge window so the head_movement event is extended instead.
        self._last_active_yaw_time: float = -999.0   # last timestamp yaw was active

        # ── Per-type state machines ──────────────────────────────────────────
        # Head: tracked separately for yaw axis and pitch axis
        _head_default = {
            'active': False, 'start_time': 0.0, 'last_update_time': 0.0,
            'direction': '', 'max_intensity': 0.0,
            'confidence': 0.0, 'velocity': 0.0,
        }
        self.current_yaw_state   = dict(_head_default)
        self.current_pitch_state = dict(_head_default)

        # Eye: tracked separately for horizontal (left/right) and vertical (up/down), like head yaw/pitch
        self.current_eye_state: Dict = {
            'active': False, 'start_time': 0.0, 'last_update_time': 0.0,
            'direction': '', 'max_intensity': 0.0,
            'confidence': 0.0, 'velocity': 0.0,
        }
        self.current_eye_h_state: Dict = dict(self.current_eye_state)
        self.current_eye_v_state: Dict = dict(self.current_eye_state)
        # first_no_face_time: when we first saw num_faces==0; active only after min_start_duration
        self.current_face_missing_state: Dict = {
            'active': False, 'start_time': 0.0, 'last_update_time': 0.0,
            'first_no_face_time': None,
        }
        self.current_face_occluded_state: Dict = {
            'active': False, 'start_time': 0.0, 'last_update_time': 0.0,
            'max_occlusion': 0.0,
        }
        # multiple_faces reuses the generic active_violations dict
        self.active_violations: Dict[str, Dict] = {
            'multiple_faces': {
                'active': False, 'start_time': 0.0, 'last_update_time': 0.0,
                'duration': 0.0, 'max_intensity': 0.0, 'confidence': 0.9, 'velocity': 0.0,
            }
        }

    # ------------------------------------------------------------------
    # Velocity helpers
    # BUG-2 FIX: label covers the full range with no empty-string return
    # ------------------------------------------------------------------

    def _calculate_velocity(self, angle_history: deque, ts_history: deque) -> float:
        """Return average angular velocity (°/s) over recent frames."""
        if len(angle_history) < 2 or len(ts_history) < 2:
            return 0.0
        try:
            velocities = []
            n = min(len(angle_history), 10)
            angles = list(angle_history)
            times  = list(ts_history)
            for i in range(1, n):
                da = abs(angles[-i] - angles[-i - 1])
                if da > 180:
                    da = 360 - da
                dt = times[-i] - times[-i - 1]
                if dt > 0:
                    velocities.append(da / dt)
            return sum(velocities) / len(velocities) if velocities else 0.0
        except (IndexError, ZeroDivisionError):
            return 0.0

    def _get_velocity_label(self, velocity: float) -> str:
        """
        Convert velocity (°/s) to human-readable label.

        BUG-2 FIX: Previously returned "" for velocity < MIN_VELOCITY_THRESHOLD,
        leaving events without any label.  Now always returns a non-empty string.
        """
        if velocity < self.config.MIN_VELOCITY_THRESHOLD:
            return "negligible"           # was "" — now always labelled
        elif velocity < self.config.SUSPICIOUS_VELOCITY_THRESHOLD:
            return "slow"
        elif velocity < self.config.HIGH_RISK_VELOCITY_THRESHOLD:
            return "moderate"
        else:
            return "rapid"

    # ------------------------------------------------------------------
    # Context-aware: safe downwards look (don't flag writing/thinking)
    # ------------------------------------------------------------------

    def _is_safe_down_look(
        self,
        direction: str,
        duration_sec: float,
        velocity_deg_per_s: float,
        max_pitch_deg: float,
    ) -> bool:
        """
        True if this "down" movement should be ignored (writing/thinking).
        All three conditions must hold: slow, short, small angle.
        """
        if direction != 'down':
            return False
        max_vel = getattr(
            self.config, 'SAFE_DOWN_MAX_VELOCITY_DEG_PER_S', 10.0
        )
        max_dur = getattr(self.config, 'SAFE_DOWN_MAX_DURATION_SEC', 5.0)
        max_pitch = getattr(self.config, 'SAFE_DOWN_MAX_PITCH_DEG', 25.0)
        return (
            velocity_deg_per_s < max_vel
            and duration_sec < max_dur
            and max_pitch_deg < max_pitch
        )

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    def _get_yaw_direction(self, yaw: Optional[float], thresholds: Dict) -> Tuple[str, float]:
        """Return (direction, intensity) for yaw, or ('', 0) if within threshold."""
        if yaw is None or abs(yaw) <= thresholds['yaw']:
            return '', 0.0
        return ('right' if yaw > 0 else 'left'), abs(yaw)

    def _get_pitch_direction(self, pitch: Optional[float], thresholds: Dict) -> Tuple[str, float]:
        """
        Return (direction, intensity) for pitch, or ('', 0) if within threshold.

        BUG-6 FIX: Direction was inverted. With solvePnP + BUG-1 normalisation,
        positive pitch means the face is tilted upward (nose tip up, chin down),
        so pitch > 0 → 'up', pitch < 0 → 'down'.  The previous mapping was the
        opposite and caused all head_up events to be labelled 'down' and vice versa.
        """
        if pitch is None or abs(pitch) <= thresholds['pitch']:
            return '', 0.0
        return ('up' if pitch > 0 else 'down'), abs(pitch)

    def _get_eye_direction(
        self, gaze_h: Optional[float], gaze_v: Optional[float], thresholds: Dict
    ) -> Tuple[str, float]:
        """Return dominant eye-gaze direction and intensity (legacy single-axis)."""
        h_ok = gaze_h is not None and abs(gaze_h) > thresholds['eye_horizontal']
        v_ok = gaze_v is not None and abs(gaze_v) > thresholds['eye_vertical']

        if not h_ok and not v_ok:
            return '', 0.0

        h_ratio = abs(gaze_h) / thresholds['eye_horizontal'] if (gaze_h and thresholds['eye_horizontal']) else 0.0
        v_ratio = abs(gaze_v) / thresholds['eye_vertical']   if (gaze_v and thresholds['eye_vertical'])   else 0.0

        if h_ok and v_ok:
            if h_ratio >= v_ratio:
                return ('right' if gaze_h > 0 else 'left'), abs(gaze_h)
            else:
                return ('down' if gaze_v > 0 else 'up'), abs(gaze_v)
        if h_ok:
            return ('right' if gaze_h > 0 else 'left'), abs(gaze_h)
        return ('down' if gaze_v > 0 else 'up'), abs(gaze_v)

    def _get_eye_h_direction(self, gaze_h: Optional[float], thresholds: Dict) -> Tuple[str, float]:
        """Return horizontal eye-gaze direction and intensity (left/right only)."""
        if gaze_h is None or abs(gaze_h) <= thresholds['eye_horizontal']:
            return '', 0.0
        return ('right' if gaze_h > 0 else 'left'), abs(gaze_h)

    def _get_eye_v_direction(self, gaze_v: Optional[float], thresholds: Dict) -> Tuple[str, float]:
        """Return vertical eye-gaze direction and intensity (up/down only)."""
        if gaze_v is None or abs(gaze_v) <= thresholds['eye_vertical']:
            return '', 0.0
        return ('down' if gaze_v > 0 else 'up'), abs(gaze_v)

    # ------------------------------------------------------------------
    # Event recording helpers
    # BUG-4 FIX: removed `if start_time_rounded == last_recorded_* : return`
    # ------------------------------------------------------------------

    def _record_head_event(
        self, start_time: float, end_time: float,
        direction: str, max_intensity: float,
        confidence: float, velocity: float,
        tvt_prediction: Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        """
        Record a head-movement event.

        BUG-4 FIX: The previous guard
            ``if start_time_rounded == self.last_recorded_head_timestamp: return``
        was removed.  It silently discarded legitimate consecutive events that
        happened to round to the same second.  The state machine already
        prevents true duplicates via gap_tolerance and min_duration.

        BUG-7 FIX (duration underreporting): The event state machine only closes
        when a gap > gap_tolerance is observed.  This means end_time
        (= last_update_time) is the last *detected* frame, but the person's head
        was still turned until approximately end_time + gap_tolerance.  We add
        half the gap_tolerance as a conservative correction so reported durations
        match the actual behavioral window rather than the last sampled frame.

        Context-aware: "down" events that are slow, short, and small angle are
        not recorded (safe downwards look). When tvt_prediction is provided,
        only record if TVT agrees (matching class and prob >= threshold).
        """
        # BUG-7 FIX: compensate for the detection-gap at the tail of the event
        gap_compensation = self.gap_tolerance * 0.5
        corrected_end    = end_time + gap_compensation
        duration    = corrected_end - start_time
        min_dur     = getattr(self.config, 'MIN_EVENT_DURATION', 0.5)
        min_conf    = getattr(self.config, 'MIN_CONFIDENCE_THRESHOLD', 0.45)

        if duration < min_dur or not direction or confidence < min_conf:
            return

        # Context-aware: safe downwards look — don't flag writing/thinking
        if self._is_safe_down_look(direction, duration, velocity, max_intensity):
            return

        # TVT gating: when TVT is enabled and confident, only record if TVT agrees
        if tvt_prediction is not None:
            prob = float(tvt_prediction.get('probability', 0.0))
            # Only gate when TVT is confident (e.g. > 0.5); else keep threshold-based behavior
            if prob > 0.5:
                head_to_tvt = {
                    'left': 'left_cheating_glance',
                    'right': 'right_cheating_glance',
                    'down': 'phone_lookdown',
                }
                expected = head_to_tvt.get(direction)
                if expected is not None:
                    tvt_class = tvt_prediction.get('behavior_class', '')
                    if prob < tvt_prob_threshold or tvt_class != expected:
                        return

        vtype = f'head_{direction}'
        ts    = round(start_time, 2)
        self.counts[vtype]     = self.counts.get(vtype, 0) + 1
        self.timestamps[vtype].append(ts)

        self.violation_events.append({
            'type':       vtype,
            'timestamp':  ts,
            'duration':   round(duration, 1),
            'intensity':  max_intensity,
            'confidence': round(confidence, 2),
            'velocity':   velocity,
        })

        if max_intensity > self.max_intensities.get(vtype, 0.0):
            self.max_intensities[vtype] = max_intensity

    def _record_eye_event(
        self, start_time: float, end_time: float,
        direction: str, max_intensity: float,
        confidence: float, velocity: float,
        tvt_prediction: Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        """
        Record an eye-gaze event.

        BUG-4 FIX: duplicate-timestamp guard removed (same rationale as
        _record_head_event). When tvt_prediction is provided, only record if
        TVT agrees (matching class and prob >= threshold).

        BUG-7 FIX: Same gap_tolerance compensation as _record_head_event.
        """
        # BUG-7 FIX: compensate for detection-gap at the tail of the event
        gap_compensation = self.gap_tolerance * 0.5
        corrected_end    = end_time + gap_compensation
        duration = corrected_end - start_time
        min_dur  = getattr(self.config, 'MIN_EVENT_DURATION', 0.5)
        min_conf = getattr(self.config, 'MIN_CONFIDENCE_THRESHOLD', 0.45)

        if duration < min_dur or not direction or confidence < min_conf:
            return

        # TVT gating (only when TVT is confident)
        if tvt_prediction is not None:
            prob = float(tvt_prediction.get('probability', 0.0))
            if prob > 0.5:
                gaze_to_tvt = {
                    'left': 'left_cheating_glance',
                    'right': 'right_cheating_glance',
                    'down': 'phone_lookdown',
                }
                expected = gaze_to_tvt.get(direction)
                if expected is not None:
                    tvt_class = tvt_prediction.get('behavior_class', '')
                    if prob < tvt_prob_threshold or tvt_class != expected:
                        return

        vtype = f'gaze_{direction}'
        ts    = round(start_time, 2)
        self.counts[vtype]     = self.counts.get(vtype, 0) + 1
        self.timestamps[vtype].append(ts)

        self.violation_events.append({
            'type':       vtype,
            'timestamp':  ts,
            'duration':   round(duration, 1),
            'intensity':  max_intensity,
            'confidence': round(confidence, 2),
            'velocity':   velocity,
        })

        if max_intensity > self.max_intensities.get(vtype, 0.0):
            self.max_intensities[vtype] = max_intensity

    def _record_face_missing_event(self, start_time: float, end_time: float) -> None:
        """Record a face-missing event (BUG-4 FIX: guard removed). Uses FACE_MISSING_MIN_DURATION when set."""
        duration = end_time - start_time
        min_dur  = getattr(self.config, 'FACE_MISSING_MIN_DURATION', None)
        if min_dur is None:
            min_dur = getattr(self.config, 'MIN_EVENT_DURATION', 0.5)
        if duration < min_dur:
            return

        ts = round(start_time, 2)
        self.counts['face_missing'] += 1
        self.timestamps['face_missing'].append(ts)
        self.violation_events.append({
            'type':       'face_missing',
            'timestamp':  ts,
            'duration':   round(duration, 1),
            'intensity':  0.0,
            'confidence': 1.0,
            'velocity':   0.0,
        })

    def _record_face_occluded_event(
        self, start_time: float, end_time: float, max_occlusion: float
    ) -> None:
        """Record a face-occlusion event."""
        duration = end_time - start_time
        min_dur  = getattr(self.config, 'MIN_EVENT_DURATION', 0.5)
        if duration < min_dur:
            return

        ts = round(start_time, 2)
        self.counts['face_occluded'] += 1
        self.timestamps['face_occluded'].append(ts)
        self.violation_events.append({
            'type':       'face_occluded',
            'timestamp':  ts,
            'duration':   round(duration, 1),
            'intensity':  round(max_occlusion * 100, 1),
            'confidence': 0.85,
            'velocity':   0.0,
        })

    # ------------------------------------------------------------------
    # Main per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: float,
        gaze_h: Optional[float],
        gaze_v: Optional[float],
        yaw: Optional[float],
        pitch: Optional[float],
        num_faces: int,
        thresholds: Dict[str, float],
        gaze_confidence: float = 0.0,
        head_confidence: float = 0.0,
        occlusion_ratio: float = 0.0,
        yaw_history:       Optional[deque] = None,
        pitch_history:     Optional[deque] = None,
        eye_angle_history: Optional[deque] = None,
        timestamp_history: Optional[deque] = None,
        tvt_prediction:    Optional[Dict] = None,
    ) -> None:
        """
        Update all violation trackers for a single video frame.

        Head movement is now tracked as TWO independent state machines —
        one for yaw (left/right) and one for pitch (up/down) — so that
        simultaneous yaw + pitch deviations produce two recorded events
        instead of one arbitrarily chosen dominant direction.

        When tvt_prediction is provided (TVT enabled), head/gaze events are
        only recorded if TVT agrees (matching behavior_class and probability
        >= TVT_PROB_THRESHOLD).
        """
        tvt_threshold = getattr(self.config, 'TVT_PROB_THRESHOLD', 0.85)
        # Compute velocities
        head_velocity = 0.0
        if yaw_history and pitch_history and timestamp_history:
            yv = self._calculate_velocity(yaw_history,   timestamp_history)
            pv = self._calculate_velocity(pitch_history, timestamp_history)
            head_velocity = max(yv, pv)

        eye_velocity = 0.0
        if eye_angle_history and timestamp_history:
            eye_velocity = self._calculate_velocity(eye_angle_history, timestamp_history)

        # ── HEAD YAW STATE MACHINE ─────────────────────────────────────────
        yaw_dir, yaw_int = self._get_yaw_direction(yaw, thresholds)
        self._update_head_axis_state(
            self.current_yaw_state, timestamp,
            yaw_dir, yaw_int, head_confidence, head_velocity,
            tvt_prediction, tvt_threshold,
            is_yaw=True,   # BUG-8: track last yaw-active time for face_missing bridge
        )

        # ── HEAD PITCH STATE MACHINE ───────────────────────────────────────
        pitch_dir, pitch_int = self._get_pitch_direction(pitch, thresholds)
        self._update_head_axis_state(
            self.current_pitch_state, timestamp,
            pitch_dir, pitch_int, head_confidence, head_velocity,
            tvt_prediction, tvt_threshold,
        )

        # ── EYE GAZE: horizontal (left/right) and vertical (up/down) tracked separately ──
        eye_h_dir, eye_h_int = self._get_eye_h_direction(gaze_h, thresholds)
        eye_v_dir, eye_v_int = self._get_eye_v_direction(gaze_v, thresholds)
        self._update_eye_axis_state(
            self.current_eye_h_state, timestamp,
            eye_h_dir, eye_h_int, gaze_confidence, eye_velocity,
            tvt_prediction, tvt_threshold,
        )
        self._update_eye_axis_state(
            self.current_eye_v_state, timestamp,
            eye_v_dir, eye_v_int, gaze_confidence, eye_velocity,
            tvt_prediction, tvt_threshold,
        )

        # ── FACE MISSING STATE MACHINE ─────────────────────────────────────
        # Only start counting after FACE_MISSING_MIN_START_DURATION consecutive no-face (reduces detector flicker FP)
        # BUG-8 FIX: Also suppress face_missing when a yaw event was active very
        # recently — extreme head turns cause MediaPipe to lose the face, which
        # should extend the head_movement event, not create a face_missing event.
        # HEAD_TURN_BRIDGE_SECONDS (default 1.0s) is the suppression window.
        head_bridge = getattr(self.config, 'HEAD_TURN_BRIDGE_SECONDS', 1.0)
        within_head_bridge = (timestamp - self._last_active_yaw_time) <= head_bridge

        min_start = getattr(self.config, 'FACE_MISSING_MIN_START_DURATION', 0.0)
        if num_faces == 0:
            state = self.current_face_missing_state
            if within_head_bridge:
                # Face lost during an extreme head turn — extend yaw event instead of logging face_missing
                if self.current_yaw_state['active']:
                    self.current_yaw_state['last_update_time'] = timestamp
                # Reset any pending face_missing accumulation so it can't open mid-bridge
                state['first_no_face_time'] = None
            elif not state['active']:
                first_no_face = state.get('first_no_face_time')
                if first_no_face is None:
                    state['first_no_face_time'] = timestamp
                elif min_start <= 0 or (timestamp - first_no_face) >= min_start:
                    state['active'] = True
                    state['start_time'] = first_no_face
                    state['last_update_time'] = timestamp
                    state['first_no_face_time'] = None
            else:
                self.current_face_missing_state['last_update_time'] = timestamp
        else:
            state = self.current_face_missing_state
            state['first_no_face_time'] = None
            if state['active']:
                gap = timestamp - state['last_update_time']
                if gap > self.gap_tolerance:
                    self._record_face_missing_event(
                        state['start_time'],
                        state['last_update_time'],
                    )
                    state['active'] = False

        # ── FACE OCCLUSION STATE MACHINE ───────────────────────────────────
        occ_threshold = getattr(self.config, 'FACE_OCCLUSION_THRESHOLD', 0.30)
        if num_faces > 0 and occlusion_ratio > occ_threshold:
            if not self.current_face_occluded_state['active']:
                self.current_face_occluded_state = {
                    'active': True, 'start_time': timestamp,
                    'last_update_time': timestamp, 'max_occlusion': occlusion_ratio,
                }
            else:
                s = self.current_face_occluded_state
                s['last_update_time'] = timestamp
                s['max_occlusion']    = max(s['max_occlusion'], occlusion_ratio)
        else:
            if self.current_face_occluded_state['active']:
                gap = timestamp - self.current_face_occluded_state['last_update_time']
                if gap > self.gap_tolerance:
                    self._record_face_occluded_event(
                        self.current_face_occluded_state['start_time'],
                        self.current_face_occluded_state['last_update_time'],
                        self.current_face_occluded_state['max_occlusion'],
                    )
                    self.current_face_occluded_state['active'] = False

        # ── MULTIPLE FACES ─────────────────────────────────────────────────
        self._check_multiple_faces(timestamp, num_faces)

    def _update_head_axis_state(
        self, state: Dict, timestamp: float,
        direction: str, intensity: float,
        confidence: float, velocity: float,
        tvt_prediction: Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
        is_yaw: bool = False,
    ) -> None:
        """
        Generic state-machine update for a single head axis (yaw OR pitch).
        Called independently for each axis so both get recorded.

        BUG-8 FIX: When is_yaw=True, record the timestamp so the face_missing
        bridge can tell whether face loss was caused by an extreme head turn.
        """
        if direction:
            if is_yaw:
                self._last_active_yaw_time = timestamp   # BUG-8: track yaw activity
            if not state['active']:
                state.update({
                    'active': True, 'start_time': timestamp,
                    'last_update_time': timestamp, 'direction': direction,
                    'max_intensity': intensity, 'confidence': confidence,
                    'velocity': velocity,
                })
            else:
                if direction != state['direction']:
                    # Direction reversed — record previous, start new
                    self._record_head_event(
                        state['start_time'], state['last_update_time'],
                        state['direction'], state['max_intensity'],
                        state['confidence'], state['velocity'],
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state.update({
                        'active': True, 'start_time': timestamp,
                        'last_update_time': timestamp, 'direction': direction,
                        'max_intensity': intensity, 'confidence': confidence,
                        'velocity': velocity,
                    })
                else:
                    state['max_intensity']    = max(state['max_intensity'], intensity)
                    state['last_update_time'] = timestamp
                    state['confidence']       = max(state['confidence'], confidence)
                    state['velocity']         = max(state['velocity'], velocity)
        else:
            if state['active']:
                gap = timestamp - state['last_update_time']
                if gap > self.gap_tolerance:
                    self._record_head_event(
                        state['start_time'], state['last_update_time'],
                        state['direction'], state['max_intensity'],
                        state['confidence'], state['velocity'],
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state['active'] = False

    def _update_eye_axis_state(
        self, state: Dict, timestamp: float,
        direction: str, intensity: float,
        confidence: float, velocity: float,
        tvt_prediction: Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        """
        Generic state-machine update for a single eye axis (horizontal OR vertical).
        Called independently for each axis so both gaze_left/right and gaze_up/down are recorded.
        """
        if direction:
            if not state['active']:
                state.update({
                    'active': True, 'start_time': timestamp,
                    'last_update_time': timestamp, 'direction': direction,
                    'max_intensity': intensity, 'confidence': confidence,
                    'velocity': velocity,
                })
            else:
                if direction != state['direction']:
                    self._record_eye_event(
                        state['start_time'], state['last_update_time'],
                        state['direction'], state['max_intensity'],
                        state['confidence'], state['velocity'],
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state.update({
                        'active': True, 'start_time': timestamp,
                        'last_update_time': timestamp, 'direction': direction,
                        'max_intensity': intensity, 'confidence': confidence,
                        'velocity': velocity,
                    })
                else:
                    state['max_intensity']    = max(state['max_intensity'], intensity)
                    state['last_update_time'] = timestamp
                    state['confidence']       = max(state['confidence'], confidence)
                    state['velocity']         = max(state['velocity'], velocity)
        else:
            if state['active']:
                gap = timestamp - state['last_update_time']
                if gap > self.gap_tolerance:
                    self._record_eye_event(
                        state['start_time'], state['last_update_time'],
                        state['direction'], state['max_intensity'],
                        state['confidence'], state['velocity'],
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state['active'] = False

    def _check_multiple_faces(self, timestamp: float, num_faces: int) -> None:
        """Handle multiple-faces detection state machine."""
        state    = self.active_violations['multiple_faces']
        is_active = bool(state['active'])

        if num_faces > 1:
            if not is_active:
                state.update({
                    'active': True, 'start_time': timestamp,
                    'last_update_time': timestamp, 'duration': 0.0,
                    'max_intensity': float(num_faces), 'confidence': 0.9, 'velocity': 0.0,
                })
            else:
                state['last_update_time'] = timestamp
                state['duration']         = timestamp - float(state['start_time'])
                state['max_intensity']    = max(float(state['max_intensity']), float(num_faces))
        else:
            if is_active:
                duration = float(state['last_update_time']) - float(state['start_time'])
                min_dur  = getattr(self.config, 'MIN_MULTIPLE_FACE_DURATION', 0.3)
                if duration >= min_dur:
                    ts = round(float(state['start_time']), 2)
                    self.counts['multiple_faces'] += 1
                    self.timestamps['multiple_faces'].append(ts)
                    self.violation_events.append({
                        'type':       'multiple_faces',
                        'timestamp':  ts,
                        'duration':   round(duration, 1),
                        'intensity':  0.0,
                        'confidence': float(state['confidence']),
                        'velocity':   0.0,
                    })
                state['active'] = False

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        return self.counts.copy()

    def get_timestamps(self, violation_type: str) -> List[float]:
        return self.timestamps.get(violation_type, [])

    def get_all_events(self) -> List[Dict]:
        """Return all violation events sorted ascending by timestamp."""
        return sorted(self.violation_events, key=lambda x: x['timestamp'])

    # ------------------------------------------------------------------
    # Finalize (called after all frames are processed)
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Flush any still-active violation states at end of video."""
        # Head yaw
        if self.current_yaw_state['active']:
            self._record_head_event(
                self.current_yaw_state['start_time'],
                self.current_yaw_state['last_update_time'],
                self.current_yaw_state['direction'],
                self.current_yaw_state['max_intensity'],
                self.current_yaw_state['confidence'],
                self.current_yaw_state['velocity'],
            )
            self.current_yaw_state['active'] = False

        # Head pitch
        if self.current_pitch_state['active']:
            self._record_head_event(
                self.current_pitch_state['start_time'],
                self.current_pitch_state['last_update_time'],
                self.current_pitch_state['direction'],
                self.current_pitch_state['max_intensity'],
                self.current_pitch_state['confidence'],
                self.current_pitch_state['velocity'],
            )
            self.current_pitch_state['active'] = False

        # Eye gaze (horizontal and vertical)
        if self.current_eye_h_state['active']:
            self._record_eye_event(
                self.current_eye_h_state['start_time'],
                self.current_eye_h_state['last_update_time'],
                self.current_eye_h_state['direction'],
                self.current_eye_h_state['max_intensity'],
                self.current_eye_h_state['confidence'],
                self.current_eye_h_state['velocity'],
            )
            self.current_eye_h_state['active'] = False
        if self.current_eye_v_state['active']:
            self._record_eye_event(
                self.current_eye_v_state['start_time'],
                self.current_eye_v_state['last_update_time'],
                self.current_eye_v_state['direction'],
                self.current_eye_v_state['max_intensity'],
                self.current_eye_v_state['confidence'],
                self.current_eye_v_state['velocity'],
            )
            self.current_eye_v_state['active'] = False

        # Face missing
        if self.current_face_missing_state['active']:
            self._record_face_missing_event(
                self.current_face_missing_state['start_time'],
                self.current_face_missing_state['last_update_time'],
            )
            self.current_face_missing_state['active'] = False
        self.current_face_missing_state['first_no_face_time'] = None

        # Face occluded
        if self.current_face_occluded_state['active']:
            self._record_face_occluded_event(
                self.current_face_occluded_state['start_time'],
                self.current_face_occluded_state['last_update_time'],
                self.current_face_occluded_state['max_occlusion'],
            )
            self.current_face_occluded_state['active'] = False

        # Multiple faces
        mf = self.active_violations['multiple_faces']
        if mf['active']:
            duration = float(mf['last_update_time']) - float(mf['start_time'])
            min_dur  = getattr(self.config, 'MIN_MULTIPLE_FACE_DURATION', 0.3)
            if duration >= min_dur:
                ts = round(float(mf['start_time']), 2)
                self.counts['multiple_faces'] += 1
                self.timestamps['multiple_faces'].append(ts)
                self.violation_events.append({
                    'type':       'multiple_faces',
                    'timestamp':  ts,
                    'duration':   round(duration, 1),
                    'intensity':  0.0,
                    'confidence': float(mf['confidence']),
                    'velocity':   0.0,
                })
            mf['active'] = False