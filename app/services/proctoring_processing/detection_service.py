"""
Detection Service
Handles gaze detection, head pose detection, and violation tracking
NO CALIBRATION - Uses fixed thresholds
"""
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, List, Union
import mediapipe as mp
from numpy.typing import NDArray

from .base_proctoring_processing_service import BaseService


class DetectionService(BaseService):
    """
    Service for detecting gaze, head pose, and tracking violations
    Enhanced with Kalman filtering and improved iris tracking
    """

    def _setup(self) -> None:
        """Initialize MediaPipe and detection models with enhanced settings"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.gaze_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lowered from 0.6
            min_tracking_confidence=0.5     # Lowered from 0.6
        )
        self.pose_face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,  # Lowered from 0.6
            min_tracking_confidence=0.5     # Lowered from 0.6
        )

        # Eye landmarks
        self.LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 373, 374, 380]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE_CORNERS = [33, 133]
        self.RIGHT_EYE_CORNERS = [362, 263]

        # Head pose landmarks
        self.landmark_indices = [1, 152, 33, 263, 61, 291]
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -3.3, -2.5),
            (-2.3, 1.65, -1.5), (2.3, 1.65, -1.5),
            (-1.5, -1.65, -1.5), (1.5, -1.65, -1.5)
        ], dtype=np.float64)

        self._init_kalman_filters()

        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size = 7  # Increased for better smoothing over more frames

        self.violation_tracker = ViolationTracker(self.config)

    def _init_kalman_filters(self) -> None:
        """Initialize Kalman filters for gaze smoothing"""
        self.kf_horizontal = cv2.KalmanFilter(2, 1)
        self.kf_horizontal.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf_horizontal.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf_horizontal.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kf_horizontal.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

        self.kf_vertical = cv2.KalmanFilter(2, 1)
        self.kf_vertical.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf_vertical.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf_vertical.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kf_vertical.measurementNoiseCov = np.array([[1]], np.float32) * 0.1

        self.kalman_initialized = False

    def _smooth_gaze_with_kalman(self, horizontal: float, vertical: float) -> Tuple[float, float]:
        """Apply Kalman filtering to smooth gaze measurements"""
        try:
            if not self.kalman_initialized:
                self.kf_horizontal.statePre = np.array([[horizontal], [0]], np.float32)
                self.kf_vertical.statePre = np.array([[vertical], [0]], np.float32)
                self.kalman_initialized = True
                return horizontal, vertical

            self.kf_horizontal.predict()
            self.kf_vertical.predict()

            h_measurement = np.array([[horizontal]], np.float32)
            v_measurement = np.array([[vertical]], np.float32)

            h_corrected = self.kf_horizontal.correct(h_measurement)
            v_corrected = self.kf_vertical.correct(v_measurement)

            return float(h_corrected[0][0]), float(v_corrected[0][0])
        except Exception:
            return horizontal, vertical

    def _calculate_eye_aspect_ratio(self, eye_coords: List[List[float]]) -> float:
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            eye_array = np.array(eye_coords, dtype=np.float64)
            v1 = np.linalg.norm(eye_array[1] - eye_array[5])
            v2 = np.linalg.norm(eye_array[2] - eye_array[4])
            h = np.linalg.norm(eye_array[0] - eye_array[3])

            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            return float(ear)
        except Exception:
            return 0.0

    def detect_gaze(self, frame: NDArray[np.uint8]) -> Tuple[Optional[float], Optional[float], int, Optional[Tuple[float, float]]]:
        """
        Detect gaze angle with enhanced iris tracking and Kalman filtering

        Args:
            frame: Input video frame

        Returns:
            Tuple of (horizontal_angle, vertical_angle, num_faces, bbox_center)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.gaze_face_mesh.process(rgb_frame)

        if results is None:
            return None, None, 0, None

        multi_face_landmarks = getattr(results, 'multi_face_landmarks', None)
        if multi_face_landmarks is None or len(multi_face_landmarks) == 0:
            return None, None, 0, None

        num_faces = len(multi_face_landmarks)
        face_landmarks = multi_face_landmarks[0].landmark

        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in face_landmarks]
        y_coords = [lm.y * h for lm in face_landmarks]
        bbox_center = (float(np.mean(x_coords)), float(np.mean(y_coords)))

        try:
            left_eye_coords: List[List[float]] = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.LEFT_EYE]
            right_eye_coords: List[List[float]] = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.RIGHT_EYE]
            left_iris_coords: List[List[float]] = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.LEFT_IRIS]
            right_iris_coords: List[List[float]] = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.RIGHT_IRIS]

            left_ear = self._calculate_eye_aspect_ratio(left_eye_coords)
            right_ear = self._calculate_eye_aspect_ratio(right_eye_coords)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.12:  # Lowered from 0.15 to 0.12
                return None, None, num_faces, bbox_center

            left_iris_valid = all(0 <= coord[0] <= w and 0 <= coord[1] <= h for coord in left_iris_coords if coord)
            right_iris_valid = all(0 <= coord[0] <= w and 0 <= coord[1] <= h for coord in right_iris_coords if coord)

            if not (left_iris_valid and right_iris_valid):
                return None, None, num_faces, bbox_center

            left_eye_center: NDArray[np.float64] = np.mean(np.array(left_eye_coords, dtype=np.float64), axis=0)
            right_eye_center: NDArray[np.float64] = np.mean(np.array(right_eye_coords, dtype=np.float64), axis=0)
            left_iris_center: NDArray[np.float64] = np.mean(np.array(left_iris_coords, dtype=np.float64), axis=0)
            right_iris_center: NDArray[np.float64] = np.mean(np.array(right_iris_coords, dtype=np.float64), axis=0)

            left_eye_width = float(np.linalg.norm(
                np.array([face_landmarks[self.LEFT_EYE_CORNERS[1]].x * w,
                         face_landmarks[self.LEFT_EYE_CORNERS[1]].y * h]) -
                np.array([face_landmarks[self.LEFT_EYE_CORNERS[0]].x * w,
                         face_landmarks[self.LEFT_EYE_CORNERS[0]].y * h])
            ))

            right_eye_width = float(np.linalg.norm(
                np.array([face_landmarks[self.RIGHT_EYE_CORNERS[1]].x * w,
                         face_landmarks[self.RIGHT_EYE_CORNERS[1]].y * h]) -
                np.array([face_landmarks[self.RIGHT_EYE_CORNERS[0]].x * w,
                         face_landmarks[self.RIGHT_EYE_CORNERS[0]].y * h])
            ))

            avg_eye_width = (left_eye_width + right_eye_width) / 2.0

            if avg_eye_width < 1.8:  # Lowered from 2.5
                return None, None, num_faces, bbox_center

            left_displacement = left_iris_center - left_eye_center
            right_displacement = right_iris_center - right_eye_center
            avg_displacement = (left_displacement + right_displacement) / 2.0

            horizontal_ratio = float(avg_displacement[0]) / avg_eye_width
            horizontal_ratio = np.clip(horizontal_ratio, -1.0, 1.0)
            horizontal_angle = np.arcsin(horizontal_ratio * 0.9) * (180.0 / np.pi)  # Increased from 0.8

            vertical_ratio = float(avg_displacement[1]) / avg_eye_width
            vertical_ratio = np.clip(vertical_ratio, -1.0, 1.0)
            vertical_angle = np.arcsin(vertical_ratio * 0.9) * (180.0 / np.pi)  # Increased from 0.8

            horizontal_angle, vertical_angle = self._smooth_gaze_with_kalman(
                horizontal_angle, vertical_angle
            )

            self.gaze_history.append((horizontal_angle, vertical_angle))
            if len(self.gaze_history) > self.max_history_size:
                self.gaze_history.pop(0)

            if len(self.gaze_history) >= 5:
                weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight to recent frames
                recent_h = [h for h, v in self.gaze_history[-5:]]
                recent_v = [v for h, v in self.gaze_history[-5:]]
                horizontal_angle = float(np.average(recent_h, weights=weights))
                vertical_angle = float(np.average(recent_v, weights=weights))

            return round(float(horizontal_angle), 2), round(float(vertical_angle), 2), num_faces, bbox_center

        except (IndexError, ValueError, TypeError):
            return None, None, num_faces, bbox_center

    def detect_head_pose(self, frame: NDArray[np.uint8]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Detect head pose angles

        Args:
            frame: Input video frame

        Returns:
            Tuple of (yaw, pitch, roll) angles
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_face_mesh.process(rgb_frame)

        if results is None:
            return None, None, None

        multi_face_landmarks = getattr(results, 'multi_face_landmarks', None)
        if multi_face_landmarks is None or len(multi_face_landmarks) == 0:
            return None, None, None

        face_landmarks = multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        landmark_coords: List[List[float]] = [[face_landmarks[i].x * w, face_landmarks[i].y * h] for i in self.landmark_indices]
        image_points = np.array(landmark_coords, dtype=np.float64)

        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vec, _ = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success or rotation_vec is None:
                return None, None, None

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            if rotation_mat is None:
                return None, None, None

            sy = float(np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2))
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])
                yaw = np.arctan2(-rotation_mat[2, 0], sy)
                roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0])
            else:
                pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1])
                yaw = np.arctan2(-rotation_mat[2, 0], sy)
                roll = 0.0

            return (
                round(float(np.degrees(yaw)), 2),
                round(float(np.degrees(pitch)), 2),
                round(float(np.degrees(roll)), 2)
            )

        except (cv2.error, ValueError, TypeError):
            return None, None, None

    def update_violations(self, timestamp: float, gaze_h: Optional[float], gaze_v: Optional[float],
                         yaw: Optional[float], pitch: Optional[float], roll: Optional[float],
                         num_faces: int, thresholds: Dict[str, float]) -> None:
        """
        Update violation tracking with current frame data
        Now includes vertical gaze detection

        Args:
            timestamp: Current video timestamp
            gaze_h: Horizontal gaze angle
            gaze_v: Vertical gaze angle
            yaw: Head yaw angle
            pitch: Head pitch angle
            roll: Head roll angle (reserved for future use)
            num_faces: Number of faces detected
            thresholds: Fixed thresholds
        """
        _ = roll
        self.violation_tracker.update(timestamp, gaze_h, gaze_v, yaw, pitch, num_faces, thresholds)

    def get_counts(self) -> Dict[str, int]:
        """Get violation counts"""
        return self.violation_tracker.get_counts()

    def get_all_timestamps(self) -> Dict[str, List[float]]:
        """Get all violation timestamps"""
        return {key: self.violation_tracker.get_timestamps(key) for key in self.violation_tracker.counts.keys()}

    def get_all_max_intensities(self) -> Dict[str, float]:
        """Get maximum intensities for all violations"""
        return self.violation_tracker.max_intensities.copy()

    def get_violation_events(self) -> List[Dict]:
        """Get all violation events with timestamps, durations, and intensities"""
        return self.violation_tracker.get_all_events()

    def cleanup(self) -> None:
        """Release MediaPipe resources and finalize active violations"""
        if hasattr(self, 'gaze_face_mesh'):
            self.gaze_face_mesh.close()
        if hasattr(self, 'pose_face_mesh'):
            self.pose_face_mesh.close()
        if hasattr(self, 'gaze_history'):
            self.gaze_history.clear()
        self.violation_tracker.finalize()


class ViolationTracker:
    """
    Duration-based violation tracker with event details
    """

    def __init__(self, config) -> None:
        self.config = config
        self.counts: Dict[str, int] = {
            'gaze_left': 0, 'gaze_right': 0, 'gaze_up': 0, 'gaze_down': 0,
            'head_left': 0, 'head_right': 0, 'head_up': 0, 'head_down': 0,
            'face_missing': 0, 'multiple_faces': 0
        }
        self.timestamps: Dict[str, List[float]] = {key: [] for key in self.counts.keys()}
        self.active_violations: Dict[str, Dict[str, Union[bool, float]]] = {
            key: {
                'active': False,
                'start_time': 0.0,
                'duration': 0.0,
                'max_intensity': 0.0,
            } for key in self.counts.keys()}
        self.max_intensities: Dict[str, float] = {key: 0.0 for key in self.counts.keys()}

        self.violation_events: List[Dict] = []

    def _start_violation(self, violation_type: str, timestamp: float, intensity: float = 0.0) -> None:
        """Start tracking a violation - captures the starting timestamp"""
        self.active_violations[violation_type] = {
            'active': True,
            'start_time': timestamp,  # This is the exact timestamp when threshold was first exceeded
            'duration': 0.0,
            'max_intensity': intensity,
        }

    def _update_violation(self, violation_type: str, timestamp: float, intensity: float = 0.0) -> None:
        """Update ongoing violation"""
        state = self.active_violations[violation_type]
        state['duration'] = timestamp - float(state['start_time'])
        state['max_intensity'] = max(float(state['max_intensity']), intensity)
        self.max_intensities[violation_type] = max(self.max_intensities[violation_type], float(state['max_intensity']))

    def _end_violation(self, violation_type: str) -> None:
        """End tracking for a violation and record if meets minimum duration"""
        state = self.active_violations[violation_type]

        if state['active']:
            duration = state['duration']
            if duration >= 0.15:  # Lowered from 0.25 seconds
                self.counts[violation_type] += 1
                self.timestamps[violation_type].append(round(float(state['start_time']), 2))

                self.violation_events.append({
                    'type': violation_type,
                    'timestamp': float(state['start_time']),  # Starting timestamp when event began
                    'duration': round(duration, 1),
                    'intensity': float(state['max_intensity'])
                })

        self.active_violations[violation_type]['active'] = False

    def _check_and_update_violation(self, violation_type: str, condition: bool,
                                    timestamp: float, intensity: float = 0.0) -> None:
        """Helper method to reduce code duplication for violation checking"""
        is_active = bool(self.active_violations[violation_type]['active'])

        if condition:
            if not is_active:
                self._start_violation(violation_type, timestamp, intensity)
            else:
                self._update_violation(violation_type, timestamp, intensity)
        else:
            if is_active:
                self._end_violation(violation_type)

    def update(self, timestamp: float, gaze_h: Optional[float], gaze_v: Optional[float],
               yaw: Optional[float], pitch: Optional[float], num_faces: int,
               thresholds: Dict[str, float]) -> None:
        """
        Update all violation states
        Now includes vertical gaze detection
        """
        self._check_and_update_violation(
            'gaze_left',
            gaze_h is not None and gaze_h < -thresholds['eye_horizontal'],
            timestamp,
            abs(gaze_h) if gaze_h is not None else 0.0
        )

        self._check_and_update_violation(
            'gaze_right',
            gaze_h is not None and gaze_h > thresholds['eye_horizontal'],
            timestamp,
            abs(gaze_h) if gaze_h is not None else 0.0
        )

        self._check_and_update_violation(
            'gaze_up',
            gaze_v is not None and gaze_v < -thresholds['eye_vertical'],
            timestamp,
            abs(gaze_v) if gaze_v is not None else 0.0
        )

        self._check_and_update_violation(
            'gaze_down',
            gaze_v is not None and gaze_v > thresholds['eye_vertical'],
            timestamp,
            abs(gaze_v) if gaze_v is not None else 0.0
        )

        self._check_and_update_violation(
            'head_left',
            yaw is not None and yaw < -thresholds['yaw'],
            timestamp,
            abs(yaw) if yaw is not None else 0.0
        )

        self._check_and_update_violation(
            'head_right',
            yaw is not None and yaw > thresholds['yaw'],
            timestamp,
            abs(yaw) if yaw is not None else 0.0
        )

        self._check_and_update_violation(
            'head_up',
            pitch is not None and pitch < -thresholds['pitch'],
            timestamp,
            abs(pitch) if pitch is not None else 0.0
        )

        self._check_and_update_violation(
            'head_down',
            pitch is not None and pitch > thresholds['pitch'],
            timestamp,
            abs(pitch) if pitch is not None else 0.0
        )

        self._check_and_update_violation('face_missing', num_faces == 0, timestamp)
        self._check_and_update_violation('multiple_faces', num_faces > 1, timestamp)

    def get_counts(self) -> Dict[str, int]:
        """Get violation counts"""
        return self.counts.copy()

    def get_timestamps(self, violation_type: str) -> List[float]:
        """Get timestamps for a violation type"""
        return self.timestamps.get(violation_type, [])

    def get_all_events(self) -> List[Dict]:
        """Get all violation events with details"""
        return self.violation_events.copy()

    def finalize(self) -> None:
        """Finalize all active violations at end of video"""
        for violation_type, state in self.active_violations.items():
            if state['active']:
                self._end_violation(violation_type)
