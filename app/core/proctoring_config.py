"""
Configuration for AI Proctoring System
Based on Industry Standards: ProctorU, Mettl, Talview, Examity, HireVue, iMocha
Version: 2.3 - Performance Optimized with Enhanced Detection
"""
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProctoringConfig:
    # Video Processing Settings - CPU Optimized
    MAX_FRAME_DIMENSION = 960  # Reduced from 1280 for better CPU performance
    TARGET_FPS = 15  # Increased from 10 to capture more eye movement events

    # Eye gaze: Separate thresholds for horizontal and vertical movements
    # Horizontal eye movements (left/right) are typically larger than vertical
    FIXED_EYE_HORIZONTAL_THRESHOLD = 8.0   # degrees (Industry: 8-12° for left/right)
    FIXED_EYE_VERTICAL_THRESHOLD = 6.0     # degrees (Industry: 6-9° for up/down)

    # Head movements: Balanced to avoid false positives
    FIXED_HEAD_YAW_THRESHOLD = 30.0        # degrees (Industry: 25-35°, ProctorU uses ~30°)
    FIXED_HEAD_PITCH_THRESHOLD = 20.0      # degrees (Industry: 18-25°, most use ~20°)
    FIXED_HEAD_ROLL_THRESHOLD = 30.0       # degrees (Industry: 25-35°, most use ~30°)

    EYE_ASPECT_RATIO_THRESHOLD = 0.12      # Below this indicates closed/blinking eyes (lowered from 0.15)
    MIN_IRIS_VISIBILITY = 0.6              # Minimum iris visibility for valid gaze reading (lowered from 0.65)

    # Frame Rejection Thresholds
    MAX_HEAD_ROTATION_SPEED = 10.0
    MAX_BBOX_CENTER_SHIFT = 0.05
    MAX_EYE_ANGLE_VARIANCE = 0.35
    MIN_FACE_PRESENCE_SCORE = 0.85

    # MediaPipe Settings
    MEDIAPIPE_MAX_FACES = 2
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

    # ==================================================================================
    # CALIBRATION & SCORING SETTINGS - CURRENTLY NOT IN USE
    # ==================================================================================
    # The following settings are preserved for future implementation of adaptive
    # calibration and risk scoring features. Currently, the system uses fixed
    # thresholds defined above for immediate detection.
    # ==================================================================================

    # # Calibration Settings (Industry Standard)
    # CALIBRATION_DURATION_SEC = 8.0  # 6-10 seconds calibration window
    # CALIBRATION_MIN_FRAMES = 80
    #
    # # Baseline Threshold Calculation (Industry Standard)
    # # HEAD_THRESHOLD = yaw_variance * 2.5 (Used by Talview & ProctorU)
    # # EYE_THRESHOLD = eye_variance * 2.2
    # VARIANCE_MULTIPLIER_HEAD = 2.5
    # VARIANCE_MULTIPLIER_EYE = 2.2
    #
    # # Rule 1: Clamp Calibration Thresholds
    # MIN_HEAD_THRESHOLD = 8.0   # degrees
    # MIN_EYE_THRESHOLD = 8.0    # degrees
    #
    # # Rule 2: Trim Outlier Frames
    # OUTLIER_TRIM_PERCENTAGE = 15  # Remove top 15% most unstable frames
    #
    # # Rule 3: Recenter Mean for Drifting Faces
    # MOVING_AVERAGE_WINDOW = 20  # last 20 frames
    #
    # # Rule 4: Freeze Calibration if Instability Continues
    # DEFAULT_YAW_VARIANCE = 5.0    # 4-6 degrees
    # DEFAULT_PITCH_VARIANCE = 5.0  # 4-6 degrees
    # DEFAULT_EYE_VARIANCE = 6.5    # 6-10 degrees
    #
    # # Baseline Hard Limits
    # MAX_BASELINE_YAW = 20.0
    # MAX_BASELINE_PITCH = 15.0
    # MAX_BASELINE_ROLL = 12.0
    # MAX_BASELINE_EYE = 0.7
    #
    # # A. Count Score (Frequency) - Event Weights
    # COUNT_WEIGHT_MILD_HEAD = 0.6      # Mild head movement (Low severity)
    # COUNT_WEIGHT_MAJOR_HEAD = 1.3     # Major head movement (Medium severity)
    # COUNT_WEIGHT_EYE_GAZE = 1.1       # Eye-gaze deviation (Medium severity)
    # COUNT_WEIGHT_FACE_MISSING = 4.0   # Face missing (High severity)
    # COUNT_WEIGHT_MULTIPLE_FACES = 6.0 # Multiple persons (Critical severity)
    #
    # # B. Duration Score - Category Weights
    # DURATION_WEIGHT_MILD = 0.2        # Minimal impact on score
    # DURATION_WEIGHT_SUSPICIOUS = 0.5  # Moderate impact on score
    # DURATION_WEIGHT_HIGH_RISK = 1.0   # Significant impact on score
    #
    # # C. Intensity Score - Severity Weights
    # INTENSITY_WEIGHT_SLIGHT = 0.3     # Within normal range
    # INTENSITY_WEIGHT_MODERATE = 0.7   # Noticeable deviation
    # INTENSITY_WEIGHT_EXTREME = 1.4    # Severe angle deviation
    #
    # # Risk Levels (0-100 Scale)
    # RISK_LOW_MAX = 30          # 0-30: No action required
    # RISK_MODERATE_MAX = 60     # 31-60: Monitor for patterns
    # RISK_SUSPICIOUS_MAX = 80   # 61-80: Review footage recommended
    # # 81-100: HIGH RISK - Manual review required

    @property
    def MYSQL_HOST(self):
        return os.getenv("MYSQL_HOST", "localhost")

    @property
    def MYSQL_PORT(self):
        port = os.getenv("MYSQL_PORT", "3306")
        return int(port) if port else 3306

    @property
    def MYSQL_USER(self):
        return os.getenv("MYSQL_USER", "root")

    @property
    def MYSQL_PASSWORD(self):
        return os.getenv("MYSQL_PASSWORD", "").strip()

    @property
    def MYSQL_DATABASE(self):
        return os.getenv("MYSQL_DATABASE", "proctoring")

    # Evidence Storage Settings
    STORE_EVIDENCE_FRAMES = True
    MAX_EVIDENCE_FRAMES_PER_ALERT = 3

    # Processing Settings
    ENABLE_ASYNC_PROCESSING = True
    MAX_VIDEO_SIZE_MB = 500
    ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.webm', '.mkv']

    @classmethod
    def get_default_thresholds(cls) -> Dict[str, float]:
        """
        Get fixed industry-standard thresholds for immediate detection
        No calibration needed - these values are used from frame 0

        Thresholds based on industry standards:
        - Eye gaze horizontal: 8 degrees (Detects left/right eye movements)
        - Eye gaze vertical: 6 degrees (More sensitive for up/down movements)
        - Head yaw: 30 degrees (Less sensitive - only flags significant head turns)
        - Head pitch: 20 degrees (Balanced - detects looking up/down significantly)
        - Head roll: 30 degrees (Less sensitive - only extreme tilts)
        """
        return {
            'eye_horizontal': cls.FIXED_EYE_HORIZONTAL_THRESHOLD,
            'eye_vertical': cls.FIXED_EYE_VERTICAL_THRESHOLD,
            'yaw': cls.FIXED_HEAD_YAW_THRESHOLD,
            'pitch': cls.FIXED_HEAD_PITCH_THRESHOLD,
            'roll': cls.FIXED_HEAD_ROLL_THRESHOLD
        }

    # # COMMENTED OUT - Not currently used in v2.1
    # @classmethod
    # def get_calibration_settings(cls) -> Dict[str, Any]:
    #     """Get calibration configuration - preserved for future use"""
    #     return {
    #         'duration_sec': cls.CALIBRATION_DURATION_SEC,
    #         'min_frames': cls.CALIBRATION_MIN_FRAMES,
    #         'variance_multiplier_head': cls.VARIANCE_MULTIPLIER_HEAD,
    #         'variance_multiplier_eye': cls.VARIANCE_MULTIPLIER_EYE,
    #         'min_head_threshold': cls.MIN_HEAD_THRESHOLD,
    #         'min_eye_threshold': cls.MIN_EYE_THRESHOLD,
    #         'outlier_trim_percentage': cls.OUTLIER_TRIM_PERCENTAGE,
    #         'moving_average_window': cls.MOVING_AVERAGE_WINDOW,
    #         'eye_aspect_ratio_threshold': cls.EYE_ASPECT_RATIO_THRESHOLD,
    #         'min_iris_visibility': cls.MIN_IRIS_VISIBILITY
    #     }
