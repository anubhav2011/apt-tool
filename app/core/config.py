"""
Configuration settings for the AI Interviewer application.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()

# ROOT_DIR = "/mnt/ai_question_generator"
ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class CORSConfig:
    ALLOW_ORIGINS: list = None
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list = None
    ALLOW_HEADERS: list = None

    def __post_init__(self):
        if self.ALLOW_ORIGINS is None:
            self.ALLOW_ORIGINS = ["*"]
            # self.ALLOW_ORIGINS = [
            #     "https://dev.foxmatrix.com",
            #     "https://qa.foxmatrix.com",
            #     "https://ai.foxmatrix.com",
            #     "https://www.foxmatrix.com",
            #     "https://foxmatrix.com",
            # ]
        if self.ALLOW_METHODS is None:
            self.ALLOW_METHODS = ["*"]
        if self.ALLOW_HEADERS is None:
            self.ALLOW_HEADERS = ["*"]


@dataclass
class RateLimitConfig:
    """Rate Limiting Configuration settings"""
    REQUESTS_PER_MINUTE: int = 62
    WINDOW_SIZE_SECONDS: int = 60
    ENABLED: bool = True


@dataclass
class PathConfig:
    """Path Configuration settings"""
    ROOT_DIR: str = ROOT_DIR
    LOG_DIR: str = "debug_logs"

    @property
    def debug_logs_dir(self) -> str:
        """Path template for debug logs"""
        return str(Path(self.ROOT_DIR) / self.LOG_DIR)

    @property
    def debug_logs_file(self) -> str:
        """Path template for debug logs"""
        return str(Path(self.ROOT_DIR) / self.LOG_DIR / "debug_logs_{}.log")


class ProctoringDatabaseConfig:
    """
    Database configuration for the AI Proctoring System.
    All values are driven by environment variables with safe fallbacks.
    """

    @property
    def HOST(self) -> str:
        return os.getenv("MYSQL_HOST", "localhost")

    @property
    def PORT(self) -> int:
        return int(os.getenv("MYSQL_PORT", "3306") or "3306")

    @property
    def USER(self) -> str:
        return os.getenv("MYSQL_USER", "root")

    @property
    def PASSWORD(self) -> str:
        return os.getenv("MYSQL_PASSWORD", "").strip()

    @property
    def DATABASE(self) -> str:
        return os.getenv("MYSQL_DATABASE", "proctoring")

    @property
    def connection_url(self) -> str:
        """Construct a MySQL connection URL from env variables."""
        return (
            f"mysql+pymysql://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DATABASE}"
        )

    def as_dict(self) -> dict:
        """Return database credentials as a plain dictionary."""
        return {
            "host":     self.HOST,
            "port":     self.PORT,
            "user":     self.USER,
            "password": self.PASSWORD,
            "database": self.DATABASE,
        }


@dataclass
class ProctoringConfig:
    """
    Configuration for AI Proctoring System.
    Based on Industry Standards: ProctorU, Mettl, Talview, Examity, HireVue, iMocha
    Version: 1.0 — Accuracy and efficiency balance
    """

    # ── Video Processing ──────────────────────────────────────────────────
    MAX_FRAME_DIMENSION: int   = 1280
    TARGET_FPS: int            = 18        # balance: better temporal resolution, still efficient (15–25 typical)
    WARMUP_SECONDS: float      = 3.0       # ignore violations in first N seconds (removes 0:00 false positives)

    # ── Gaze & Pose Thresholds (Calibration-Free) ─────────────────────────
    FIXED_EYE_HORIZONTAL_THRESHOLD: float = 6.0    # degrees; reduced false flags on wide monitors
    FIXED_EYE_VERTICAL_THRESHOLD: float   = 8.0    # higher: accommodate keyboard/scratch-paper glances
    FIXED_HEAD_YAW_THRESHOLD: float       = 42.0   # degrees; only clear left/right look-aways (was 35)
    FIXED_HEAD_PITCH_THRESHOLD: float     = 30.0   # degrees; clear looking down/up (was 25)
    FIXED_HEAD_ROLL_THRESHOLD: float      = 20.0   # "ear-to-shoulder" phone-listening posture

    # ── EAR & Blink Detection ─────────────────────────────────────────────
    EYE_ASPECT_RATIO_THRESHOLD: float = 0.18
    MIN_IRIS_VISIBILITY: float        = 0.65
    BLINK_EAR_THRESHOLD: float        = 0.14
    MIN_BLINK_DURATION: float         = 0.05   # seconds
    MAX_BLINK_DURATION: float         = 0.35   # seconds
    NORMAL_BLINK_RATE_MIN: int        = 5      # blinks/min
    NORMAL_BLINK_RATE_MAX: int        = 30     # blinks/min
    ABNORMAL_BLINK_WINDOW: float      = 60.0   # seconds

    # ── Temporal Consistency ──────────────────────────────────────────────
    MIN_EVENT_DURATION: float  = 0.6    # capture short genuine glances while filtering sub-second noise
    EVENT_GAP_TOLERANCE: float = 0.25   # tighter segment boundaries for distinct glances

    # ── Frame Rejection ───────────────────────────────────────────────────
    MAX_HEAD_ROTATION_SPEED: float = 10.0
    MAX_BBOX_CENTER_SHIFT: float   = 0.05
    MAX_EYE_ANGLE_VARIANCE: float  = 0.35
    MIN_FACE_PRESENCE_SCORE: float = 0.90
    MIN_CONFIDENCE_THRESHOLD: float = 0.55  # include more real events while filtering low-confidence noise

    # ── Movement Velocity (Cheat Pattern Detection) ────────────────────────
    MIN_VELOCITY_THRESHOLD: float        = 5.0     # deg/s — ignore micro-tremors
    SUSPICIOUS_VELOCITY_THRESHOLD: float = 25.0    # deg/s — fast glance
    HIGH_RISK_VELOCITY_THRESHOLD: float  = 50.0    # deg/s — whiplash / device-hiding
    VELOCITY_HISTORY_SIZE: int           = 10

    # ── Occlusion & Multi-Face ─────────────────────────────────────────────
    FACE_OCCLUSION_THRESHOLD: float   = 0.25   # 25% occlusion triggers detection
    MIN_MULTIPLE_FACE_DURATION: float = 0.2    # brief second-face appearances are logged

    # ── Face Missing (reduce detector-flicker false positives) ──────────────
    FACE_MISSING_MIN_START_DURATION: float = 0.5   # only start counting after 0.5s consecutive no-face
    FACE_MISSING_MIN_DURATION: float       = 1.0   # only record if face missing ≥ 1s

    # ── Head-turn Bridge (BUG-8 FIX) ─────────────────────────────────────────
    HEAD_TURN_BRIDGE_SECONDS: float = 1.0   # set to 0.0 to disable

    # ── MediaPipe Settings ────────────────────────────────────────────────
    MEDIAPIPE_MAX_FACES: int                    = 2
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float   = 0.7
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float    = 0.7

    # ── High-risk Report Filter ───────────────────────────────────────────
    REPORT_ONLY_HIGH_RISK: bool             = True
    MIN_HEAD_INTENSITY_HIGH_RISK: float     = 35.0   # degrees; include moderate head turns
    MIN_EYE_INTENSITY_HIGH_RISK: float      = 10.0   # degrees; include moderate gaze deviations
    MIN_HEAD_DURATION_HIGH_RISK: float      = 0.8    # seconds; head duration >= this + fast velocity = keep

    # ── Evidence Storage ──────────────────────────────────────────────────
    STORE_EVIDENCE_FRAMES: bool          = True
    MAX_EVIDENCE_FRAMES_PER_ALERT: int   = 3

    # ── Processing Limits ─────────────────────────────────────────────────
    ENABLE_ASYNC_PROCESSING: bool  = True
    MAX_VIDEO_SIZE_MB: int         = 500
    ALLOWED_VIDEO_FORMATS: list    = None

    def __post_init__(self):
        if self.ALLOWED_VIDEO_FORMATS is None:
            self.ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.webm', '.mkv']

    # ── Context-aware Proctoring (safe downwards look) ───────────────────
    SAFE_DOWN_MAX_VELOCITY_DEG_PER_S: float = 5.0    # very slow only = writing/thinking
    SAFE_DOWN_MAX_DURATION_SEC: float       = 2.5    # short = natural
    SAFE_DOWN_MAX_PITCH_DEG: float          = 18.0   # small angle only = normal

    # ── High-risk Filter: lower bar for head up/down (pitch) ──────────────
    MIN_HEAD_INTENSITY_HIGH_RISK_PITCH: float = 26.0  # head_up/head_down use this so moderate pitch is reported

    # ── TVT (Temporal Vision Transformer) ─────────────────────────────────
    ENABLE_TVT: bool               = False   # set True to use temporal landmark model
    TVT_TEMPORAL_WINDOW: int       = 24      # frames per window (16–32)
    TVT_CPU_OPTIMIZED: bool        = True
    TVT_PROB_THRESHOLD: float      = 0.85    # only record when TVT prob above this
    TVT_INFERENCE_INTERVAL_SEC: float = 1.0  # run TVT every N seconds

    # ── Parallel Video Processing ────────────────────────────────────────
    ENABLE_PARALLEL_PROCESSING: bool = False  # set True for 3-thread pipeline

    # ── Scoring Configuration ─────────────────────────────────────────────
    # Risk Level Thresholds (0-100 scale)
    RISK_LOW_MAX: int                  = 30      # Clean (0-30)
    RISK_MODERATE_MAX: int             = 60      # Moderate (31-60)
    RISK_SUSPICIOUS_MAX: int           = 80      # Suspicious (61-80), above 80 = HIGH_RISK

    # Score Component Weights
    WEIGHT_GAZE: float                 = 0.35    # Eye gaze risk weight
    WEIGHT_HEAD: float                 = 0.40    # Head movement risk weight
    WEIGHT_FACE: float                 = 0.25    # Face detection risk weight

    # Count Weights (violation frequency impact)
    COUNT_WEIGHT_EYE_GAZE: float       = 1.1     # Medium severity
    COUNT_WEIGHT_MILD_HEAD: float      = 0.6     # Low severity
    COUNT_WEIGHT_MAJOR_HEAD: float     = 1.3     # Medium severity
    COUNT_WEIGHT_FACE_MISSING: float   = 4.0     # High severity
    COUNT_WEIGHT_MULTIPLE_FACES: float = 6.0     # Critical severity

    # Duration Weights (seconds of violation impact)
    DURATION_WEIGHT_MILD: float        = 0.2     # Minimal impact
    DURATION_WEIGHT_SUSPICIOUS: float  = 0.5     # Moderate impact
    DURATION_WEIGHT_HIGH_RISK: float   = 1.0     # Significant impact

    # Intensity Weights (angle deviation severity)
    INTENSITY_WEIGHT_SLIGHT: float     = 0.3     # Within normal range
    INTENSITY_WEIGHT_MODERATE: float   = 0.7     # Noticeable deviation
    INTENSITY_WEIGHT_EXTREME: float    = 1.4     # Severe angle deviation

    # ── Threshold Accessors ───────────────────────────────────────────────

    def get_default_thresholds(self) -> Dict[str, float]:
        """
        Return fixed industry-standard thresholds.
        No calibration — same values applied to every candidate.
        """
        return {
            'eye_horizontal':     self.FIXED_EYE_HORIZONTAL_THRESHOLD,
            'eye_vertical':       self.FIXED_EYE_VERTICAL_THRESHOLD,
            'yaw':                self.FIXED_HEAD_YAW_THRESHOLD,
            'pitch':              self.FIXED_HEAD_PITCH_THRESHOLD,
            'roll':               self.FIXED_HEAD_ROLL_THRESHOLD,
            'min_event_duration': self.MIN_EVENT_DURATION,
        }


# ── Singleton Instances ───────────────────────────────────────────────────────

PATH_CONFIG             = PathConfig()
CORS_CONFIG             = CORSConfig()
RATE_LIMIT_CONFIG       = RateLimitConfig()
PROCTORING_CONFIG       = ProctoringConfig()
PROCTORING_DB_CONFIG    = ProctoringDatabaseConfig()
