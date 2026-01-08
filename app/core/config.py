"""
Configuration settings for the AI Interviewer application.
"""
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

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
    def DEBUG_LOGS_DIR(self) -> str:
        """Path template for debug logs"""
        return str(Path(self.ROOT_DIR) / self.LOG_DIR )

    @property
    def DEBUG_LOGS_FILE(self) -> str:
        """Path template for debug logs"""
        return str(Path(self.ROOT_DIR) / self.LOG_DIR / "debug_logs_{}.log")

PATH_CONFIG = PathConfig()
CORS_CONFIG = CORSConfig()
RATE_LIMIT_CONFIG = RateLimitConfig()
