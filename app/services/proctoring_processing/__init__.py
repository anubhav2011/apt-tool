"""
Proctoring Service Package
Contains all computation and business logic services
NO CALIBRATION - All calibration and scoring services are commented out
"""

# Import active services
from .video_processing_service import VideoProcessingService
from .detection_service import DetectionService

# from .calibration_service import CalibrationService
# from .scoring_service import ScoringService

# Export only active services
__all__ = [
    'VideoProcessingService',
    'DetectionService',
]

# Commented out exports for removed services
# __all__ = [
#     'VideoProcessingService',
#     'DetectionService',
#     'CalibrationService',
#     'ScoringService',
# ]
