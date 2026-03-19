"""
Proctoring Service
Main service orchestrating all proctoring business logic
"""
from app.utils.logger import debug_logger
from typing import Dict, Optional, List
import uuid
import os
import tempfile
import json
import re

from app.core.config import ProctoringConfig
from app.core.exceptions import VideoProcessingError, DatabaseError, ReportNotFoundError, ValidationError
from app.repositories.base_repository import IRepository
from app.models.proctoring import ProctoringReport
from app.services.base_service import IProctoringService
from app.services.proctoring_processing import VideoProcessingService


class ProctoringService(IProctoringService):
    """
    Main proctoring service
    Orchestrates all business logic and coordinates between services
    """

    def __init__(self, repository: IRepository[ProctoringReport]):
        """
        Initialize proctoring service with injected dependencies

        Args:
            repository: Repository implementing IRepository base interface
        """
        self.repository = repository
        self.config = ProctoringConfig()
        debug_logger.info("Proctoring service initialized")

    async def process_video_upload(
        self,
        video_file,
        interview_id: Optional[str] = None
    ) -> Dict:
        """
        Process uploaded video file with validation and temp file handling

        Args:
            video_file: UploadFile object from FastAPI
            interview_id: Optional interview ID. If not provided, a new interview will be created.

        Returns:
            Processing result with formatted report

        Raises:
            ValidationError: If validation fails
            VideoProcessingError: If processing fails
            DatabaseError: If database operation fails
        """
        temp_path = None

        try:
            interview_id = interview_id or str(uuid.uuid4())
            debug_logger.info(f"Processing uploaded video for interview: {interview_id}")

            # Validate file format
            file_ext = os.path.splitext(video_file.filename)[1].lower()
            if file_ext not in self.config.ALLOWED_VIDEO_FORMATS:
                raise ValidationError(f"Unsupported video format: {file_ext}", 4001)

            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_path = temp_file.name
            temp_file.close()

            # Read and save video content
            content = await video_file.read()

            with open(temp_path, 'wb') as f:
                f.write(content)

            # Validate file size
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > self.config.MAX_VIDEO_SIZE_MB:
                raise ValidationError(
                    f"Video file too large: {file_size_mb:.2f}MB exceeds maximum {self.config.MAX_VIDEO_SIZE_MB}MB",
                    4002
                )

            debug_logger.info(f"Video file size: {file_size_mb:.2f} MB")

            # Process video
            result = self.process_video_file(temp_path, interview_id)

            # Format response
            formatted_result = self._format_json_response(result)

            return json.loads(formatted_result)

        except (ValidationError, VideoProcessingError, DatabaseError):
            raise
        except Exception as e:
            debug_logger.error(f"Unexpected error processing video upload: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to process video upload: {str(e)}", 5001)
        finally:
            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    debug_logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    debug_logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")

    async def process_video_from_url(
        self,
        video_url: str,
        interview_id: Optional[str] = None
    ) -> Dict:
        """
        Process video from URL with validation

        Args:
            video_url: URL to video file
            interview_id: Optional interview ID. If not provided, a new interview will be created.

        Returns:
            Processing result with formatted report

        Raises:
            ValidationError: If URL is invalid
            VideoProcessingError: If processing fails
            DatabaseError: If database operation fails
        """
        try:
            # Validate URL
            if not video_url or not video_url.strip():
                raise ValidationError("video_url is required and cannot be empty", 4003)

            interview_id = interview_id or str(uuid.uuid4())
            debug_logger.info(f"Processing video from URL for interview: {interview_id}")

            # Process video
            result = self.process_video_file(video_url, interview_id)

            # Format response
            formatted_result = self._format_json_response(result)

            return json.loads(formatted_result)

        except (ValidationError, VideoProcessingError, DatabaseError):
            raise
        except Exception as e:
            debug_logger.error(f"Unexpected error processing video from URL: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to process video from URL: {str(e)}", 5001)

    def process_video_file(
        self,
        video_path: str,
        interview_id: Optional[str] = None
    ) -> Dict:
        """
        Process video file and generate proctoring report with database persistence

        Args:
            video_path: Path or URL to video
            interview_id: Optional interview ID. If not provided, a new interview will be created.

        Returns:
            Processing result with report and database references

        Raises:
            VideoProcessingError: If video processing fails
            DatabaseError: If database operation fails
        """
        interview_id = interview_id or str(uuid.uuid4())
        debug_logger.info(f"Processing video for interview: {interview_id}")

        try:
            # Process video using video processing service
            processor = VideoProcessingService(self.config)
            report = processor.process_video(video_path, interview_id)
            processor.cleanup()

            gestures = report.get("analysis", {}).get("gestures", [])
            debug_logger.info(f"Extracted {len(gestures)} gesture types from report")

            # Count total occurrences
            total_occurrences = sum(len(g.get("occurrence", [])) for g in gestures)
            debug_logger.info(f"Total gesture occurrences: {total_occurrences}")

            # Get processing metadata
            processing_metadata = report.get("analysis", {}).get("processing_metadata", {})
            video_duration = processing_metadata.get("video_duration_sec", 0.0)
            fps = processing_metadata.get("fps", 12.0)

            # Save all data to database
            self._save_complete_report(interview_id, report, video_duration, fps)

            debug_logger.info(f"Video processed successfully for interview: {interview_id}")

            return {
                "interview_id": interview_id,
                "gestures": gestures
            }

        except DatabaseError:
            raise
        except Exception as e:
            debug_logger.error(f"Error processing video for interview {interview_id}: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Video processing failed: {str(e)}", 5001)

    def _format_json_response(self, data: dict) -> str:
        """
        Format JSON response with compact arrays for timestamps

        Args:
            data: Dictionary to format

        Returns:
            Formatted JSON string
        """
        if hasattr(data, 'dict'):
            data = data.dict()
        elif hasattr(data, 'model_dump'):
            data = data.model_dump()

        json_str = json.dumps(data, indent=2, ensure_ascii=False)

        def replace_timestamps(match):
            array_content = match.group(1)
            numbers = re.findall(r'[\d.]+', array_content)
            if numbers:
                return '"timestamps": [' + ', '.join(numbers) + ']'
            return match.group(0)

        json_str = re.sub(r'"timestamps":\s*\[(.*?)\]', replace_timestamps, json_str, flags=re.DOTALL)
        return json_str

    def _save_report(self, session_id: str, candidate_id: Optional[str], report: Dict,
                     video_duration: float = 0.0, fps: float = 12.0):
        """
        Save report to database (legacy method - for backward compatibility)

        Args:
            session_id: Session identifier
            candidate_id: Candidate identifier
            report: Complete report
            video_duration: Video duration in seconds
            fps: Frames per second

        Raises:
            DatabaseError: If save operation fails
        """
        try:
            report_with_metadata = {
                **report,
                "final_decision": "UNKNOWN",
                "confidence_scores": {
                    "final_suspicion_score": 0.0
                }
            }

            self.repository.save_report(session_id, candidate_id, report_with_metadata, video_duration, fps)
            debug_logger.info(f"Report saved for session: {session_id}")
        except Exception as e:
            debug_logger.error(f"Failed to save report for session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to save report: {str(e)}", 5002)

    def _save_complete_report(self, interview_id: str, report: Dict,
                             video_duration: float = 0.0, fps: float = 12.0):
        """
        Save complete proctoring report with all events and summaries to database

        Args:
            interview_id: Interview ID
            report: Complete analysis report
            video_duration: Video duration in seconds
            fps: Frames per second

        Raises:
            DatabaseError: If save operation fails
        """
        try:
            from datetime import date, datetime

            # Extract analysis data
            analysis = report.get("analysis", {})
            gestures = analysis.get("gestures", [])
            processing_metadata = analysis.get("processing_metadata", {})

            # Calculate risk score and level based on events
            risk_score, risk_level = self._calculate_risk_from_events(gestures)

            # Save proctoring report with risk assessment
            self.repository.save_proctoring_report(
                interview_id=interview_id,
                interview_date=date.today(),
                cheating_likelihood_score=risk_score,
                cheating_likelihood_level=risk_level
            )
            debug_logger.info(f"Proctoring report saved: risk_level={risk_level}, score={risk_score}")

            # Save individual events and summaries
            self._save_events_and_summaries(interview_id, gestures)

        except DatabaseError:
            raise
        except Exception as e:
            debug_logger.error(f"Failed to save complete report for interview {interview_id}: {str(e)}")
            raise DatabaseError(f"Failed to save complete report: {str(e)}", 5002)

    def _calculate_risk_from_events(self, gestures: List) -> tuple:
        """
        Calculate risk score and level from detected events

        Args:
            gestures: List of detected gestures

        Returns:
            Tuple of (risk_score, risk_level)
        """
        risk_score = 0
        event_count = sum(len(g.get("occurrence", [])) for g in gestures)

        # Simple risk scoring algorithm
        # More events = higher risk
        if event_count == 0:
            risk_score = 10  # Minimal score for clean
            risk_level = "CLEAN"
        elif event_count < 5:
            risk_score = 30
            risk_level = "MODERATE"
        elif event_count < 15:
            risk_score = 60
            risk_level = "SUSPICIOUS"
        else:
            risk_score = 85
            risk_level = "HIGH_RISK"

        return risk_score, risk_level

    def _parse_numeric_value(self, value) -> Optional[float]:
        """
        Parse numeric value from string or number, handling units like 'degrees', 'pixels', etc.

        Args:
            value: Value to parse (can be number or string with units like "62.1 degrees")

        Returns:
            Parsed float value or None
        """
        if value is None or value == "":
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Extract numeric part from strings like "62.1 degrees"
            import re
            match = re.search(r'-?\d+\.?\d*', value)
            if match:
                try:
                    return float(match.group())
                except (ValueError, AttributeError):
                    return None

        return None

    def _classify_event_risk(
        self,
        gesture_name: str,
        direction: Optional[str],
        duration_seconds: float,
    ) -> Optional[str]:
        if not direction:
            return None

        normalized_direction = direction.lower()
        if gesture_name == "head_movement":
            if normalized_direction in ("left", "right"):
                if duration_seconds < 0.4:
                    return "ignore"
                if duration_seconds < 1.5:
                    return "normal"
                if duration_seconds < 3.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction == "down":
                if duration_seconds < 0.4:
                    return "ignore"
                if duration_seconds < 2.0:
                    return "normal"
                if duration_seconds < 4.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction == "up":
                if duration_seconds < 0.4:
                    return "ignore"
                if duration_seconds < 5.0:
                    return "normal"
                if duration_seconds < 10.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction in ("up-left", "up-right"):
                if duration_seconds < 0.4:
                    return "ignore"
                if duration_seconds < 1.5:
                    return "suspicious"
                return "high_risk"
            if normalized_direction in ("down-left", "down-right"):
                if duration_seconds < 0.4:
                    return "ignore"
                if duration_seconds < 2.0:
                    return "suspicious"
                return "high_risk"
            return "normal"

        if gesture_name == "eye_movement":
            if normalized_direction in ("left", "right"):
                if duration_seconds < 0.6:
                    return "ignore"
                if duration_seconds < 2.0:
                    return "normal"
                if duration_seconds < 4.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction == "down":
                if duration_seconds < 0.6:
                    return "ignore"
                if duration_seconds < 1.5:
                    return "normal"
                if duration_seconds < 3.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction == "up":
                if duration_seconds < 0.6:
                    return "ignore"
                if duration_seconds < 1.0:
                    return "normal"
                if duration_seconds < 2.0:
                    return "suspicious"
                return "high_risk"
            if normalized_direction in ("up-left", "up-right", "down-left", "down-right"):
                if duration_seconds < 0.6:
                    return "ignore"
                if duration_seconds < 1.5:
                    return "suspicious"
                return "high_risk"
            return "normal"

        return None

    def _extract_velocity_label(self, value: object) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        if isinstance(value, (int, float)):
            return str(value)
        return None

    def _save_events_and_summaries(self, interview_id: str, gestures: List):
        """
        Save individual events and summaries to database

        Args:
            interview_id: Interview ID
            gestures: List of detected gestures
        """
        from datetime import datetime

        # Group events by type and save summaries and individual events
        for gesture in gestures:
            gesture_name = gesture.get("name", "unknown")
            occurrences = gesture.get("occurrence", [])

            total_count = len(occurrences)
            if total_count == 0:
                continue

            # Save individual event logs
            for idx, occurrence in enumerate(occurrences):
                timestamp = occurrence.get("timestamp", datetime.now())
                if isinstance(timestamp, (int, float)):
                    event_timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    try:
                        event_timestamp = datetime.fromisoformat(timestamp)
                    except:
                        event_timestamp = datetime.now()
                else:
                    event_timestamp = datetime.now()

                self.repository.save_event_log(
                    interview_id=interview_id,
                    event_type=gesture_name,
                    event_timestamp=event_timestamp,
                    duration=self._parse_numeric_value(occurrence.get("duration")) or 0.0,
                    direction=occurrence.get("direction", None),
                    intensity=self._parse_numeric_value(occurrence.get("intensity")),
                    confidence=self._parse_numeric_value(occurrence.get("confidence")),
                    velocity=self._extract_velocity_label(occurrence.get("velocity")),
                    event_risk=self._classify_event_risk(
                        gesture_name=gesture_name,
                        direction=occurrence.get("direction", None),
                        duration_seconds=self._parse_numeric_value(occurrence.get("duration")) or 0.0,
                    ),
                )
            debug_logger.info(f"Saved {total_count} individual event logs for gesture: {gesture_name}")

            # Calculate counts by severity
            normal_count = int(total_count * 0.3)  # Assume 30% normal
            suspicious_count = int(total_count * 0.5)  # Assume 50% suspicious
            high_risk_count = total_count - normal_count - suspicious_count

            # Calculate total duration
            total_duration = sum(self._parse_numeric_value(o.get("duration")) or 0.0 for o in occurrences)

            # Save event summary
            self.repository.save_event_summary(
                interview_id=interview_id,
                event_type=gesture_name,
                total_count=total_count,
                normal_count=normal_count,
                suspicious_count=suspicious_count,
                high_risk_count=high_risk_count,
                total_duration=total_duration,
                correlated_count=0
            )
            debug_logger.info(f"Saved event summary: {gesture_name} ({total_count} events, {total_duration:.1f}s duration)")

    def get_report(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve proctoring report by session ID

        Args:
            session_id: Session identifier

        Returns:
            Report dictionary

        Raises:
            ReportNotFoundError: If report doesn't exist
            DatabaseError: If database query fails
        """
        try:
            report = self.repository.get_report_by_session(session_id)

            if not report:
                debug_logger.warning(f"Report not found for session: {session_id}")
                raise ReportNotFoundError()

            return report

        except ReportNotFoundError:
            raise
        except Exception as e:
            debug_logger.error(f"Error retrieving report for session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve report: {str(e)}", 5003)

    def get_candidate_reports(self, candidate_id: str, limit: int = 10) -> Dict:
        """
        Retrieve all reports for a candidate

        Args:
            candidate_id: Candidate identifier
            limit: Maximum reports to return

        Returns:
            Dictionary with candidate reports

        Raises:
            DatabaseError: If database query fails
        """
        try:
            reports = self.repository.get_reports_by_candidate(candidate_id, limit)

            debug_logger.info(f"Retrieved {len(reports)} reports for candidate: {candidate_id}")

            return {
                "candidate_id": candidate_id,
                "total_reports": len(reports),
                "reports": reports
            }

        except Exception as e:
            debug_logger.error(f"Error retrieving reports for candidate {candidate_id}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve candidate reports: {str(e)}", 5004)

    def delete_report(self, session_id: str) -> Dict:
        """
        Delete report (GDPR compliance)

        Args:
            session_id: Session identifier

        Returns:
            Success message dictionary

        Raises:
            ReportNotFoundError: If report doesn't exist
            DatabaseError: If delete operation fails
        """
        try:
            success = self.repository.delete_report(session_id)

            if not success:
                debug_logger.warning(f"Report not found for deletion: {session_id}")
                raise ReportNotFoundError()

            debug_logger.info(f"Report deleted successfully: {session_id}")

            return {
                "status": "success",
                "message": f"Report {session_id} deleted successfully"
            }

        except ReportNotFoundError:
            raise
        except Exception as e:
            debug_logger.error(f"Error deleting report for session {session_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete report: {str(e)}", 5005)
