"""
Proctoring Service
Main service orchestrating all proctoring business logic
"""
from app.utils.logger import debug_logger
from typing import Dict, Optional
import uuid
import os
import tempfile
import json
import re

from app.core.proctoring_config import ProctoringConfig
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

    async def process_video_upload(self, video_file, session_id: Optional[str] = None,
                                   candidate_id: Optional[str] = None) -> Dict:
        """
        Process uploaded video file with validation and temp file handling

        Args:
            video_file: UploadFile object from FastAPI
            session_id: Optional session ID
            candidate_id: Optional candidate ID

        Returns:
            Processing result with formatted report

        Raises:
            ValidationError: If validation fails
            VideoProcessingError: If processing fails
            DatabaseError: If database operation fails
        """
        temp_file = None
        temp_path = None

        try:
            session_id = session_id or str(uuid.uuid4())
            debug_logger.info(f"Processing uploaded video for session: {session_id}")

            # Validate file format
            file_ext = os.path.splitext(video_file.filename)[1].lower()
            if file_ext not in ProctoringConfig.ALLOWED_VIDEO_FORMATS:
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
            if file_size_mb > ProctoringConfig.MAX_VIDEO_SIZE_MB:
                raise ValidationError(
                    f"Video file too large: {file_size_mb:.2f}MB exceeds maximum {ProctoringConfig.MAX_VIDEO_SIZE_MB}MB",
                    4002
                )

            debug_logger.info(f"Video file size: {file_size_mb:.2f} MB")

            # Process video
            result = self.process_video_file(temp_path, session_id, candidate_id)

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

    async def process_video_from_url(self, video_url: str, session_id: Optional[str] = None,
                                     candidate_id: Optional[str] = None) -> Dict:
        """
        Process video from URL with validation

        Args:
            video_url: URL to video file
            session_id: Optional session ID
            candidate_id: Optional candidate ID

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

            session_id = session_id or str(uuid.uuid4())
            debug_logger.info(f"Processing video from URL for session: {session_id}")

            # Process video
            result = self.process_video_file(video_url, session_id, candidate_id)

            # Format response
            formatted_result = self._format_json_response(result)

            return json.loads(formatted_result)

        except (ValidationError, VideoProcessingError, DatabaseError):
            raise
        except Exception as e:
            debug_logger.error(f"Unexpected error processing video from URL: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Failed to process video from URL: {str(e)}", 5001)

    def process_video_file(self, video_path: str, session_id: Optional[str] = None,
                          candidate_id: Optional[str] = None) -> Dict:
        """
        Process video file and generate proctoring report

        Args:
            video_path: Path or URL to video
            session_id: Optional session ID
            candidate_id: Optional candidate ID

        Returns:
            Processing result with report containing only gestures array

        Raises:
            VideoProcessingError: If video processing fails
            DatabaseError: If database operation fails
        """
        session_id = session_id or str(uuid.uuid4())
        debug_logger.info(f"Processing video for session: {session_id}")

        try:
            # Process video using video processing service
            processor = VideoProcessingService(self.config)
            report = processor.process_video(video_path, session_id)
            processor.cleanup()

            gestures = report.get("analysis", {}).get("gestures", [])

            debug_logger.info(f"[v0] Extracted {len(gestures)} gesture types from report")

            # Count total occurrences
            total_occurrences = sum(len(g.get("occurrence", [])) for g in gestures)
            debug_logger.info(f"[v0] Total gesture occurrences: {total_occurrences}")

            # Save full report to database for internal tracking
            processing_metadata = report.get("analysis", {}).get("processing_metadata", {})
            video_duration = processing_metadata.get("video_duration_sec", 0.0)
            fps = 12.0

            self._save_report(session_id, candidate_id, report, video_duration, fps)

            debug_logger.info(f"Video processed successfully for session: {session_id}")

            return {
                "gestures": gestures
            }

        except Exception as e:
            debug_logger.error(f"Error processing video for session {session_id}: {str(e)}", exc_info=True)
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
        Save report to database

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
