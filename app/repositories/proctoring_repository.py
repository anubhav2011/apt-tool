"""
Proctoring Repository
All database operations for proctoring reports using SQLAlchemy ORM
Contains actual implementation of database logic
"""
from app.utils.logger import debug_logger
from typing import Optional, List, Dict
import uuid
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.proctoring import ProctoringReport
from app.repositories.base_repository import BaseRepository

logger = debug_logger


class ProctoringRepository(BaseRepository[ProctoringReport]):
    """
    Repository for proctoring data persistence
    Handles all database CRUD operations using SQLAlchemy ORM
    Contains actual database logic implementation
    No abstract methods - only concrete implementations
    """

    def __init__(self, db: Session):
        """
        Initialize repository with database session

        Args:
            db: SQLAlchemy database session
        """
        super().__init__(ProctoringReport, db)
        debug_logger.info("ProctoringRepository initialized")

    def get_configuration(self) -> Optional[Dict]:
        """
        Get all proctoring configuration from database
        Fetches configuration data from proctoring_config table

        Returns:
            Dict: Configuration dictionary with all proctoring settings
            None: If configuration doesn't exist in database
        """
        try:
            # TODO: Query actual proctoring_config table when model is created
            # For now, return default configuration structure
            config = {
                'max_frame_dimension': 1280,
                'target_fps': 12,
                'calibration_duration_sec': 8.0,
                'calibration_min_frames': 80,
                'max_head_rotation_speed': 10.0,
                'max_bbox_center_shift': 0.05,
                'max_eye_angle_variance': 0.4,
                'min_face_presence_score': 0.85,
                'max_baseline_yaw': 20.0,
                'max_baseline_pitch': 15.0,
                'max_baseline_roll': 12.0,
                'max_baseline_eye': 0.8,
                'multiplier_eye': 2.0,
                'multiplier_yaw': 1.6,
                'multiplier_pitch': 1.4,
                'multiplier_roll': 1.2,
                'max_adaptive_eye': 1.4,
                'max_adaptive_yaw': 35.0,
                'max_adaptive_pitch': 30.0,
                'max_adaptive_roll': 25.0,
                'severity_suspicious_count_min': 15,
                'severity_high_risk_count_min': 30,
                'weight_gaze': 0.45,
                'weight_head': 0.30,
                'weight_face': 0.25,
                'risk_clean_max': 0.20,
                'risk_borderline_max': 0.40,
                'risk_suspicious_max': 0.65
            }

            debug_logger.info("Retrieved proctoring configuration from database")
            return config

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving configuration: {e}")
            return None

    def create(self, obj: ProctoringReport) -> Optional[ProctoringReport]:
        """
        Create a new proctoring report record in database

        Args:
            obj: ProctoringReport instance to create

        Returns:
            Created ProctoringReport instance or None if failed
        """
        try:
            self.db.add(obj)
            self.db.commit()
            self.db.refresh(obj)
            debug_logger.info(f"Created ProctoringReport record: {obj.id}")
            return obj
        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to create ProctoringReport: {e}")
            return None

    def get_by_id(self, id: str) -> Optional[ProctoringReport]:
        """
        Get proctoring report by ID from database

        Args:
            id: Record ID

        Returns:
            ProctoringReport instance or None if not found
        """
        try:
            return self.db.query(ProctoringReport).filter(ProctoringReport.id == id).first()
        except SQLAlchemyError as e:
            debug_logger.error(f"Failed to get ProctoringReport by ID: {e}")
            return None

    def get_all(self, limit: int = 100, offset: int = 0) -> List[ProctoringReport]:
        """
        Get all proctoring reports with pagination from database

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of ProctoringReport instances
        """
        try:
            return (
                self.db.query(ProctoringReport)
                .order_by(ProctoringReport.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )
        except SQLAlchemyError as e:
            debug_logger.error(f"Failed to get all ProctoringReports: {e}")
            return []

    def save_report(self, session_id: str, candidate_id: Optional[str],
                    report: Dict, video_duration: float = 0.0, fps: float = 12.0) -> bool:
        """
        Save proctoring report to database with all metadata

        Args:
            session_id: Unique session identifier
            candidate_id: Candidate identifier
            report: Complete proctoring report dictionary
            video_duration: Video duration in seconds
            fps: Frames per second

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            report_id = str(uuid.uuid4())
            final_decision = report.get('final_decision', 'UNKNOWN')
            final_suspicion_score = report.get('confidence_scores', {}).get('final_suspicion_score', 0.0)

            # Create database object
            db_report = ProctoringReport(
                id=report_id,
                session_id=session_id,
                candidate_id=candidate_id,
                json_report=report,
                video_duration_sec=video_duration,
                fps=fps,
                final_decision=final_decision,
                final_suspicion_score=final_suspicion_score
            )

            # Use create method
            result = self.create(db_report)
            if result:
                debug_logger.info(f"Report saved successfully for session: {session_id}")
                return True
            return False

        except Exception as e:
            debug_logger.error(f"Failed to save report for session {session_id}: {e}")
            return False

    def get_report_by_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve proctoring report by session ID from database

        Args:
            session_id: Session identifier

        Returns:
            Dict: Report data or None if not found
        """
        try:
            report = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.session_id == session_id)
                .order_by(ProctoringReport.created_at.desc())
                .first()
            )

            if report:
                debug_logger.info(f"Retrieved report for session: {session_id}")
                return report.json_report

            debug_logger.warning(f"No report found for session: {session_id}")
            return None

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving report for session {session_id}: {e}")
            return None

    def get_reports_by_candidate(self, candidate_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve all reports for a specific candidate from database

        Args:
            candidate_id: Candidate identifier
            limit: Maximum number of reports to return

        Returns:
            List[Dict]: List of reports
        """
        try:
            reports = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.candidate_id == candidate_id)
                .order_by(ProctoringReport.created_at.desc())
                .limit(limit)
                .all()
            )

            debug_logger.info(f"Retrieved {len(reports)} reports for candidate: {candidate_id}")
            return [report.json_report for report in reports]

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving reports for candidate {candidate_id}: {e}")
            return []

    def delete_report(self, session_id: str) -> bool:
        """
        Delete report by session ID from database (GDPR compliance)

        Args:
            session_id: Session identifier

        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            rows_deleted = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.session_id == session_id)
                .delete()
            )
            self.db.commit()

            if rows_deleted > 0:
                debug_logger.info(f"Deleted {rows_deleted} report(s) for session: {session_id}")
                return True

            debug_logger.warning(f"No report found to delete for session: {session_id}")
            return False

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Database error deleting report for session {session_id}: {e}")
            return False

    def get_all_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get all session reports with pagination from database

        Args:
            limit: Maximum number of reports to return
            offset: Number of records to skip

        Returns:
            List[Dict]: List of session summaries
        """
        try:
            reports = self.get_all(limit=limit, offset=offset)
            debug_logger.info(f"Retrieved {len(reports)} session reports")
            return [report.to_dict() for report in reports]

        except Exception as e:
            debug_logger.error(f"Error retrieving all sessions: {e}")
            return []

    def get_reports_by_decision(self, decision: str, limit: int = 50) -> List[Dict]:
        """
        Get reports filtered by final decision from database

        Args:
            decision: Final decision (e.g., 'CLEAN', 'SUSPICIOUS', 'HIGH_RISK')
            limit: Maximum number of reports to return

        Returns:
            List[Dict]: List of reports with specified decision
        """
        try:
            reports = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.final_decision == decision)
                .order_by(ProctoringReport.created_at.desc())
                .limit(limit)
                .all()
            )

            debug_logger.info(f"Retrieved {len(reports)} reports with decision: {decision}")
            return [report.json_report for report in reports]

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving reports by decision {decision}: {e}")
            return []
