"""
Proctoring Repository
All database operations for proctoring reports using SQLAlchemy ORM
Contains actual implementation of database logic
"""
from app.utils.logger import debug_logger
from typing import Optional, List, Dict
import uuid
from datetime import datetime, date
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.models.proctoring import (
    ProctoringReport,
    ProctoringEventLog,
    ProctoringEventSummary
)
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
            debug_logger.info(f"Created ProctoringReport record: {obj.interview_id}")
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
            return self.db.query(ProctoringReport).filter(ProctoringReport.interview_id == id).first()
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
        Save proctoring report to database (legacy method for backward compatibility)
        Maps old session-based approach to new interview-based schema

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
            # Use session_id as interview_id (no separate interviews table in this app)
            interview_id = session_id

            # Save proctoring report
            return self.save_proctoring_report(
                interview_id=interview_id,
                interview_date=date.today(),
                cheating_likelihood_score=int(report.get('confidence_scores', {}).get('final_suspicion_score', 0) * 100),
                cheating_likelihood_level=report.get('final_decision', 'UNKNOWN')
            )

        except Exception as e:
            debug_logger.error(f"Failed to save report for session {session_id}: {e}")
            return False

    def get_report_by_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve proctoring report by session ID from database
        Uses Interview session_id to find related reports

        Args:
            session_id: Session identifier

        Returns:
            Dict: Report data or None if not found
        """
        try:
            # session_id is used as interview_id (one report per interview)
            report = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == session_id)
                .first()
            )

            if report:
                debug_logger.info(f"Retrieved report for session: {session_id}")
                return report.to_dict()

            debug_logger.warning(f"No report found for session: {session_id}")
            return None

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving report for session {session_id}: {e}")
            return None

    def get_reports_by_candidate(self, candidate_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve all reports for a specific candidate from database.
        Without an Interview model, returns all reports up to limit (candidate_id ignored).
        """
        try:
            reports = (
                self.db.query(ProctoringReport)
                .order_by(ProctoringReport.created_at.desc())
                .limit(limit)
                .all()
            )
            debug_logger.info(f"Retrieved {len(reports)} reports for candidate: {candidate_id}")
            return [report.to_dict() for report in reports]
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
            # session_id is used as interview_id for proctoring report lookup
            rows_deleted = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == session_id)
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
                .filter(ProctoringReport.cheating_likelihood_level == decision)
                .order_by(ProctoringReport.created_at.desc())
                .limit(limit)
                .all()
            )

            debug_logger.info(f"Retrieved {len(reports)} reports with decision: {decision}")
            return [report.to_dict() for report in reports]

        except SQLAlchemyError as e:
            debug_logger.error(f"Database error retrieving reports by decision {decision}: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────────
    # EVENT LOG METHODS
    # ─────────────────────────────────────────────────────────────────────────────

    def save_event_log(self, interview_id: str, event_type: str,
                       event_timestamp: datetime,
                       duration: Optional[float] = None,
                       direction: Optional[str] = None,
                       intensity: Optional[float] = None,
                       confidence: Optional[float] = None,
                       velocity: Optional[str] = None,
                       event_risk: Optional[str] = None) -> bool:
        """
        Save individual event log to database

        Args:
            interview_id: ID of the interview this event belongs to
            event_type: Type of event (gaze_left, head_right, face_missing, etc)
            event_timestamp: When the event occurred
            duration: Duration of the event
            direction: Direction of movement
            intensity: Intensity/severity of the event
            confidence: Confidence score of detection
            velocity: Speed of movement

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            event_log = ProctoringEventLog(
                id=str(uuid.uuid4()),
                interview_id=interview_id,
                event_type=event_type,
                event_timestamp=event_timestamp,
                duration=duration,
                direction=direction,
                intensity=intensity,
                confidence=confidence,
                velocity=velocity,
                event_risk=event_risk,
            )

            self.db.add(event_log)
            self.db.commit()
            debug_logger.info(f"Created EventLog: {event_type} for interview {interview_id}")
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save event log: {e}")
            return False

    def save_event_summary(self, interview_id: str, event_type: str,
                          total_count: int, normal_count: int,
                          suspicious_count: int, high_risk_count: int,
                          total_duration: float, correlated_count: int = 0) -> bool:
        """
        Save aggregated event summary to database

        Args:
            interview_id: ID of the interview
            event_type: Type of event being summarized
            total_count: Total occurrences
            normal_count: Normal/acceptable occurrences
            suspicious_count: Suspicious occurrences
            high_risk_count: High risk occurrences
            total_duration: Total duration in seconds
            correlated_count: Number of correlated events

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            summary = ProctoringEventSummary(
                id=str(uuid.uuid4()),
                interview_id=interview_id,
                event_type=event_type,
                total_count=total_count,
                normal_count=normal_count,
                suspicious_count=suspicious_count,
                high_risk_count=high_risk_count,
                total_duration=total_duration,
                correlated_count=correlated_count
            )

            self.db.add(summary)
            self.db.commit()
            debug_logger.info(f"Created EventSummary: {event_type} for interview {interview_id}")
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save event summary: {e}")
            return False

    def save_proctoring_report(self, interview_id: str, interview_date: date,
                              cheating_likelihood_score: int,
                              cheating_likelihood_level: str) -> bool:
        """
        Save proctoring report (risk score and level) to database

        Args:
            interview_id: ID of the interview
            interview_date: Date of the interview
            cheating_likelihood_score: Risk score (0-100)
            cheating_likelihood_level: Risk level (CLEAN, MODERATE, SUSPICIOUS, HIGH_RISK)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            existing = self.db.query(ProctoringReport).filter(ProctoringReport.interview_id == interview_id).first()
            if existing:
                existing.interview_date = interview_date
                existing.cheating_likelihood_score = cheating_likelihood_score
                existing.cheating_likelihood_level = cheating_likelihood_level
                self.db.commit()
            else:
                report = ProctoringReport(
                    interview_id=interview_id,
                    interview_date=interview_date,
                    cheating_likelihood_score=cheating_likelihood_score,
                    cheating_likelihood_level=cheating_likelihood_level
                )
                self.db.add(report)
                self.db.commit()
            debug_logger.info(
                f"Created ProctoringReport: {cheating_likelihood_level} ({cheating_likelihood_score}/100) "
                f"for interview {interview_id}"
            )
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save proctoring report: {e}")
            return False

    def get_report_by_interview_id(self, interview_id: str) -> Optional[Dict]:
        """Get proctoring report for an interview"""
        try:
            report = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == interview_id)
                .order_by(ProctoringReport.created_at.desc())
                .first()
            )

            if report:
                return report.to_dict()
            return None

        except SQLAlchemyError as e:
            debug_logger.error(f"Failed to get report for interview {interview_id}: {e}")
            return None

    def get_event_summaries(self, interview_id: str) -> List[Dict]:
        """Get all event summaries for an interview"""
        try:
            summaries = (
                self.db.query(ProctoringEventSummary)
                .filter(ProctoringEventSummary.interview_id == interview_id)
                .order_by(ProctoringEventSummary.event_type)
                .all()
            )

            return [summary.to_dict() for summary in summaries]

        except SQLAlchemyError as e:
            debug_logger.error(f"Failed to get event summaries for interview {interview_id}: {e}")
            return []


