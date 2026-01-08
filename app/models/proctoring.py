"""
Proctoring Database Models
SQLAlchemy ORM models for proctoring reports

Table: proctoring_reports (MySQL Storage Model from PDF)
"""
from sqlalchemy import Column, String, JSON, DateTime, Float, Integer, func, Index

from app.core.database import Base


class ProctoringReport(Base):
    """
    Proctoring Report Model
    Stores complete proctoring analysis results matching PDF specification

    Columns (from PDF):
    - id: INT AUTO (Primary key)
    - session_id: VARCHAR (Unique session identifier)
    - calibration: JSON (Calibration data and thresholds)
    - counts: JSON (Event occurrence counts)
    - duration_sec: JSON (Duration by risk category)
    - intensity: JSON (Average angle deviations)
    - alerts: JSON (Timeline of suspicious events)
    - risk_score: INT (Final risk score 0-100)
    - risk_level: VARCHAR (Risk classification)
    - created_at: TIMESTAMP (Record creation time)
    """
    __tablename__ = 'proctoring_reports'

    # Primary Key
    id = Column(String(36), primary_key=True, index=True)

    # Session Information
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    candidate_id = Column(String(255), nullable=True, index=True)

    # Report Data (JSON columns per PDF spec)
    calibration = Column(JSON, nullable=False, comment="Calibration data and thresholds")
    counts = Column(JSON, nullable=False, comment="Event occurrence counts")
    duration_sec = Column(JSON, nullable=False, comment="Duration by risk category")
    intensity = Column(JSON, nullable=False, comment="Average angle deviations")
    alerts = Column(JSON, nullable=False, comment="Timeline of suspicious events")

    # Risk Assessment (per PDF spec)
    risk_score = Column(Integer, nullable=False, index=True, comment="Final risk score (0-100)")
    risk_level = Column(String(50), nullable=False, index=True, comment="Risk classification")

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True, comment="Record creation time")
    video_duration_sec = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)

    # Legacy field for backward compatibility
    json_report = Column(JSON, nullable=True, comment="Complete JSON report (legacy)")
    final_suspicion_score = Column(Float, nullable=True, comment="Legacy suspicion score (deprecated)")
    final_decision = Column(String(50), nullable=True, comment="Legacy decision field (deprecated)")

    # Create composite indexes for common queries
    __table_args__ = (
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_candidate_created', 'candidate_id', 'created_at'),
        Index('idx_risk_score', 'risk_score', 'created_at'),
        Index('idx_risk_level', 'risk_level', 'created_at'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )

    def __repr__(self):
        return f"<ProctoringReport(id={self.id}, session_id={self.session_id}, risk_level={self.risk_level}, risk_score={self.risk_score})>"

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'candidate_id': self.candidate_id,
            'calibration': self.calibration,
            'counts': self.counts,
            'duration_sec': self.duration_sec,
            'intensity': self.intensity,
            'alerts': self.alerts,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'video_duration_sec': self.video_duration_sec,
            'fps': self.fps
        }
