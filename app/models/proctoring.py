"""
Database Models for Proctoring System
Creates the proctoring database schema:
- proctoring_events_logs: Individual event/frame-level data
- proctoring_reports: Summary reports with risk scores
- proctoring_event_summary: Aggregated event summaries
"""

from sqlalchemy import Column, String, DateTime, Float, Integer, Date, func, Index, ForeignKey
from sqlalchemy.dialects.mysql import TEXT
from datetime import datetime
import uuid as uuid_lib

from app.core.database import Base


class ProctoringEventLog(Base):
    """
    Proctoring Event Logs Table
    Stores individual event/frame-level detection data
    References proctoring_reports via interview_id.
    """
    __tablename__ = 'proctoring_events_logs'

    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid_lib.uuid4()))

    # Foreign key to proctoring_reports (one report per interview)
    interview_id = Column(String(36), ForeignKey('proctoring_reports.interview_id', ondelete='CASCADE'), nullable=False, index=True)

    # Event Data
    event_type = Column(String(100), nullable=False, index=True)
    event_timestamp = Column(DateTime, nullable=False, index=True)

    # Event Metrics
    duration = Column(Float, nullable=True)
    direction = Column(String(50), nullable=True)
    intensity = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    # Stored as label (e.g. "slow", "moderate", "rapid") from analyzer output
    velocity = Column(String(20), nullable=True)
    # Per-event risk classification (e.g. "normal", "suspicious", "high_risk")
    event_risk = Column(String(20), nullable=True, index=True)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_interview_timestamp', 'interview_id', 'event_timestamp'),
        Index('idx_event_type_timestamp', 'event_type', 'event_timestamp'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )

    def to_dict(self):
        return {
            'id': self.id,
            'interview_id': self.interview_id,
            'event_type': self.event_type,
            'event_timestamp': self.event_timestamp.isoformat() if self.event_timestamp else None,
            'duration': self.duration,
            'direction': self.direction,
            'intensity': self.intensity,
            'confidence': self.confidence,
            'velocity': self.velocity,
            'event_risk': self.event_risk,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ProctoringReport(Base):
    """
    Proctoring Reports Table
    Stores complete proctoring analysis reports with risk scores.
    One row per interview; interview_id is the primary key.
    """
    __tablename__ = 'proctoring_reports'

    # Primary Key: one report per interview
    interview_id = Column(String(36), primary_key=True)

    # Report Data
    interview_date = Column(Date, nullable=False, index=True)
    cheating_likelihood_score = Column(Integer, nullable=False)
    cheating_likelihood_level = Column(String(50), nullable=False, index=True)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_interview_created', 'interview_id', 'created_at'),
        Index('idx_risk_score', 'cheating_likelihood_score'),
        Index('idx_risk_level', 'cheating_likelihood_level'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )

    def to_dict(self):
        return {
            'id': self.interview_id,
            'interview_id': self.interview_id,
            'interview_date': self.interview_date.isoformat() if self.interview_date else None,
            'cheating_likelihood_score': self.cheating_likelihood_score,
            'cheating_likelihood_level': self.cheating_likelihood_level,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ProctoringEventSummary(Base):
    """
    Proctoring Event Summary Table
    Stores aggregated/summarized event data by type.
    References proctoring_reports via interview_id.
    """
    __tablename__ = 'proctoring_event_summary'

    # Primary Key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid_lib.uuid4()))

    # Foreign key to proctoring_reports (one report per interview)
    interview_id = Column(String(36), ForeignKey('proctoring_reports.interview_id', ondelete='CASCADE'), nullable=False, index=True)

    # Summary Data
    event_type = Column(String(100), nullable=False, index=True)
    total_count = Column(Integer, nullable=False, default=0)
    normal_count = Column(Integer, nullable=False, default=0)
    suspicious_count = Column(Integer, nullable=False, default=0)
    high_risk_count = Column(Integer, nullable=False, default=0)

    # Duration Metrics
    total_duration = Column(Float, nullable=False, default=0.0)

    # Correlation/Analysis
    correlated_count = Column(Integer, nullable=False, default=0)

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_interview_event_type', 'interview_id', 'event_type'),
        Index('idx_event_type_created', 'event_type', 'created_at'),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )

    def to_dict(self):
        return {
            'id': self.id,
            'interview_id': self.interview_id,
            'event_type': self.event_type,
            'total_count': self.total_count,
            'normal_count': self.normal_count,
            'suspicious_count': self.suspicious_count,
            'high_risk_count': self.high_risk_count,
            'total_duration': self.total_duration,
            'correlated_count': self.correlated_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

