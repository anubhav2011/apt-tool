"""
Base Repository
Generic base repository interface with essential abstract methods
"""
from app.utils.logger import debug_logger
from abc import ABC, abstractmethod
from typing import Optional, Generic, TypeVar, Type, Dict
from sqlalchemy.orm import Session

from app.core.database import Base

# Generic type for SQLAlchemy models
ModelType = TypeVar("ModelType", bound=Base)


class IRepository(ABC, Generic[ModelType]):
    """
    Interface for repository operations
    Only essential abstract methods are defined here
    """

    @abstractmethod
    def get_configuration(self) -> Optional[Dict]:
        """
        Get configuration from database
        Returns configuration dictionary with all proctoring settings
        """
        pass

    def get_report_by_session(self, session_id):
        pass

    def save_report(self, session_id, candidate_id, report_with_metadata, video_duration, fps):
        pass

    def get_reports_by_candidate(self, candidate_id, limit):
        pass

    def delete_report(self, session_id):
        pass


class BaseRepository(IRepository[ModelType]):
    """
    Generic base repository class
    Provides common initialization pattern
    Subclasses implement abstract methods with actual database logic
    """

    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository with model and database session

        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
        debug_logger.debug(f"{self.__class__.__name__} initialized with model {model.__name__}")

    @abstractmethod
    def get_configuration(self) -> Optional[Dict]:
        """Get configuration from database - must be implemented in subclass"""
        pass
