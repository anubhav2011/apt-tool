"""
Dependency Injection
Clean dependency management following FastAPI best practices
"""
from app.utils.logger import debug_logger
from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.proctoring_config import ProctoringConfig
from app.core.database import get_db_session
from app.repositories.base_repository import IRepository
from app.repositories.proctoring_repository import ProctoringRepository
from app.services.base_service import IProctoringService
from app.services.proctoring_service import ProctoringService
from app.models.proctoring import ProctoringReport

logger = debug_logger

def get_config() -> ProctoringConfig:
    """
    Get configuration instance
    Returns singleton configuration instance

    Returns:
        ProctoringConfig instance
    """
    return ProctoringConfig()


def get_repository(db: Session = Depends(get_db_session)) -> IRepository[ProctoringReport]:
    """
    Get repository instance with injected database session
    Returns base interface type for better abstraction and testability
    The concrete implementation (ProctoringRepository) extends BaseRepository

    Args:
        db: Database session from FastAPI dependency

    Returns:
        Repository instance implementing IRepository interface
    """
    # BaseRepository is called through inheritance in ProctoringRepository
    # ProctoringRepository(db) -> calls BaseRepository.__init__(ProctoringReport, db)
    return ProctoringRepository(db=db)


def get_proctoring_service(
    repository: IRepository[ProctoringReport] = Depends(get_repository)
) -> IProctoringService:
    """
    Get proctoring service instance with proper dependency injection
    All dependencies are injected via interfaces for loose coupling

    Args:
        repository: Injected repository implementing IRepository base interface

    Returns:
        Proctoring service instance implementing IProctoringService
    """
    return ProctoringService(repository=repository)
