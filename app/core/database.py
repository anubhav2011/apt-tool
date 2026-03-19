"""
Database Configuration
SQLAlchemy setup and session management
"""
from app.utils.logger import debug_logger
from sqlalchemy import create_engine, text, URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

# Create declarative base for models
Base = declarative_base()

# Global engine and session factory
engine = None
SessionLocal = None


def init_database(config):
    """
    Initialize database engine and session factory
    Creates database if it doesn't exist

    Args:
        config: Configuration object with database settings
    """
    global engine, SessionLocal

    try:
        mysql_host     = config.HOST
        mysql_port     = config.PORT
        mysql_user     = config.USER
        mysql_password = config.PASSWORD
        mysql_database = config.DATABASE

        debug_logger.info(f"Initializing database connection...")
        debug_logger.info(f"Host: {mysql_host}")
        debug_logger.info(f"Port: {mysql_port}")
        debug_logger.info(f"User: {mysql_user}")
        debug_logger.info(f"Database: {mysql_database}")

        # Step 1: Connect to MySQL without database to create database
        debug_logger.info("Connecting to MySQL server to create database...")

        temp_url_params = {
            "drivername": "mysql+pymysql",
            "username": mysql_user,
            "host": mysql_host,
            "port": mysql_port,
            "query": {"charset": "utf8mb4"}
        }

        if mysql_password:
            temp_url_params["password"] = mysql_password

        temp_database_url = URL.create(**temp_url_params)
        temp_engine = create_engine(
            temp_database_url,
            connect_args={"connect_timeout": 10},
            echo=False
        )

        try:
            with temp_engine.connect() as conn:
                # Test connection
                result = conn.execute(text("SELECT 1"))
                debug_logger.info(f"Connected to MySQL server: {result.scalar()}")

                # Create database if it doesn't exist
                debug_logger.info(f"Creating database '{mysql_database}' if it doesn't exist...")
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{mysql_database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
                debug_logger.info(f"Database '{mysql_database}' ready")
        finally:
            temp_engine.dispose()

        # Step 2: Create main engine with database specified
        debug_logger.info(f"Connecting to database '{mysql_database}'...")

        main_url_params = {
            "drivername": "mysql+pymysql",
            "username": mysql_user,
            "host": mysql_host,
            "port": mysql_port,
            "database": mysql_database,
            "query": {"charset": "utf8mb4"}
        }

        if mysql_password:
            main_url_params["password"] = mysql_password

        main_database_url = URL.create(**main_url_params)

        engine = create_engine(
            main_database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=5,
            max_overflow=10,
            echo=False,
            connect_args={"connect_timeout": 10}
        )

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            debug_logger.info(f"Connected to database '{mysql_database}': {result.scalar()}")

        # Step 3: Create session factory
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

        debug_logger.info("[OK] Database engine initialized successfully")
        return True

    except ImportError as e:
        debug_logger.error(
            f"[ERROR] Database driver not found: {e}\n"
            "Please install: pip install pymysql cryptography"
        )
        raise
    except Exception as e:
        debug_logger.error(f"[ERROR] Database initialization failed: {type(e).__name__}: {e}")
        raise


def create_tables():
    """Create all database tables using SQLAlchemy ORM models"""
    global engine
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    try:
        # Import models to register them with Base
        from app.models.proctoring import (
            ProctoringReport,
            ProctoringEventLog,
            ProctoringEventSummary
        )

        # Drop existing proctoring tables so they are recreated with the new schema
        # (interview_id as PK on proctoring_reports; events/summary FK to proctoring_reports)
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            conn.execute(text("DROP TABLE IF EXISTS proctoring_events_logs"))
            conn.execute(text("DROP TABLE IF EXISTS proctoring_event_summary"))
            conn.execute(text("DROP TABLE IF EXISTS proctoring_reports"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            conn.commit()

        # Create all tables from registered models (proctoring_reports first due to FKs)
        Base.metadata.create_all(bind=engine)
        debug_logger.info("[OK] Database tables created successfully")

    except Exception as e:
        debug_logger.error(f"[ERROR] Failed to create tables: {e}")
        raise


@contextmanager
def get_db() -> Session:
    """
    Get database session context manager

    Usage:
        with get_db() as db:
            db.query(Model).all()

    Yields:
        SQLAlchemy Session
    """
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Get database session for dependency injection

    Usage with FastAPI:
        @app.get("/")
        def endpoint(db: Session = Depends(get_db_session)):
            ...

    Yields:
        SQLAlchemy Session
    """
    global SessionLocal
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
