"""
Database Configuration
SQLAlchemy setup and session management
"""
from app.utils.logger import debug_logger
from sqlalchemy import create_engine, text, URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from urllib.parse import quote_plus

# Create declarative base for models
Base = declarative_base()

# Global engine and session factory
engine = None
SessionLocal = None


def init_database(config):
    """
    Initialize database engine and session factory

    Args:
        config: Configuration object with database settings
    """
    global engine, SessionLocal

    try:
        mysql_host = config.MYSQL_HOST
        mysql_port = config.MYSQL_PORT
        mysql_user = config.MYSQL_USER
        mysql_password = config.MYSQL_PASSWORD
        mysql_database = config.MYSQL_DATABASE

        debug_logger.info(f"Initializing database connection...")
        debug_logger.info(f"Host: {mysql_host}")
        debug_logger.info(f"Port: {mysql_port}")
        debug_logger.info(f"User: {mysql_user}")
        debug_logger.info(f"Database: {mysql_database}")
        debug_logger.info(f"Password: {'(set)' if mysql_password else '(empty)'}")

        # This prevents parsing issues with special characters
        url_params = {
            "drivername": "mysql+pymysql",
            "username": mysql_user,
            "password": mysql_password if mysql_password else None,
            "host": mysql_host,
            "port": mysql_port,
            "database": mysql_database,
            "query": {
                "charset": "utf8mb4"
            }
        }

        # Remove password key if empty to avoid authentication issues
        if not mysql_password:
            url_params.pop("password")

        database_url = URL.create(**url_params)

        # Log sanitized URL for debugging
        safe_url = f"mysql+pymysql://{mysql_user}:{'***' if mysql_password else '(none)'}@{mysql_host}:{mysql_port}/{mysql_database}"
        debug_logger.info(f"Connection URL: {safe_url}")

        engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_size=5,  # Connection pool size
            max_overflow=10,  # Maximum overflow connections
            echo=False,  # Set to True for SQL query logging
            connect_args={
                "connect_timeout": 10,  # 10 second connection timeout
            }
        )

        debug_logger.info("Testing database connection...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            debug_logger.info(f"Connection test successful: {result.scalar()}")

            # Check database version
            version_result = conn.execute(text("SELECT VERSION()"))
            version = version_result.scalar()
            debug_logger.info(f"MySQL version: {version}")

        # Create session factory
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
            "Please install PyMySQL: pip install pymysql cryptography"
        )
        raise
    except Exception as e:
        debug_logger.error(f"[ERROR] Database initialization failed: {type(e).__name__}: {e}")
        debug_logger.error("Please check:")
        debug_logger.error("  1. MySQL server is running")
        debug_logger.error("  2. Database credentials in .env file are correct")
        debug_logger.error("  3. Database exists (or will be created)")
        debug_logger.error("  4. User has proper permissions")
        raise


def create_tables():
    """Create all database tables"""
    global engine
    if engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    try:
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

    Returns:
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
