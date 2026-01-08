"""
FastAPI Backend for AI Proctoring System
Production-ready, CPU-optimized, enterprise-grade
Restructured with clean architecture
"""
import sys
from pathlib import Path

# Add parent directory to path so 'app' module can be imported correctly
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.utils.logger import debug_logger
from app.core.proctoring_config import ProctoringConfig
from app.core.database import init_database, create_tables
from app.api.v1.router import api_router

config = ProctoringConfig()

# --------------------------------------------------
# Lifespan Handler (Startup + Shutdown)
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # -------- Startup --------
    debug_logger.info("AI Proctoring System v2.0 starting up...")
    debug_logger.info(
        f"CPU-optimized | FPS={config.TARGET_FPS} | "
        f"Max Frame={config.MAX_FRAME_DIMENSION}px"
    )

    try:
        init_database(config)
        create_tables()
        debug_logger.info("Database initialized with SQLAlchemy ORM")
    except Exception as e:
        debug_logger.exception("Database initialization failed")
        raise

    yield  # Application runs here

    # -------- Shutdown --------
    debug_logger.info("AI Proctoring System shutting down...")
    # Optional: close DB pools, cleanup temp files, etc.

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="AI Proctoring System API",
    description="CPU-optimized video proctoring with count-based detection and database storage",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --------------------------------------------------
# Middleware
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Routes
# --------------------------------------------------
app.include_router(api_router, prefix="/api/v1")

# --------------------------------------------------
# Uvicorn Entry
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
