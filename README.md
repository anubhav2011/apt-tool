# AI Proctoring System

A production-ready, CPU-optimized AI proctoring solution built with FastAPI that provides real-time behavioral analysis for online examinations. The system leverages computer vision and machine learning to detect suspicious activities while maintaining performance and accuracy.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Project Structure](#project-structure)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

### Core Capabilities

- **Adaptive Calibration System**: Industry-standard calibration (6-10 seconds) with anti-cheat mechanisms
- **Multi-Modal Behavior Analysis**: Simultaneous tracking of gaze direction, head pose, and facial presence
- **Real-Time Video Processing**: CPU-optimized frame processing at 12 FPS for efficient resource utilization
- **Intelligent Scoring Engine**: Weighted confidence scoring with risk classification (Clean, Borderline, Suspicious, High Risk)
- **Comprehensive Reporting**: Detailed analytics with temporal analysis and evidence storage
- **RESTful API**: Clean, documented API endpoints for easy integration

### Technical Highlights

- **CPU-Optimized Performance**: Engineered for production deployment without GPU requirements
- **Adaptive Threshold System**: Dynamic threshold computation based on individual calibration baselines
- **Anti-Cheat Validation**: Multi-layer validation to prevent calibration manipulation
- **Frame Rejection Logic**: Intelligent filtering of unstable frames for accurate baseline computation
- **Extensible Architecture**: Clean separation of concerns with service-oriented design

## Architecture

The system follows a layered architecture pattern:

```
┌─────────────────────────────────────────┐
│          API Layer (FastAPI)            │
│  ┌───────────┐  ┌──────────────────┐   │
│  │ Endpoints │  │  Error Handling  │   │
│  └───────────┘  └──────────────────┘   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          Service Layer                  │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │ Proctoring  │  │   Processing    │  │
│  │   Service   │  │    Services     │  │
│  └─────────────┘  └─────────────────┘  │
│         ↓               ↓               │
│  ┌──────────┐  ┌──────────────────┐    │
│  │Calibration│  │Detection│Scoring│    │
│  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Repository Layer                  │
│  ┌─────────────────────────────────┐   │
│  │    Data Access & Persistence    │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         Database (MySQL)                │
│  ┌─────────────────────────────────┐   │
│  │  Proctoring Reports & Sessions  │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Prerequisites

- **Python**: 3.10 or higher
- **MySQL**: 8.0 or higher
- **Operating System**: Linux, macOS, or Windows
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: Multi-core processor recommended

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-proctoring-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Database

Create a MySQL database and configure the connection:

```sql
CREATE DATABASE proctoring CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=proctoring

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
```

### 6. Initialize Database Schema

The database tables will be created automatically on first run, or you can manually initialize:

```bash
python -c "from app.core.database import init_db; init_db()"
```

## Configuration

The system is highly configurable through `app/core/proctoring_config.py`. Key settings include:

### Video Processing

```python
MAX_FRAME_DIMENSION = 1280  # Frame resolution
TARGET_FPS = 12             # Processing frame rate
```

### Calibration Parameters

```python
CALIBRATION_DURATION_SEC = 8.0   # Calibration window
CALIBRATION_MIN_FRAMES = 80      # Minimum required frames
```

### Detection Thresholds

```python
MAX_BASELINE_YAW = 20.0      # degrees
MAX_BASELINE_PITCH = 15.0    # degrees
MAX_BASELINE_ROLL = 12.0     # degrees
MAX_BASELINE_EYE = 0.8       # radians (~46 degrees)
```

### Risk Classification

```python
RISK_CLEAN_MAX = 0.20        # Clean: 0-20%
RISK_BORDERLINE_MAX = 0.40   # Borderline: 20-40%
RISK_SUSPICIOUS_MAX = 0.65   # Suspicious: 40-65%
                              # High Risk: >65%
```

## Usage

### Starting the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Documentation

### Process Video Endpoint

**POST** `/api/v1/proctoring/process`

Process a proctoring video and generate a behavioral analysis report.

**Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/proctoring/process" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@exam_recording.mp4" \
  -F "student_id=STU123456" \
  -F "exam_id=EXAM001" \
  -F "calibration_duration=8.0"
```

**Response:**

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_id": "STU123456",
  "exam_id": "EXAM001",
  "video_duration_sec": 3600.0,
  "calibration_status": "SUCCESS",
  "risk_classification": "CLEAN",
  "confidence_score": 0.92,
  "alerts": {
    "gaze_deviation": 5,
    "head_movement": 3,
    "face_absence": 1,
    "multiple_faces": 0
  },
  "temporal_analysis": {
    "high_activity_periods": [
      {"start": 1200, "end": 1350, "alert_density": 0.08}
    ]
  },
  "created_at": "2025-12-18T10:30:00Z"
}
```

### Get Report Endpoint

**GET** `/api/v1/reports/{report_id}`

Retrieve a specific proctoring report.

**Response:**

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "student_id": "STU123456",
  "exam_id": "EXAM001",
  "video_duration_sec": 3600.0,
  "calibration_status": "SUCCESS",
  "baseline_metrics": {
    "eye_variance": 0.25,
    "yaw_variance": 8.5,
    "pitch_variance": 6.2,
    "roll_variance": 4.1
  },
  "adaptive_thresholds": {
    "eye": 0.80,
    "yaw": 21.6,
    "pitch": 14.68,
    "roll": 9.92
  },
  "alert_details": [...],
  "confidence_score": 0.92,
  "risk_classification": "CLEAN"
}
```

### List Reports Endpoint

**GET** `/api/v1/reports?student_id={student_id}&exam_id={exam_id}&skip=0&limit=10`

Retrieve multiple reports with optional filtering.

### Health Check Endpoint

**GET** `/api/v1/health`

Check API and database connectivity status.

## Database Schema

### proctoring_reports Table

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(36) | Primary key (UUID) |
| student_id | VARCHAR(100) | Student identifier |
| exam_id | VARCHAR(100) | Exam identifier |
| video_duration_sec | FLOAT | Total video length |
| calibration_status | VARCHAR(50) | SUCCESS/FAILED/SKIPPED |
| baseline_metrics | JSON | Calibration baseline data |
| adaptive_thresholds | JSON | Computed thresholds |
| alert_details | JSON | Array of alert objects |
| alert_counts | JSON | Aggregated alert counts |
| temporal_analysis | JSON | Time-based patterns |
| confidence_score | FLOAT | Overall confidence (0-1) |
| risk_classification | VARCHAR(50) | Risk level assessment |
| created_at | DATETIME | Record creation time |
| updated_at | DATETIME | Last modification time |

## Project Structure

```
ai-proctoring-system/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── proctoring.py      # Video processing endpoints
│   │       │   ├── reports.py         # Report management endpoints
│   │       │   └── health.py          # Health check endpoint
│   │       └── router.py              # API router configuration
│   ├── core/
│   │   ├── config.py                  # Application configuration
│   │   ├── database.py                # Database connection
│   │   ├── dependencies.py            # Dependency injection
│   │   ├── exceptions.py              # Custom exceptions
│   │   ├── middlewares.py             # Request middlewares
│   │   └── proctoring_config.py       # Proctoring parameters
│   ├── models/
│   │   └── proctoring.py              # SQLAlchemy models
│   ├── repositories/
│   │   ├── base_repository.py         # Base repository pattern
│   │   └── proctoring_repository.py   # Data access layer
│   ├── schemas/
│   │   └── models.py                  # Pydantic schemas
│   ├── services/
│   │   ├── proctoring_service.py      # Main business logic
│   │   └── proctoring_processing/
│   │       ├── calibration_service.py  # Calibration logic
│   │       ├── detection_service.py    # Alert detection
│   │       ├── scoring_service.py      # Confidence scoring
│   │       └── video_processing_service.py
│   └── utils/
│       └── logger.py                   # Logging configuration
├── debug_logs/                         # Application logs
├── main.py                             # Application entry point
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

## Development

### Running in Development Mode

```bash
# Enable auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Code Quality Tools

```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

### Running Tests

```bash
pytest tests/ -v
```

## Deployment

### Production Considerations

1. **Environment Variables**: Use production database credentials
2. **Process Manager**: Deploy with Gunicorn or similar WSGI server
3. **Reverse Proxy**: Use Nginx or Apache for SSL and load balancing
4. **Monitoring**: Implement logging and health check monitoring
5. **Scaling**: Consider horizontal scaling for high traffic

### Docker Deployment

```bash
# Build image
docker build -t ai-proctoring-system .

# Run container
docker run -d -p 8000:8000 \
  -e MYSQL_HOST=db_host \
  -e MYSQL_USER=db_user \
  -e MYSQL_PASSWORD=db_pass \
  ai-proctoring-system
```

### Gunicorn Production Server

```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --log-level info
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**Built with**: FastAPI, OpenCV, MediaPipe, SQLAlchemy, MySQL

**Maintained by**: Your Organization

**Documentation**: https://your-docs-url.com

**Support**: support@your-organization.com
