"""
Proctoring Endpoints
Request/response handling only - all logic in services
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional

from app.schemas.models import ProcessVideoRequest
from app.services.base_service import IProctoringService
from app.core.dependencies import get_proctoring_service
from app.core.exceptions import VideoProcessingError, DatabaseError, ValidationError

router = APIRouter()


@router.post("/process-video")
async def process_video_file(
    video: UploadFile = File(...),
    interview_id: Optional[str] = Form(None),
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Process uploaded video file for proctoring analysis

    Args:
        video: Video file to process
        interview_id: Interview ID (UUID). If not provided, a new interview will be created.

    Returns:
        Analysis result with risk score and alerts
    """
    try:
        result = await proctoring_service.process_video_upload(
            video_file=video,
            interview_id=interview_id
        )
        return result

    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"response": e.message, "code": e.code})
    except VideoProcessingError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })


@router.post("/process-video-url")
async def process_video_url(
    request: ProcessVideoRequest,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Process video from URL for proctoring analysis

    Args:
        request: Request containing video_url and interview_id

    Returns:
        Analysis result with risk score and alerts
    """
    try:
        result = await proctoring_service.process_video_from_url(
            video_url=request.video_url,
            interview_id=request.interview_id
        )
        return result

    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"response": e.message, "code": e.code})
    except VideoProcessingError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })
