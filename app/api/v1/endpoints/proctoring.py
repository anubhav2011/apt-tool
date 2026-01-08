"""
Proctoring Endpoints
Request/response handling only - all logic in services
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional

from app.schemas.models import ProcessVideoRequest, ProcessVideoResponse
from app.services.base_service import IProctoringService
from app.core.dependencies import get_proctoring_service
from app.core.exceptions import VideoProcessingError, DatabaseError, ValidationError

router = APIRouter()


@router.post("/process-video")
async def process_video_file(
    video: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    candidate_id: Optional[str] = Form(None),
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Process uploaded video file for proctoring analysis
    Returns analysis with gestures array
    """
    try:
        result = await proctoring_service.process_video_upload(
            video_file=video,
            session_id=session_id,
            candidate_id=candidate_id
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
    """
    try:
        result = await proctoring_service.process_video_from_url(
            video_url=request.video_url,
            session_id=request.session_id,
            candidate_id=request.candidate_id
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
