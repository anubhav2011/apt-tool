"""
Report Management Endpoints
Request/response handling only - all logic in services
"""
from fastapi import APIRouter, HTTPException, Depends

from app.services.base_service import IProctoringService
from app.core.dependencies import get_proctoring_service
from app.core.exceptions import ReportNotFoundError, DatabaseError

router = APIRouter()


@router.get("/{session_id}")
async def get_report(
    session_id: str,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Retrieve proctoring report by session ID
    """
    try:
        result = proctoring_service.get_report(session_id)
        return result

    except ReportNotFoundError as e:
        raise HTTPException(status_code=404, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })


@router.get("/candidate/{candidate_id}")
async def get_candidate_reports(
    candidate_id: str,
    limit: int = 10,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Retrieve all proctoring reports for a candidate
    """
    try:
        result = proctoring_service.get_candidate_reports(candidate_id, limit)
        return result

    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })


@router.delete("/{session_id}")
async def delete_report(
    session_id: str,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Delete proctoring report (GDPR compliance)
    """
    try:
        result = proctoring_service.delete_report(session_id)
        return result

    except ReportNotFoundError as e:
        raise HTTPException(status_code=404, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })
