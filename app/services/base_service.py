"""
Base service interfaces
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict


class IProctoringService(ABC):
    """Interface for proctoring service"""

    @abstractmethod
    async def process_video_upload(self, video_file, session_id: Optional[str] = None,
                                   candidate_id: Optional[str] = None) -> Dict:
        """Process uploaded video file"""
        raise NotImplementedError

    @abstractmethod
    async def process_video_from_url(self, video_url: str, session_id: Optional[str] = None,
                                     candidate_id: Optional[str] = None) -> Dict:
        """Process video from URL"""
        raise NotImplementedError

    @abstractmethod
    def process_video_file(self, video_path: str, session_id: Optional[str] = None,
                           candidate_id: Optional[str] = None) -> Dict:
        """Process video file and return report"""
        raise NotImplementedError

    @abstractmethod
    def get_report(self, session_id: str) -> Optional[Dict]:
        """Get report by session ID"""
        raise NotImplementedError

    @abstractmethod
    def get_candidate_reports(self, candidate_id: str, limit: int = 10) -> Dict:
        """Get all reports for a candidate"""
        raise NotImplementedError

    @abstractmethod
    def delete_report(self, session_id: str) -> Dict:
        """Delete a report"""
        raise NotImplementedError
