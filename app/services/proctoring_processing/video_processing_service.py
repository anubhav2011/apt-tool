"""
Video Processing Service
Handles video download, frame extraction, and processing orchestration
NO CALIBRATION - Uses industry-standard default thresholds immediately
"""
import cv2
import numpy as np
import tempfile
import os
from typing import Dict
import requests
from app.utils.logger import debug_logger

from app.core.proctoring_config import ProctoringConfig
from app.core.exceptions import VideoDownloadError, VideoProcessingError
from .base_proctoring_processing_service import IVideoProcessingService
from .detection_service import DetectionService


class VideoProcessingService(IVideoProcessingService):
    """
    Service for video processing pipeline
    Orchestrates frame extraction and analysis
    """

    def __init__(self, config: ProctoringConfig) -> None:
        """
        Initialize video processing service

        Args:
            config: Configuration object
        """
        self.config = config
        self._setup()

    def _setup(self):
        """Initialize child services"""
        self.detection_service = DetectionService(self.config)
        debug_logger.info("Video processing service initialized with industry-standard thresholds")

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to max dimension for CPU optimization

        Args:
            frame: Input video frame

        Returns:
            Resized frame
        """
        h, w = frame.shape[:2]
        max_dim = max(h, w)

        if max_dim > self.config.MAX_FRAME_DIMENSION:
            scale = self.config.MAX_FRAME_DIMENSION / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return frame

    def _download_video(self, video_url: str) -> str:
        """
        Download video from URL to temporary file

        Args:
            video_url: URL of the video

        Returns:
            Path to downloaded file

        Raises:
            VideoDownloadError: If download fails
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()

        try:
            debug_logger.info(f"Downloading video from: {video_url}")
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            debug_logger.info(f"Video downloaded successfully to: {temp_path}")
            return temp_path

        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            debug_logger.error(f"Failed to download video: {str(e)}")
            raise VideoDownloadError(f"Failed to download video: {str(e)}", 5010)

    def process_video(self, video_path: str, session_id: str) -> Dict:
        """
        Main video processing pipeline - NO CALIBRATION

        Args:
            video_path: Path or URL to video file
            session_id: Session identifier

        Returns:
            Complete proctoring report in new format

        Raises:
            VideoProcessingError: If video processing fails
        """
        import time
        start_time = time.time()
        video_path_to_process = video_path
        downloaded_file = None

        try:
            # Download if URL
            if video_path.startswith('http://') or video_path.startswith('https://'):
                downloaded_file = self._download_video(video_path)
                video_path_to_process = downloaded_file

            # Open video
            cap = cv2.VideoCapture(video_path_to_process)
            if not cap.isOpened():
                raise VideoProcessingError(f"Failed to open video: {video_path_to_process}", 5011)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            debug_logger.info(f"Video properties - FPS: {fps}, Frames: {total_frames}, Duration: {video_duration:.1f}s")

            frame_skip = max(1, int(fps / self.config.TARGET_FPS))
            debug_logger.info(f"Processing every {frame_skip} frame(s) to achieve ~{self.config.TARGET_FPS} FPS")

            thresholds = self.config.get_default_thresholds()
            debug_logger.info(
                f"Using industry-standard thresholds: "
                f"eye_h={thresholds['eye_horizontal']}°, eye_v={thresholds['eye_vertical']}°, "
                f"yaw={thresholds['yaw']}°, pitch={thresholds['pitch']}°"
            )

            # Processing state
            frame_idx = 0
            frames_processed = 0
            current_time = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames for efficiency
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                # Resize frame for CPU optimization
                frame = self._resize_frame(frame)
                current_time = frame_idx / fps

                # Detect features
                gaze_h, gaze_v, num_faces, bbox_center = self.detection_service.detect_gaze(frame)
                yaw, pitch, roll = self.detection_service.detect_head_pose(frame)

                if frames_processed % 50 == 0:
                    debug_logger.info(
                        f"Frame {frames_processed} ({current_time:.1f}s): "
                        f"gaze_h={gaze_h}, gaze_v={gaze_v}, yaw={yaw}, pitch={pitch}, num_faces={num_faces}"
                    )

                self.detection_service.update_violations(
                    current_time, gaze_h, gaze_v, yaw, pitch, roll, num_faces, thresholds
                )

                frames_processed += 1
                frame_idx += 1

                if frames_processed % 100 == 0:
                    debug_logger.info(f"Processed {frames_processed} frames ({current_time:.1f}s)")

            cap.release()

            self.detection_service.violation_tracker.finalize()

            # Processing metadata
            processing_time = time.time() - start_time

            report = self._generate_new_format_report(
                session_id,
                thresholds,
                processing_time,
                video_duration,
                frames_processed
            )

            debug_logger.info(
                f"Processing complete in {processing_time:.2f}s - "
                f"Total gesture types detected: {len(report['analysis']['gestures'])}"
            )
            return report

        except VideoDownloadError:
            raise
        except VideoProcessingError:
            raise
        except Exception as e:
            debug_logger.error(f"Error processing video: {str(e)}", exc_info=True)
            raise VideoProcessingError(f"Video processing failed: {str(e)}", 5012)

        finally:
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.unlink(downloaded_file)
                    debug_logger.info(f"Cleaned up downloaded file: {downloaded_file}")
                except Exception as e:
                    debug_logger.warning(f"Failed to cleanup downloaded file: {str(e)}")

    def _generate_new_format_report(
        self,
        session_id: str,
        thresholds: Dict[str, float],
        processing_time: float,
        video_duration: float,
        frames_processed: int
    ) -> Dict:
        """
        Generate report in new format matching apt_response.json

        Events are recorded at STARTING TIMESTAMP when threshold is crossed.
        Duration is how long the violation continues.
        No merging - each distinct direction change creates new event.

        Format:
        {
          "gestures": [
            {
              "name": "head_movement",
              "occurrence": [
                {"timestamp": "0:00", "duration": 16.5, "direction": "down", "intensity": "180 degrees"}
              ]
            }
          ]
        }
        """
        violation_events = self.detection_service.get_violation_events()

        debug_logger.info(f"Total violation events detected: {len(violation_events)}")

        event_counts = {}
        for event in violation_events:
            event_type = event['type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        debug_logger.info(f"Event breakdown: {event_counts}")

        gestures = []

        head_events = [e for e in violation_events if e['type'] in ['head_left', 'head_right', 'head_up', 'head_down']]
        head_events.sort(key=lambda x: x['timestamp'])

        head_occurrences = []
        if head_events:
            for event in head_events:
                direction = event['type'].replace('head_', '')
                head_occurrences.append({
                    "timestamp": self._format_timestamp(event['timestamp']),
                    "duration": round(event['duration'], 1),
                    "direction": direction,
                    "intensity": f"{round(event['intensity'], 0)} degrees"
                })

        if head_occurrences:
            debug_logger.info(f"Adding {len(head_occurrences)} head movement occurrences")
            gestures.append({
                "name": "head_movement",
                "occurrence": head_occurrences
            })

        gaze_events = [e for e in violation_events if e['type'] in ['gaze_left', 'gaze_right', 'gaze_up', 'gaze_down']]
        gaze_events.sort(key=lambda x: x['timestamp'])

        gaze_occurrences = []
        if gaze_events:
            for event in gaze_events:
                direction = event['type'].replace('gaze_', '')
                gaze_occurrences.append({
                    "timestamp": self._format_timestamp(event['timestamp']),
                    "duration": round(event['duration'], 1),
                    "direction": direction,
                    "intensity": f"{round(event['intensity'], 0)} degrees"
                })

        if gaze_occurrences:
            debug_logger.info(f"Adding {len(gaze_occurrences)} eye movement occurrences")
            gestures.append({
                "name": "Eye_movement",
                "occurrence": gaze_occurrences
            })

        face_missing_events = [e for e in violation_events if e['type'] == 'face_missing']
        face_missing_events.sort(key=lambda x: x['timestamp'])

        face_missing_occurrences = []
        if face_missing_events:
            for event in face_missing_events:
                face_missing_occurrences.append({
                    "timestamp": self._format_timestamp(event['timestamp']),
                    "duration": round(event['duration'], 1),
                    "direction": "",
                    "intensity": ""
                })

        if face_missing_occurrences:
            debug_logger.info(f"Adding {len(face_missing_occurrences)} face missing occurrences")
            gestures.append({
                "name": "face_missing",
                "occurrence": face_missing_occurrences
            })

        multiple_faces_events = [e for e in violation_events if e['type'] == 'multiple_faces']
        multiple_faces_events.sort(key=lambda x: x['timestamp'])

        multiple_faces_occurrences = []
        if multiple_faces_events:
            for event in multiple_faces_events:
                multiple_faces_occurrences.append({
                    "timestamp": self._format_timestamp(event['timestamp']),
                    "duration": round(event['duration'], 1),
                    "direction": "",
                    "intensity": ""
                })

        if multiple_faces_occurrences:
            debug_logger.info(f"Adding {len(multiple_faces_occurrences)} multiple faces occurrences")
            gestures.append({
                "name": "multiple_faces",
                "occurrence": multiple_faces_occurrences
            })

        return {
            "session_id": session_id,
            "status": "success",
            "message": "Video processed successfully",
            "analysis": {
                "gestures": gestures,
                "thresholds_used": thresholds,
                "processing_metadata": {
                    "processing_time_sec": round(processing_time),
                    "video_duration_sec": round(video_duration),
                    "frames_processed": frames_processed
                }
            }
        }

    def _format_timestamp(self, seconds: float) -> str:
        """
        Convert seconds to M:SS format (minutes:seconds)

        Examples:
        - 22.345 seconds -> "0:22"
        - 105.678 seconds -> "1:45"
        - 0.078 seconds -> "0:00"
        - 65.5 seconds -> "1:05"

        Args:
            seconds: Timestamp in seconds (starting time of the event)

        Returns:
            Formatted timestamp string
        """
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}:{remaining_seconds:02d}"

    def cleanup(self):
        """Release all resources"""
        self.detection_service.cleanup()
        debug_logger.info("Video processing service cleaned up")
