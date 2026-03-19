

"""
Video Processing Service
Handles video download, frame extraction, and processing orchestration.

Changes vs previous version
----------------------------
* BUG-5 FIX : "Eye_movement" → "eye_movement" (consistent snake_case)
* BUG-2 FIX (companion): _get_velocity_label now always returns a non-empty
  string — matching the fix inside ViolationTracker._get_velocity_label.
  Both copies are kept in sync deliberately so the report layer never
  produces empty velocity fields.
* Improved logging: per-frame data now logged every 30 frames (was 50) for
  easier debugging of suspicious-event windows.
"""
import cv2
import numpy as np
import tempfile
import os
import queue
import threading
from typing import Dict, List
import requests
from app.utils.logger import debug_logger

from app.core.config import ProctoringConfig
from app.core.exceptions import VideoDownloadError, VideoProcessingError
from .base_proctoring_processing_service import IVideoProcessingService
from .detection_service import DetectionService

# Sentinel for end of stream in parallel pipeline
_SENTINEL = object()


class VideoProcessingService(IVideoProcessingService):
    """
    Service for video processing pipeline.
    Orchestrates frame extraction and analysis.
    """

    def __init__(self, config: ProctoringConfig) -> None:
        self.config = config
        self._setup()

    def _setup(self) -> None:
        self.detection_service = DetectionService(self.config)
        debug_logger.info("VideoProcessingService initialised")

    # ------------------------------------------------------------------
    # Frame resize
    # ------------------------------------------------------------------

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w    = frame.shape[:2]
        max_dim = max(h, w)
        if max_dim > self.config.MAX_FRAME_DIMENSION:
            scale = self.config.MAX_FRAME_DIMENSION / max_dim
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return frame

    # ------------------------------------------------------------------
    # Parallel pipeline (3 threads: reader, analyzer, aggregator)
    # ------------------------------------------------------------------

    def _run_parallel_pipeline(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        frame_skip: int,
        thresholds: Dict[str, float],
        warmup: float,
    ) -> int:
        """Run frame reader, analyzer, and aggregator in parallel. Returns frames_processed."""
        frame_queue: queue.Queue = queue.Queue(maxsize=60)
        results_queue: queue.Queue = queue.Queue(maxsize=60)
        frames_processed = [0]  # mutable so aggregator can update

        def reader() -> None:
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue
                    current_time = frame_idx / fps
                    frame = self._resize_frame(frame)
                    frame_queue.put((frame_idx, current_time, frame))
                    frame_idx += 1
            finally:
                frame_queue.put(_SENTINEL)

        def analyzer() -> None:
            enable_tvt = getattr(self.config, 'ENABLE_TVT', False)
            while True:
                try:
                    item = frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    results_queue.put(_SENTINEL)
                    return
                frame_idx, current_time, frame = item
                gaze_h, gaze_v, num_faces, bbox_center, gaze_confidence, occlusion_ratio = \
                    self.detection_service.detect_gaze(frame)
                yaw, pitch, roll, head_confidence = \
                    self.detection_service.detect_head_pose(frame)
                landmark_vector = None
                if enable_tvt:
                    landmark_vector = self.detection_service.get_landmark_vector(frame)
                results_queue.put((frame_idx, {
                    'timestamp': current_time,
                    'gaze_h': gaze_h, 'gaze_v': gaze_v,
                    'yaw': yaw, 'pitch': pitch, 'roll': roll,
                    'num_faces': num_faces,
                    'gaze_confidence': gaze_confidence,
                    'head_confidence': head_confidence,
                    'occlusion_ratio': occlusion_ratio,
                    'landmark_vector': landmark_vector,
                }))

        def aggregator() -> None:
            buffer: Dict[int, Dict] = {}
            next_expected = 0
            count = 0
            while True:
                try:
                    item = results_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                if item is _SENTINEL:
                    break
                frame_idx, result = item
                buffer[frame_idx] = result
                while next_expected in buffer:
                    r = buffer.pop(next_expected)
                    if r['timestamp'] >= warmup:
                        self.detection_service.update_violations(
                            r['timestamp'],
                            r['gaze_h'], r['gaze_v'],
                            r['yaw'], r['pitch'], r['roll'],
                            r['num_faces'], thresholds,
                            r['gaze_confidence'], r['head_confidence'], r['occlusion_ratio'],
                            landmark_vector=r.get('landmark_vector'),
                        )
                    count += 1
                    next_expected += 1
            frames_processed[0] = count

        t_reader = threading.Thread(target=reader)
        t_analyzer = threading.Thread(target=analyzer)
        t_aggregator = threading.Thread(target=aggregator)
        t_reader.start()
        t_analyzer.start()
        t_aggregator.start()
        t_reader.join()
        t_analyzer.join()
        t_aggregator.join()
        return frames_processed[0]

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    def _download_video(self, video_url: str) -> str:
        """Download video from URL into a temporary file."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = tmp.name
        tmp.close()
        try:
            debug_logger.info(f"Downloading video: {video_url}")
            resp = requests.get(video_url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            debug_logger.info(f"Download complete → {temp_path}")
            return temp_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise VideoDownloadError(f"Failed to download video: {e}", 5010)

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, session_id: str) -> Dict:
        """
        Main video processing pipeline.

        Parameters
        ----------
        video_path : str
            Local file path or HTTP(S) URL.
        session_id : str
            Unique identifier for this session.

        Returns
        -------
        dict
            Complete proctoring report.
        """
        import time
        start_time        = time.time()
        downloaded_file   = None
        video_path_to_use = video_path

        try:
            if video_path.startswith('http://') or video_path.startswith('https://'):
                downloaded_file   = self._download_video(video_path)
                video_path_to_use = downloaded_file

            cap = cv2.VideoCapture(video_path_to_use)
            if not cap.isOpened():
                raise VideoProcessingError(
                    f"Cannot open video: {video_path_to_use}", 5011
                )

            fps           = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps if fps > 0 else 0.0

            debug_logger.info(
                f"Video → FPS:{fps:.1f} | Frames:{total_frames} | "
                f"Duration:{video_duration:.1f}s"
            )

            frame_skip = max(1, int(fps / self.config.TARGET_FPS))
            debug_logger.info(
                f"Sampling every {frame_skip} frame(s) "
                f"≈ {self.config.TARGET_FPS} FPS"
            )

            thresholds = self.config.get_default_thresholds()
            debug_logger.info(
                f"Thresholds → eye_h={thresholds['eye_horizontal']}° "
                f"eye_v={thresholds['eye_vertical']}° "
                f"yaw={thresholds['yaw']}° "
                f"pitch={thresholds['pitch']}°"
            )

            warmup = getattr(self.config, 'WARMUP_SECONDS', 0.0)
            frame_idx        = 0
            frames_processed = 0


            if getattr(self.config, 'ENABLE_PARALLEL_PROCESSING', False):
                debug_logger.info("Using parallel 3-thread pipeline")
                frames_processed = self._run_parallel_pipeline(
                    cap, fps, frame_skip, thresholds, warmup,
                )
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue

                    frame        = self._resize_frame(frame)
                    current_time = frame_idx / fps

                    (gaze_h, gaze_v, num_faces, bbox_center,
                     gaze_confidence, occlusion_ratio) = \
                        self.detection_service.detect_gaze(frame)

                    yaw, pitch, roll, head_confidence = \
                        self.detection_service.detect_head_pose(frame)

                    # Optional: landmark vector for TVT
                    landmark_vector = None
                    if getattr(self.config, 'ENABLE_TVT', False):
                        landmark_vector = self.detection_service.get_landmark_vector(frame)

                    # Log every 30 processed frames for debugging suspicious windows
                    if frames_processed % 30 == 0:
                        debug_logger.info(
                            f"[{current_time:.1f}s] "
                            f"gaze=({gaze_h},{gaze_v}) "
                            f"head=({yaw},{pitch}) "
                            f"faces={num_faces} "
                            f"g_conf={gaze_confidence:.2f} "
                            f"h_conf={head_confidence:.2f} "
                            f"occ={occlusion_ratio:.2f}"
                        )

                    if current_time >= warmup:
                        self.detection_service.update_violations(
                            current_time, gaze_h, gaze_v,
                            yaw, pitch, roll,
                            num_faces, thresholds,
                            gaze_confidence, head_confidence, occlusion_ratio,
                            landmark_vector=landmark_vector,
                        )

                    frames_processed += 1
                    frame_idx        += 1

            cap.release()
            self.detection_service.violation_tracker.finalize()

            processing_time = time.time() - start_time

            report = self._generate_report(
                session_id, thresholds,
                processing_time, video_duration, frames_processed,
            )

            debug_logger.info(
                f"Done in {processing_time:.2f}s — "
                f"{len(report['analysis']['gestures'])} gesture types"
            )
            return report

        except (VideoDownloadError, VideoProcessingError):
            raise
        except Exception as e:
            debug_logger.error(f"Error processing video: {e}", exc_info=True)
            raise VideoProcessingError(f"Video processing failed: {e}", 5012)

        finally:
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.unlink(downloaded_file)
                except Exception as e:
                    debug_logger.warning(f"Failed to delete temp file: {e}")

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        session_id: str,
        thresholds: Dict,
        processing_time: float,
        video_duration: float,
        frames_processed: int,
    ) -> Dict:
        """
        Build the final gestures report from all recorded violation events.

        BUG-5 FIX: gesture name changed from "Eye_movement" → "eye_movement".
        """
        events = self.detection_service.get_violation_events()
        debug_logger.info(f"Total violation events: {len(events)}")

        # Tally by type for debug logging
        tally: Dict[str, int] = {}
        for ev in events:
            tally[ev['type']] = tally.get(ev['type'], 0) + 1
        debug_logger.info(f"Event breakdown: {tally}")

        head_events          = [e for e in events if e['type'].startswith('head_')]
        eye_events           = [e for e in events if e['type'].startswith('gaze_')]
        face_missing_events  = [e for e in events if e['type'] == 'face_missing']
        multi_face_events    = [e for e in events if e['type'] == 'multiple_faces']
        face_occluded_events = [e for e in events if e['type'] == 'face_occluded']

        # Apply high-risk filter when enabled (only suspicious/high-risk in output)
        report_only_high_risk = getattr(self.config, 'REPORT_ONLY_HIGH_RISK', False)
        if report_only_high_risk:
            head_events = [e for e in head_events if self._is_high_risk_head_event(e)]
            eye_events  = [e for e in eye_events if self._is_high_risk_eye_event(e)]
            debug_logger.info(
                f"After high-risk filter → head: {len(head_events)}, eye: {len(eye_events)}"
            )

        gestures: List[Dict] = []

        # ── Head movement ──────────────────────────────────────────────────
        if head_events:
            debug_logger.info(f"Head events: {len(head_events)}")
            gestures.append({
                "name": "head_movement",
                "occurrence": [
                    {
                        "timestamp":  self._fmt_ts(e['timestamp']),
                        "duration":   round(e['duration'], 1),
                        "direction":  e['type'].replace('head_', ''),
                        "intensity":  f"{round(e['intensity'], 1)} degrees",
                        "confidence": round(e.get('confidence', 0.0), 2),
                        "velocity":   self._get_velocity_label(e.get('velocity', 0.0)),
                    }
                    for e in head_events
                ],
            })

        # ── Eye / gaze movement ────────────────────────────────────────────
        # BUG-5 FIX: "Eye_movement" → "eye_movement"
        if eye_events:
            debug_logger.info(f"Eye events: {len(eye_events)}")
            gestures.append({
                "name": "eye_movement",          # ← was "Eye_movement"
                "occurrence": [
                    {
                        "timestamp":  self._fmt_ts(e['timestamp']),
                        "duration":   round(e['duration'], 1),
                        "direction":  e['type'].replace('gaze_', ''),
                        "intensity":  f"{round(e['intensity'], 1)} degrees",
                        "confidence": round(e.get('confidence', 0.0), 2),
                        "velocity":   self._get_velocity_label(e.get('velocity', 0.0)),
                    }
                    for e in eye_events
                ],
            })

        # ── Face missing ───────────────────────────────────────────────────
        if face_missing_events:
            debug_logger.info(f"Face missing events: {len(face_missing_events)}")
            gestures.append({
                "name": "face_missing",
                "occurrence": [
                    {
                        "timestamp":  self._fmt_ts(e['timestamp']),
                        "duration":   round(e['duration'], 1),
                        "direction":  "",
                        "intensity":  "",
                        "confidence": round(e.get('confidence', 1.0), 2),
                        "velocity":   "",
                    }
                    for e in face_missing_events
                ],
            })

        # ── Multiple faces ─────────────────────────────────────────────────
        if multi_face_events:
            debug_logger.info(f"Multiple faces events: {len(multi_face_events)}")
            gestures.append({
                "name": "multiple_faces",
                "occurrence": [
                    {
                        "timestamp":  self._fmt_ts(e['timestamp']),
                        "duration":   round(e['duration'], 1),
                        "direction":  "",
                        "intensity":  "",
                        "confidence": round(e.get('confidence', 0.9), 2),
                        "velocity":   "",
                    }
                    for e in multi_face_events
                ],
            })

        # ── Face occluded ──────────────────────────────────────────────────
        if face_occluded_events:
            debug_logger.info(f"Face occluded events: {len(face_occluded_events)}")
            gestures.append({
                "name": "face_occluded",
                "occurrence": [
                    {
                        "timestamp":  self._fmt_ts(e['timestamp']),
                        "duration":   round(e['duration'], 1),
                        "direction":  "",
                        "intensity":  f"{round(e['intensity'], 1)}% occluded",
                        "confidence": round(e.get('confidence', 0.85), 2),
                        "velocity":   "",
                    }
                    for e in face_occluded_events
                ],
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
                    "video_duration_sec":  round(video_duration),
                    "frames_processed":    frames_processed,
                },
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_high_risk_head_event(self, e: Dict) -> bool:
        """True if head event is suspicious/high-risk. Uses lower intensity bar for head up/down (pitch)."""
        intensity = float(e.get('intensity', 0))
        duration  = float(e.get('duration', 0))
        velocity  = float(e.get('velocity', 0))
        etype     = e.get('type', '')
        # Head up/down (pitch): use lower bar so moderate up/down movements are reported
        if etype in ('head_up', 'head_down'):
            min_int = getattr(self.config, 'MIN_HEAD_INTENSITY_HIGH_RISK_PITCH', 26.0)
        else:
            min_int = getattr(self.config, 'MIN_HEAD_INTENSITY_HIGH_RISK', 40.0)
        min_dur   = getattr(self.config, 'MIN_HEAD_DURATION_HIGH_RISK', 0.8)
        fast      = getattr(self.config, 'SUSPICIOUS_VELOCITY_THRESHOLD', 25.0)
        return intensity >= min_int or (duration >= min_dur and velocity >= fast)

    def _is_high_risk_eye_event(self, e: Dict) -> bool:
        """True if eye event is suspicious/high-risk. Uses same bar for all gaze directions (incl. up/down)."""
        intensity = float(e.get('intensity', 0))
        duration  = float(e.get('duration', 0))
        velocity  = float(e.get('velocity', 0))
        min_int   = getattr(self.config, 'MIN_EYE_INTENSITY_HIGH_RISK', 10.0)
        fast      = getattr(self.config, 'SUSPICIOUS_VELOCITY_THRESHOLD', 25.0)
        return intensity >= min_int or (duration >= 0.8 and velocity >= fast)

    def _get_velocity_label(self, velocity: float) -> str:
        """
        Convert velocity (°/s) to human-readable label.

        BUG-2 FIX: always returns a non-empty string — mirrors the fix in
        ViolationTracker._get_velocity_label.
        """
        if velocity < self.config.MIN_VELOCITY_THRESHOLD:
            return "negligible"
        elif velocity < self.config.SUSPICIOUS_VELOCITY_THRESHOLD:
            return "slow"
        elif velocity < self.config.HIGH_RISK_VELOCITY_THRESHOLD:
            return "moderate"
        else:
            return "rapid"

    def _fmt_ts(self, seconds: float) -> str:
        """Convert seconds to M:SS timestamp string."""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.detection_service.cleanup()
        debug_logger.info("VideoProcessingService cleaned up")
