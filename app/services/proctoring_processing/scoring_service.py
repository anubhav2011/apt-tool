"""
Scoring Service - CURRENTLY NOT IN USE

This service was designed for adaptive risk scoring and calibration-based detection.
Since the system now uses fixed industry-standard thresholds with direct detection,
this scoring logic is not needed.

Preserved for potential future features:
- Adaptive calibration per user
- Risk score calculation (0-100)
- Weighted violation analysis
- Alert severity classification

To re-enable: Uncomment code below and add import in __init__.py
"""

# from typing import List, Dict
# import numpy as np
#
# from .base_proctoring_processing_service import BaseService
# from app.schemas.models import (
#     Alert,
#     AlertType,
#     AlertSeverity,
#     ConfidenceScores,
#     RiskLevel,
#     ViolationCounts,
#     AdaptiveThresholds,
#     Baseline,
# )
#
#
# class ScoringService(BaseService):
#     """
#     Service for risk scoring and report generation
#     NOTE: Currently not in use - preserved for future implementation
#     """
#
#     def _setup(self) -> None:
#         """No additional setup needed"""
#         pass
#
#     # --------------------------------------------------
#     # STATIC HELPERS (NO SELF USAGE)
#     # --------------------------------------------------
#     @staticmethod
#     def _normalize_count(count: int, total_frames: int) -> float:
#         """
#         Normalize count to 0-1 range
#         """
#         if total_frames <= 0 or count <= 0:
#             return 0.0
#         return min(1.0, count * 0.08)
#
#     @staticmethod
#     def _compute_intensity_factor(
#         max_intensity: float, threshold: float
#     ) -> float:
#         """
#         Compute intensity factor for risk calculation
#         """
#         if threshold <= 0:
#             return 1.0
#         if max_intensity <= threshold:
#             return 1.0
#
#         excess_ratio = max_intensity / threshold
#         intensity = 1.0 + min(1.0, (excess_ratio - 1.0) * 0.6)
#         return round(float(intensity), 2)
#
#     @staticmethod
#     def _determine_severity(count: int) -> AlertSeverity:
#         """
#         Determine alert severity based on count
#         """
#         if count >= 3:
#             return AlertSeverity.HIGH_RISK
#         elif count >= 1:
#             return AlertSeverity.SUSPICIOUS
#         return AlertSeverity.MINOR
#
#     @staticmethod
#     def _create_alert(
#         violation_type: AlertType,
#         total_count: int,
#         severity: AlertSeverity,
#         timestamps: List[float],
#     ) -> Alert:
#         """
#         Create alert object
#         """
#         return Alert(
#             type=violation_type,
#             total_occurrences=total_count,
#             severity=severity,
#             timestamps=timestamps,
#         )
#
#     # --------------------------------------------------
#     # RISK COMPUTATION (USES CONFIG → INSTANCE METHOD)
#     # --------------------------------------------------
#     def _compute_risk_scores(
#         self,
#         counts: ViolationCounts,
#         max_intensities: Dict[str, float],
#         thresholds: AdaptiveThresholds,
#         total_frames: int,
#     ) -> ConfidenceScores:
#         """
#         Compute risk scores for all categories
#         """
#
#         # -------- GAZE RISK --------
#         total_gaze = counts.gaze_left + counts.gaze_right
#         normalized_gaze = self._normalize_count(total_gaze, total_frames)
#
#         gaze_intensity = max(
#             self._compute_intensity_factor(
#                 max_intensities.get("gaze_left", 0.0), thresholds.eye
#             ),
#             self._compute_intensity_factor(
#                 max_intensities.get("gaze_right", 0.0), thresholds.eye
#             ),
#         )
#
#         eye_risk = min(1.0, normalized_gaze * gaze_intensity)
#
#         # -------- HEAD RISK --------
#         total_head = (
#             counts.head_left
#             + counts.head_right
#             + counts.head_up
#             + counts.head_down
#         )
#
#         normalized_head = self._normalize_count(total_head, total_frames)
#
#         head_intensity = max(
#             self._compute_intensity_factor(
#                 max_intensities.get("head_left", 0.0), thresholds.yaw
#             ),
#             self._compute_intensity_factor(
#                 max_intensities.get("head_right", 0.0), thresholds.yaw
#             ),
#             self._compute_intensity_factor(
#                 max_intensities.get("head_up", 0.0), thresholds.pitch
#             ),
#             self._compute_intensity_factor(
#                 max_intensities.get("head_down", 0.0), thresholds.pitch
#             ),
#         )
#
#         head_risk = min(1.0, normalized_head * head_intensity)
#
#         # -------- FACE RISK --------
#         total_face = counts.face_missing + counts.multiple_faces
#         face_risk = self._normalize_count(total_face, total_frames)
#
#         # -------- FINAL SCORE --------
#         final_score = (
#             self.config.WEIGHT_GAZE * eye_risk
#             + self.config.WEIGHT_HEAD * head_risk
#             + self.config.WEIGHT_FACE * face_risk
#         )
#
#         return ConfidenceScores(
#             eye_risk=round(eye_risk, 3),
#             head_risk=round(head_risk, 3),
#             face_risk=round(face_risk, 3),
#             final_suspicion_score=round(float(final_score), 3),
#         )
#
#     # --------------------------------------------------
#     # RISK CLASSIFICATION
#     # --------------------------------------------------
#     def _classify_risk_level(self, risk_score: int) -> RiskLevel:
#         """
#         Risk Level Classifications (from PDF):
#         - LOW: 0-30 (No action required)
#         - MODERATE: 31-60 (Monitor for patterns)
#         - SUSPICIOUS: 61-80 (Review footage recommended)
#         - HIGH RISK: 81-100 (Manual review required)
#         """
#         if risk_score > self.config.RISK_SUSPICIOUS_MAX:
#             return RiskLevel.HIGH_RISK
#         elif risk_score > self.config.RISK_MODERATE_MAX:
#             return RiskLevel.SUSPICIOUS
#         elif risk_score > self.config.RISK_LOW_MAX:
#             return RiskLevel.MODERATE
#         return RiskLevel.CLEAN
#
#     # --------------------------------------------------
#     # REPORT GENERATION
#     # --------------------------------------------------
#     def generate_report(
#         self,
#         session_id: str,
#         calibration_result: Dict,
#         counts: Dict[str, int],
#         timestamps_dict: Dict[str, List[float]],
#         max_intensities: Dict[str, float],
#         processing_metadata: Dict,
#     ) -> Dict:
#         """
#         Generate complete proctoring report following PDF specification
#         """
#
#         baseline = Baseline(**calibration_result.get("baseline", {}))
#         thresholds_dict = calibration_result.get(
#             "adaptive_thresholds", self.config.get_default_thresholds()
#         )
#         thresholds = AdaptiveThresholds(**thresholds_dict)
#
#         calibration = {
#             "status": calibration_result.get("status", "FAILED"),
#             "yaw_threshold": thresholds.yaw,
#             "pitch_threshold": thresholds.pitch,
#             "eye_threshold": thresholds.eye,
#         }
#
#         violation_counts = ViolationCounts(**counts)
#
#         # Compute scores using PDF formulas
#         count_score = self._compute_count_score(violation_counts)
#         duration_info = self._compute_duration_score(timestamps_dict, thresholds, baseline)
#         intensity_score = self._compute_intensity_score(max_intensities, thresholds, baseline)
#
#         # Final risk score (0-100)
#         risk_score = self._compute_final_risk_score(
#             count_score,
#             duration_info["total_duration_score"],
#             intensity_score
#         )
#
#         risk_level = self._classify_risk_level(risk_score)
#
#         # Build alerts
#         alerts: List[Alert] = []
#
#         # Gaze alerts
#         gaze_total = counts["gaze_left"] + counts["gaze_right"]
#         if gaze_total > 0:
#             severity = self._determine_severity(gaze_total)
#             timestamps = sorted(
#                 set(
#                     timestamps_dict.get("gaze_left", [])
#                     + timestamps_dict.get("gaze_right", [])
#                 )
#             )
#             alerts.append(
#                 self._create_alert(
#                     AlertType.GAZE_AWAY, gaze_total, severity, timestamps
#                 )
#             )
#
#         # Head alerts
#         head_total = (
#             counts["head_left"]
#             + counts["head_right"]
#             + counts["head_up"]
#             + counts["head_down"]
#         )
#         if head_total > 0:
#             severity = self._determine_severity(head_total)
#             timestamps = sorted(
#                 set(
#                     timestamps_dict.get("head_left", [])
#                     + timestamps_dict.get("head_right", [])
#                     + timestamps_dict.get("head_up", [])
#                     + timestamps_dict.get("head_down", [])
#                 )
#             )
#             alerts.append(
#                 self._create_alert(
#                     AlertType.HEAD_MOVEMENT, head_total, severity, timestamps
#                 )
#             )
#
#         # Face alerts
#         if counts["face_missing"] > 0:
#             alerts.append(
#                 self._create_alert(
#                     AlertType.FACE_MISSING,
#                     counts["face_missing"],
#                     self._determine_severity(counts["face_missing"]),
#                     timestamps_dict.get("face_missing", []),
#                 )
#             )
#
#         if counts["multiple_faces"] > 0:
#             alerts.append(
#                 self._create_alert(
#                     AlertType.MULTIPLE_FACES,
#                     counts["multiple_faces"],
#                     self._determine_severity(counts["multiple_faces"]),
#                     timestamps_dict.get("multiple_faces", []),
#                 )
#             )
#
#         self.log_info(
#             f"Report generated | Risk Level={risk_level.value} | Risk Score={risk_score}/100"
#         )
#
#         return {
#             "session_id": session_id,
#             "calibration": calibration,
#             "counts": violation_counts.model_dump(),
#             "duration_sec": {
#                 "mild": duration_info["mild_sec"],
#                 "suspicious": duration_info["suspicious_sec"],
#                 "high_risk": duration_info["high_risk_sec"]
#             },
#             "intensity": {
#                 "avg_head_angle": round(max(
#                     max_intensities.get("head_left", 0.0),
#                     max_intensities.get("head_right", 0.0),
#                     max_intensities.get("head_up", 0.0),
#                     max_intensities.get("head_down", 0.0)
#                 ), 1),
#                 "avg_eye_angle": round(max(
#                     max_intensities.get("gaze_left", 0.0),
#                     max_intensities.get("gaze_right", 0.0)
#                 ), 1)
#             },
#             "alerts": [alert.model_dump() for alert in alerts],
#             "risk_score": risk_score,
#             "risk_level": risk_level.value,
#         }
#
#     # --------------------------------------------------
#     # PDF SPECIFIC METHODS
#     # --------------------------------------------------
#     def _compute_count_score(self, counts: ViolationCounts) -> float:
#         """
#         count_score = COUNT * COUNT_WEIGHT
#
#         Event Weights (from PDF):
#         - Mild head movement: 0.6 (Low severity)
#         - Major head movement: 1.3 (Medium severity)
#         - Eye-gaze deviation: 1.1 (Medium severity)
#         - Face missing: 4.0 (High severity)
#         - Multiple persons: 6.0 (Critical severity)
#         """
#         count_score = 0.0
#
#         # Eye gaze deviations (1.1 weight - Medium)
#         gaze_total = counts.gaze_left + counts.gaze_right
#         count_score += gaze_total * self.config.COUNT_WEIGHT_EYE_GAZE
#
#         # Classify head movements as mild or major based on count
#         head_left_score = counts.head_left * (
#             self.config.COUNT_WEIGHT_MAJOR_HEAD if counts.head_left > 10
#             else self.config.COUNT_WEIGHT_MILD_HEAD
#         )
#         head_right_score = counts.head_right * (
#             self.config.COUNT_WEIGHT_MAJOR_HEAD if counts.head_right > 10
#             else self.config.COUNT_WEIGHT_MILD_HEAD
#         )
#         head_up_score = counts.head_up * (
#             self.config.COUNT_WEIGHT_MAJOR_HEAD if counts.head_up > 10
#             else self.config.COUNT_WEIGHT_MILD_HEAD
#         )
#         head_down_score = counts.head_down * (
#             self.config.COUNT_WEIGHT_MAJOR_HEAD if counts.head_down > 10
#             else self.config.COUNT_WEIGHT_MILD_HEAD
#         )
#
#         count_score += head_left_score + head_right_score + head_up_score + head_down_score
#
#         # Face missing (4.0 weight - High severity)
#         count_score += counts.face_missing * self.config.COUNT_WEIGHT_FACE_MISSING
#
#         # Multiple faces (6.0 weight - Critical severity)
#         count_score += counts.multiple_faces * self.config.COUNT_WEIGHT_MULTIPLE_FACES
#
#         return round(float(count_score), 2)
#
#     def _compute_duration_score(
#         self,
#         timestamps_dict: Dict[str, List[float]],
#         thresholds: AdaptiveThresholds,
#         baseline: Baseline
#     ) -> Dict[str, float]:
#         """
#         duration_score = TOTAL_DURATION_SEC * DURATION_WEIGHT
#
#         Category Weights (from PDF):
#         - Mild: 0.2 (Minimal impact on score)
#         - Suspicious: 0.5 (Moderate impact on score)
#         - High Risk: 1.0 (Significant impact on score)
#
#         Returns dict with mild_sec, suspicious_sec, high_risk_sec, and total_duration_score
#         """
#         mild_duration = 0.0
#         suspicious_duration = 0.0
#         high_risk_duration = 0.0
#
#         # Calculate duration for each violation type
#         for violation_type, timestamps in timestamps_dict.items():
#             if not timestamps:
#                 continue
#
#             # Estimate duration (each violation event = ~4 seconds minimum)
#             violation_duration = len(timestamps) * 4.0
#
#             # Get intensity for this violation type
#             max_intensity = 0.0
#             if "gaze" in violation_type:
#                 # Use eye threshold
#                 if "left" in violation_type or "right" in violation_type:
#                     max_intensity = abs(baseline.eye_mean) if baseline.eye_mean > thresholds.eye else thresholds.eye
#             elif "head" in violation_type:
#                 # Use yaw/pitch threshold
#                 if "left" in violation_type or "right" in violation_type:
#                     max_intensity = abs(baseline.yaw_mean) if baseline.yaw_mean > thresholds.yaw else thresholds.yaw
#                 elif "up" in violation_type or "down" in violation_type:
#                     max_intensity = abs(baseline.pitch_mean) if baseline.pitch_mean > thresholds.pitch else thresholds.pitch
#
#             # Classify into risk categories based on intensity
#             if "face_missing" in violation_type or "multiple_faces" in violation_type:
#                 high_risk_duration += violation_duration
#             elif max_intensity > 0:
#                 threshold_ratio = max_intensity / max(thresholds.yaw, thresholds.pitch, thresholds.eye)
#                 if threshold_ratio > 2.0:
#                     high_risk_duration += violation_duration
#                 elif threshold_ratio > 1.5:
#                     suspicious_duration += violation_duration
#                 else:
#                     mild_duration += violation_duration
#             else:
#                 mild_duration += violation_duration
#
#         # Apply weights
#         duration_score = (
#             mild_duration * self.config.DURATION_WEIGHT_MILD +
#             suspicious_duration * self.config.DURATION_WEIGHT_SUSPICIOUS +
#             high_risk_duration * self.config.DURATION_WEIGHT_HIGH_RISK
#         )
#
#         return {
#             "mild_sec": round(mild_duration, 1),
#             "suspicious_sec": round(suspicious_duration, 1),
#             "high_risk_sec": round(high_risk_duration, 1),
#             "total_duration_score": round(float(duration_score), 2)
#         }
#
#     def _compute_intensity_score(
#         self,
#         max_intensities: Dict[str, float],
#         thresholds: AdaptiveThresholds,
#         baseline: Baseline
#     ) -> float:
#         """
#         intensity = abs(angle - baseline_mean)
#         intensity_score = (intensity / threshold) * INTENSITY_WEIGHT
#
#         Severity Weights (from PDF):
#         - Slight: 0.3 (Within normal range)
#         - Moderate: 0.7 (Noticeable deviation)
#         - Extreme: 1.4 (Severe angle deviation)
#         """
#         intensity_score = 0.0
#
#         for violation_type, max_intensity in max_intensities.items():
#             if max_intensity == 0.0:
#                 continue
#
#             # Get baseline mean and threshold for this violation type
#             baseline_mean = 0.0
#             threshold = 0.0
#
#             if "gaze" in violation_type:
#                 baseline_mean = baseline.eye_mean
#                 threshold = thresholds.eye
#             elif "head_left" in violation_type or "head_right" in violation_type:
#                 baseline_mean = baseline.yaw_mean
#                 threshold = thresholds.yaw
#             elif "head_up" in violation_type or "head_down" in violation_type:
#                 baseline_mean = baseline.pitch_mean
#                 threshold = thresholds.pitch
#             else:
#                 continue  # Skip face-based events
#
#             # Calculate intensity deviation
#             intensity_deviation = abs(max_intensity - baseline_mean)
#
#             if threshold > 0:
#                 intensity_ratio = intensity_deviation / threshold
#
#                 # Determine severity weight
#                 if intensity_ratio > 2.0:
#                     weight = self.config.INTENSITY_WEIGHT_EXTREME
#                 elif intensity_ratio > 1.0:
#                     weight = self.config.INTENSITY_WEIGHT_MODERATE
#                 else:
#                     weight = self.config.INTENSITY_WEIGHT_SLIGHT
#
#                 intensity_score += intensity_ratio * weight
#
#         return round(float(intensity_score), 2)
#
#     def _compute_final_risk_score(
#         self,
#         count_score: float,
#         duration_score: float,
#         intensity_score: float
#     ) -> int:
#         """
#         total_raw = count_score + duration_score + intensity_score
#         RISK_SCORE = min(100, normalize(total_raw))
#         normalize(x) = (x / MAX_POSSIBLE_SCORE) * 100
#
#         Returns: Risk score from 0-100
#         """
#         total_raw = count_score + duration_score + intensity_score
#
#         # Normalize to 0-100 scale
#         # MAX_POSSIBLE_SCORE is estimated based on worst-case scenario
#         MAX_POSSIBLE_SCORE = 200.0  # Reasonable upper bound
#
#         normalized_score = (total_raw / MAX_POSSIBLE_SCORE) * 100.0
#         risk_score = min(100, int(round(normalized_score)))
#
#         return risk_score
#
#     def cleanup(self) -> None:
#         self.log_info("Scoring service cleaned up")
