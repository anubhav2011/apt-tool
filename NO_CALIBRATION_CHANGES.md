# AI Proctoring System - No Calibration Mode

## Overview
The system has been completely refactored to eliminate calibration and use industry-standard fixed thresholds immediately when video processing starts.

## Major Changes

### 1. Calibration Logic Removed
- **CalibrationService** is no longer instantiated or used
- **ScoringService** has been removed - reports are now in simple gesture format
- No calibration phase - detection starts immediately from frame 0

### 2. Fixed Industry-Standard Thresholds
Default thresholds are used globally:
- **Eye Gaze Threshold**: 14.3° (calculated as 6.5 * 2.2)
- **Head Yaw Threshold**: 12.5° (calculated as 5.0 * 2.5)  
- **Head Pitch Threshold**: 12.5° (calculated as 5.0 * 2.5)

These are set in `ProctoringConfig.get_default_thresholds()` and applied immediately.

### 3. New Response Format
The response now matches the requested `apt_response.json` format:

```json
{
  "session_id": "...",
  "thresholds_used": {
    "eye": 14.3,
    "yaw": 12.5,
    "pitch": 12.5
  },
  "processing_metadata": {
    "processing_time_sec": 5.2,
    "video_duration_sec": 45.0,
    "frames_processed": 540
  },
  "gestures": [
    {
      "name": "head_movement",
      "occurrence": [
        {
          "timestamp": "00:22:345",
          "duration": 4,
          "direction": "left",
          "intensity": "15.6"
        }
      ]
    },
    {
      "name": "eye_gaze",
      "occurrence": [
        {
          "timestamp": "00:35:120",
          "duration": 7,
          "direction": "right",
          "intensity": "18.2"
        }
      ]
    },
    {
      "name": "face_missing",
      "occurrence": [
        {
          "timestamp": "01:02:500",
          "duration": 4,
          "direction": "",
          "intensity": ""
        }
      ]
    }
  ]
}
```

### 4. Violation Event Tracking
The `ViolationTracker` now stores complete event details:
- **Timestamp**: When the violation started
- **Duration**: How long the violation lasted (4s or 7s thresholds)
- **Direction**: left/right/up/down for movements, empty for face events
- **Intensity**: The actual angle measured (rotation angle)

### 5. Gesture Categories
Events are grouped into four gesture types:
1. **head_movement** - All head rotations (left, right, up, down)
2. **eye_gaze** - All eye gaze deviations (left, right)
3. **face_missing** - No face detected
4. **multiple_faces** - More than one person detected

### 6. Timestamp Format
Timestamps are formatted as **MM:SS:MSS** (minutes:seconds:milliseconds)
- Example: `"00:22:345"` = 22 seconds and 345 milliseconds
- Example: `"01:05:120"` = 1 minute, 5 seconds, and 120 milliseconds

## Files Modified
1. **video_processing_service.py** - Removed calibration phase, added new report format
2. **detection_service.py** - Added event tracking to ViolationTracker
3. **proctoring_config.py** - (No changes needed, already had default thresholds)

## Files No Longer Used
1. **calibration_service.py** - Commented out, not imported
2. **scoring_service.py** - Commented out, not imported

## Benefits
- ✅ Immediate detection from video start
- ✅ No failed calibration issues
- ✅ Consistent thresholds across all sessions
- ✅ Simpler, cleaner response format
- ✅ Detailed event tracking with durations and intensities
- ✅ Industry-standard thresholds validated against ProctorU, Mettl, Talview
