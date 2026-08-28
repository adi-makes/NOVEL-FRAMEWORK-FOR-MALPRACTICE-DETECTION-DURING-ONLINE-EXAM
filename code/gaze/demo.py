"""
Real-time OpenCV Demo Application for Gaze Estimation.

Displays camera feed with face/iris landmarks, raw and calibrated gaze predictions,
head yaw/pitch angles, confidence meters, and the 7 temporal gaze features.
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

# Add parent directory to sys.path to support execution as python code/gaze/demo.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from gaze.gaze_estimator import (
    GazeEstimator,
    LEFT_IRIS_CENTER, LEFT_IRIS_PERIMETER,
    RIGHT_IRIS_CENTER, RIGHT_IRIS_PERIMETER,
    LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER
)
from gaze.calibration import GazeCalibrator
from gaze.feature_extractor import GazeFeatureExtractor


def run_gaze_demo(camera_id: int = 0, calib_file: str = None):
    """Runs real-time OpenCV webcam gaze tracking demo."""
    print("[Gaze Demo] Initializing GazeEstimator & Calibrator...")
    calibrator = GazeCalibrator(calib_file=calib_file)
    if calibrator.is_fitted:
        print("[Gaze Demo] Loaded calibration parameters.")
    else:
        print("[Gaze Demo] No calibration file loaded; displaying raw predictions.")

    estimator = GazeEstimator(calibrator=calibrator)
    feature_extractor = GazeFeatureExtractor(window_seconds=10.0)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Gaze Demo] Error: Cannot open camera {camera_id}.")
        return

    print("[Gaze Demo] Starting webcam loop. Press 'q' or ESC to exit.")
    fps_counter = 0
    start_time = time.time()
    fps = 0.0

    cv2.namedWindow("NOVA Gaze Stream Demo", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Gaze Demo] Frame read failed.")
            break

        # Mirror frame horizontally for user comfort
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        t0 = time.time()
        result = estimator.process_frame(frame, timestamp=t0)
        t_proc = (time.time() - t0) * 1000.0  # ms per frame

        # Update 10s temporal feature buffer
        temporal_result = feature_extractor.update(result)
        feats = temporal_result["features"]

        fps_counter += 1
        if time.time() - start_time >= 1.0:
            fps = fps_counter / (time.time() - start_time)
            fps_counter = 0
            start_time = time.time()

        # Visualization
        disp_frame = frame.copy()

        if result["face_detected"] and result["landmarks_3d"]:
            lms = result["landmarks_3d"]

            # Draw key eye boundary landmarks
            for idx in [LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER]:
                pt = (int(lms[idx][0] * w), int(lms[idx][1] * h))
                cv2.circle(disp_frame, pt, 2, (0, 255, 0), -1)

            # Draw iris landmarks (cyan for left, magenta for right)
            for idx in [LEFT_IRIS_CENTER] + LEFT_IRIS_PERIMETER:
                pt = (int(lms[idx][0] * w), int(lms[idx][1] * h))
                cv2.circle(disp_frame, pt, 2, (255, 255, 0), -1)

            for idx in [RIGHT_IRIS_CENTER] + RIGHT_IRIS_PERIMETER:
                pt = (int(lms[idx][0] * w), int(lms[idx][1] * h))
                cv2.circle(disp_frame, pt, 2, (255, 0, 255), -1)

            # Draw gaze target crosshair on screen if available
            if result["gaze_x"] is not None and result["gaze_y"] is not None:
                gx_px = int(result["gaze_x"] * w)
                gy_px = int(result["gaze_y"] * h)

                cv2.drawMarker(disp_frame, (gx_px, gy_px), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
                cv2.circle(disp_frame, (gx_px, gy_px), 8, (0, 255, 255), 2)

        # Overlay Info Panel
        overlay = disp_frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, disp_frame, 0.4, 0, disp_frame)

        # Text Lines
        cv2.putText(disp_frame, f"FPS: {fps:.1f} ({t_proc:.1f} ms/frame)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        face_status = "DETECTED" if result["face_detected"] else "NOT DETECTED"
        face_color = (0, 255, 0) if result["face_detected"] else (0, 0, 255)
        cv2.putText(disp_frame, f"Face: {face_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

        gx_str = f"{result['gaze_x']:.2f}" if result['gaze_x'] is not None else "N/A"
        gy_str = f"{result['gaze_y']:.2f}" if result['gaze_y'] is not None else "N/A"
        cv2.putText(disp_frame, f"Gaze (X, Y): ({gx_str}, {gy_str})", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        conf_str = f"{result['gaze_confidence']:.2f}"
        cv2.putText(disp_frame, f"Confidence: {conf_str}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        yaw_str = f"{result['head_yaw']:.1f} deg" if result['head_yaw'] is not None else "N/A"
        pitch_str = f"{result['head_pitch']:.1f} deg" if result['head_pitch'] is not None else "N/A"
        cv2.putText(disp_frame, f"Head Yaw / Pitch: {yaw_str} / {pitch_str}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 7 Temporal Features Summary Header
        cv2.putText(disp_frame, "--- 7D Temporal Features (10s) ---", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(disp_frame, f"1. Gaze Dev: {feats['gaze_deviation']:.2f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"2. Fixation Dur: {feats['fixation_duration']:.2f}s", (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"3. Fixation Count: {feats['fixation_count']:.0f}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"4. Saccade Vel: {feats['saccade_velocity']:.2f}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"5. Confidence: {feats['gaze_confidence']:.2f}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"6. Head Yaw: {feats['head_yaw']:.1f} deg", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(disp_frame, f"7. Head Pitch: {feats['head_pitch']:.1f} deg", (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("NOVA Gaze Stream Demo", disp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("[Gaze Demo] Exiting demo.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOVA Gaze Stream OpenCV Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--calibration", type=str, default=None, help="Path to calibration.json file")
    args = parser.parse_args()

    run_gaze_demo(camera_id=args.camera, calib_file=args.calibration)
