"""
Person 3 — Object Detection & Environment Pipeline — Integration Demo
======================================================================

Runs the full webcam → YOLO → TemporalWindowAggregator pipeline in real time.
Displays the 5 environment features required by the fusion model:
    phone_detected | phone_confidence | notes_detected
    person_count_anomaly | suspicious_objects_count

Controls:
    q / Q  — quit and release the camera cleanly

Usage:
    cd /home/agnivesh/novel
    source venv/bin/activate
    python p3/tests/test_object_detection.py
    python p3/tests/test_object_detection.py --conf 0.40 --camera-index 0
    python p3/tests/test_object_detection.py --max-frames 200

NOTE: This script is an integration demo, NOT a unit test.
      Unit tests live in p3/tests/test_environment_features.py.
      No "CHEATING" or "MALPRACTICE" labels are displayed.
      Object detection provides evidence only; final judgement belongs
      to the downstream fusion model.
"""

import argparse
import os
import sys
import time
import cv2

# Allow importing from p3/src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.camera import Camera
from src.environment_features import TemporalWindowAggregator
from src.object_detector import ObjectDetector


def _put_text(frame, text: str, pos: tuple, color=(0, 255, 255), scale=0.60, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="P3 Object Detection & Environment Pipeline Demo"
    )
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Confidence threshold (default: 0.40)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to run (0 = infinite)")
    args = parser.parse_args()

    # --- Initialise components ---
    print(f"[P3] Camera index: {args.camera_index}")
    cam = Camera(camera_index=args.camera_index)
    if not cam.start():
        print("[P3] ERROR: Could not open camera. Exiting.")
        return

    info = cam.get_info()
    if info:
        print(f"[P3] Camera: {info['width']}x{info['height']} @ {info['fps']:.1f} FPS")

    print(f"[P3] Loading ObjectDetector (YOLOv8n, conf={args.conf})...")
    detector = ObjectDetector(confidence_threshold=args.conf)

    print("[P3] Initialising TemporalWindowAggregator (10s window)...")
    aggregator = TemporalWindowAggregator(
        window_seconds=10.0,
        persistence_ratio=0.30,
        confidence_threshold=args.conf,
    )

    window_name = "P3 Environment Detection  |  Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # FPS tracking
    frame_count = 0
    total_frames = 0
    fps_display = 0.0
    fps_start = time.time()

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                print("[P3] WARNING: Failed to capture frame.")
                break

            frame_count += 1
            total_frames += 1

            # Update FPS counter every second
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # ── Step 1: YOLO detection ──────────────────────────────────────
            detection_output = detector.detect(frame, frame_index=total_frames)
            detection_output["frame_bgr"] = frame

            # ── Step 2: Temporal feature aggregation ────────────────────────
            env = aggregator.update(detection_output)

            # ── Step 3: Annotate frame with bounding boxes ──────────────────
            annotated = detector.draw_detections(frame, detection_output)

            # ── Step 4: Overlay 5 environment features ──────────────────────
            inference_ms = detection_output["inference_time_ms"]

            # Header: pipeline telemetry
            _put_text(
                annotated,
                f"FPS: {fps_display:.1f}  |  YOLO: {inference_ms:.1f}ms  |  "
                f"Device: {detector.device}  |  Window: {env['window_duration_seconds']:.1f}s",
                (10, 28),
                color=(0, 255, 255),
                scale=0.55,
            )

            # Row 1: phone
            phone_str = "DETECTED" if env["phone_detected"] else "none"
            phone_color = (0, 80, 255) if env["phone_detected"] else (180, 180, 180)
            _put_text(
                annotated,
                f"Phone: {phone_str}  (conf {env['phone_confidence']:.2f})",
                (10, 58),
                color=phone_color,
            )

            # Row 2: notes
            notes_str = "DETECTED" if env["notes_detected"] else "none"
            notes_color = (0, 80, 255) if env["notes_detected"] else (180, 180, 180)
            _put_text(
                annotated,
                f"Notes (book proxy): {notes_str}",
                (10, 86),
                color=notes_color,
            )

            # Row 3: person anomaly
            person_anom = env["person_count_anomaly"]
            person_str = (
                f"ANOMALY  ({env['raw_person_count']} persons)" if person_anom
                else f"OK  ({env['raw_person_count']} person)"
            )
            person_color = (0, 80, 255) if person_anom else (0, 220, 80)
            _put_text(
                annotated,
                f"Persons: {person_str}",
                (10, 114),
                color=person_color,
            )

            # Row 4: suspicious objects
            susp_count = env["suspicious_objects_count"]
            susp_color = (0, 80, 255) if susp_count > 0 else (180, 180, 180)
            _put_text(
                annotated,
                f"Suspicious objects: {susp_count}",
                (10, 142),
                color=susp_color,
            )

            # Footer: obs count
            _put_text(
                annotated,
                f"Obs in window: {env['window_obs_count']}  |  Press Q to quit",
                (10, 170),
                color=(160, 160, 160),
                scale=0.48,
                thickness=1,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("[P3] Quit requested.")
                break

            if args.max_frames > 0 and total_frames >= args.max_frames:
                print(f"[P3] Reached max-frames limit ({args.max_frames}). Exiting.")
                break

    except KeyboardInterrupt:
        print("[P3] Interrupted.")
    except Exception as exc:
        print(f"[P3] Unexpected error: {exc}")
        raise
    finally:
        print("[P3] Releasing camera and windows...")
        cam.release()
        cv2.destroyAllWindows()
        print(
            f"[P3] Done. Processed {total_frames} frames  "
            f"avg FPS ≈ {fps_display:.1f}"
        )


if __name__ == "__main__":
    main()
