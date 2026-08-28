import argparse
import os
import sys
import time
import cv2

# Add parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.camera import Camera
from src.environment_features import EnvironmentFeatureExtractor
from src.object_detector import ObjectDetector


def main():
    parser = argparse.ArgumentParser(
        description="P3 Object Detection & Environment Pipeline Test"
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Max frames to run (0 for infinite)"
    )
    args = parser.parse_args()

    print(f"Initializing camera at index {args.camera_index}...")
    cam = Camera(camera_index=args.camera_index)
    if not cam.start():
        print("Error: Could not open camera.")
        return

    print(f"Initializing ObjectDetector (YOLOv8n, conf={args.conf})...")
    detector = ObjectDetector(confidence_threshold=args.conf)

    print("Initializing EnvironmentFeatureExtractor...")
    env_extractor = EnvironmentFeatureExtractor(
        history_window_size=10, stability_threshold_ratio=0.5
    )

    window_name = "P3 Object & Environment Detection Test - Press 'q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_frames = 0
    start_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                print("Warning: Failed to capture frame.")
                break

            frame_count += 1
            total_frames += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Step 1: Run Object Detector
            detection_output = detector.detect(frame, frame_index=total_frames)

            # Step 2: Extract Environment Features
            env_features = env_extractor.extract_features(detection_output)

            # Step 3: Draw Bounding Boxes & Labels
            annotated_frame = detector.draw_detections(frame, detection_output)

            # Overlay pipeline telemetry & environment summary
            inference_ms = detection_output["inference_time_ms"]
            persons = env_features["person_count"]
            phone_str = "YES" if env_features["phone_detected"] else "NO"
            phone_stable = "STABLE" if env_features["temporal_stability"]["phone_stable"] else "RAW"
            book_str = "YES" if env_features["book_detected"] else "NO"

            # Telemetry header text
            line1 = f"Pipeline FPS: {fps_display:.1f} | YOLO Infer: {inference_ms:.1f}ms | Device: {detector.device}"
            line2 = f"Persons: {persons} | Phone: {phone_str} ({phone_stable}) | Book: {book_str}"

            cv2.putText(
                annotated_frame,
                line1,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                line2,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if persons == 1 else (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                "Press 'Q' to Exit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("Quit requested by user.")
                break

            if args.max_frames > 0 and total_frames >= args.max_frames:
                print(f"Reached max frame limit ({args.max_frames}). Exiting.")
                break

    except Exception as e:
        print(f"Unexpected error during object detection test: {e}")
    finally:
        print("Cleaning up resources...")
        cam.release()
        cv2.destroyAllWindows()
        print("Object & Environment Detection test completed cleanly.")


if __name__ == "__main__":
    main()
