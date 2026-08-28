import argparse
import os
import sys
import time
import cv2

# Add parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera import Camera
from src.face_landmarks import FaceLandmarkDetector


def main():
    parser = argparse.ArgumentParser(description="P3 Face & Iris Landmark Detection Test")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Max frames to run (0 for infinite)"
    )
    args = parser.parse_args()

    print(f"Initializing camera at index {args.camera_index}...")
    cam = Camera(camera_index=args.camera_index)
    if not cam.start():
        print("Error: Could not open camera.")
        return

    print("Initializing Face & Iris Landmark Detector (MediaPipe FaceMesh)...")
    detector = FaceLandmarkDetector(max_num_faces=1, refine_landmarks=True)

    window_name = "P3 Face & Iris Landmarks Test - Press 'q' to Quit"
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

            # Process frame for face & iris landmarks
            result = detector.process(frame)

            # Draw mesh & highlights
            annotated_frame = detector.draw_landmarks(frame, result, draw_iris_highlight=True)

            # Overlay status text
            status_color = (0, 255, 0) if result["face_detected"] else (0, 0, 255)
            status_text = (
                f"Face Detected: YES ({result['num_faces']})"
                if result["face_detected"]
                else "Face Detected: NO"
            )

            cv2.putText(
                annotated_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"FPS: {fps_display:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                "Press 'Q' to Exit",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
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
        print(f"Unexpected error during landmark test: {e}")
    finally:
        print("Cleaning up resources...")
        detector.close()
        cam.release()
        cv2.destroyAllWindows()
        print("Face & Iris Landmark test completed cleanly.")


if __name__ == "__main__":
    main()
