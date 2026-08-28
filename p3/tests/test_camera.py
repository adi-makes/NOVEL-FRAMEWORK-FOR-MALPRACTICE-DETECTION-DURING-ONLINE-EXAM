import argparse
import os
import sys
import time
import cv2

# Add parent directory to sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.camera import Camera


def main():
    parser = argparse.ArgumentParser(description="P3 Camera Test Script")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to capture (0 for infinite)")
    args = parser.parse_args()

    print(f"Initializing camera at index {args.camera_index}...")

    cam = Camera(camera_index=args.camera_index)
    if not cam.start():
        print("Failed to initialize camera. Exiting.")
        return

    info = cam.get_info()
    if info:
        print(f"Camera Info: Resolution = {info['width']}x{info['height']}, FPS = {info['fps']}")

    window_name = "P3 Camera Test - Press 'q' to Quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    total_frames = 0
    start_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                print("Error: Could not read frame from camera.")
                break

            frame_count += 1
            total_frames += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps_display = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Overlay info on frame
            overlay_text = f"FPS: {fps_display:.1f} | Res: {frame.shape[1]}x{frame.shape[0]}"
            cv2.putText(
                frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'Q' to Exit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            # Wait for 1 ms and check if 'q' or 'Q' key was pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("Quit requested by user.")
                break

            if args.max_frames > 0 and total_frames >= args.max_frames:
                print(f"Reached maximum frame limit of {args.max_frames}. Exiting test.")
                break

    except Exception as e:
        print(f"Unexpected error during camera test: {e}")
    finally:
        print("Releasing camera and destroying windows...")
        cam.release()
        cv2.destroyAllWindows()
        print("Camera test completed cleanly.")


if __name__ == "__main__":
    main()
