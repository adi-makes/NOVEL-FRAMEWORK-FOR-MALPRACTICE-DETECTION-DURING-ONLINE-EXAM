"""
Person 3 — Environment Pipeline Benchmark
==========================================

Measures real inference latency and end-to-end pipeline FPS on the local laptop.
All values are measured from actual hardware — no theoretical figures.

Usage:
    cd /home/agnivesh/novel
    source venv/bin/activate
    python p3/src/benchmark.py
    python p3/src/benchmark.py --frames 100 --warmup 10 --camera-index 0
    python p3/src/benchmark.py --no-camera  # synthetic frames only (no webcam needed)

Output fields recorded:
    Python version
    OpenCV version
    Ultralytics version
    PyTorch version
    YOLO model
    Input resolution (from camera or synthetic)
    Inference resolution (YOLO input)
    Device (CPU / CUDA)
    Warmup frames
    Benchmark frames
    YOLO inference latency (mean, p50, p95, p99)
    Camera capture latency (mean)
    Full pipeline FPS (capture + YOLO + aggregator + overhead)
"""

import sys
import os
import time
import argparse
import platform
import statistics

# Allow importing from p3/src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from src.camera import Camera
from src.object_detector import ObjectDetector
from src.environment_features import TemporalWindowAggregator


def _synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a realistic-looking BGR synthetic frame (no webcam needed)."""
    frame = np.random.randint(80, 180, (height, width, 3), dtype=np.uint8)
    # Add some structure so YOLO sees a plausible image
    cv2.rectangle(frame, (width // 4, height // 4), (3 * width // 4, 3 * height // 4),
                  (200, 200, 200), -1)
    return frame


def _percentile(data: list, pct: float) -> float:
    if not data:
        return 0.0
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * pct / 100)
    idx = min(idx, len(sorted_d) - 1)
    return sorted_d[idx]


def run_benchmark(
    n_warmup: int = 10,
    n_frames: int = 100,
    camera_index: int = 0,
    use_camera: bool = True,
    confidence: float = 0.40,
) -> dict:
    """
    Run the benchmark and return a results dict.

    Parameters
    ----------
    n_warmup  : warmup frames (discarded from timing)
    n_frames  : frames to measure
    camera_index : webcam index
    use_camera   : if False, use synthetic frames (no webcam required)
    confidence   : YOLO confidence threshold
    """
    import ultralytics
    import torch

    print("=" * 60)
    print("  Person 3 — Environment Pipeline Benchmark")
    print("=" * 60)

    # --- System info ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "CUDA (unknown)"

    print(f"  Python    : {platform.python_version()}")
    print(f"  OpenCV    : {cv2.__version__}")
    print(f"  Ultralytics: {ultralytics.__version__}")
    print(f"  PyTorch   : {torch.__version__}")
    print(f"  Device    : {device}" + (f"  ({gpu_name})" if gpu_name else ""))
    print(f"  Warmup    : {n_warmup} frames")
    print(f"  Benchmark : {n_frames} frames")
    print(f"  Confidence: {confidence}")
    print()

    # --- Camera / synthetic frame setup ---
    cam = None
    frame_width, frame_height = 640, 480

    if use_camera:
        cam = Camera(camera_index=camera_index)
        if not cam.start():
            print("  WARNING: Camera not available. Falling back to synthetic frames.")
            use_camera = False
        else:
            info = cam.get_info()
            if info:
                frame_width = info["width"]
                frame_height = info["height"]
                print(f"  Camera    : {frame_width}x{frame_height} @ {info['fps']:.0f} FPS")

    if not use_camera:
        print(f"  Mode      : Synthetic frames ({frame_width}x{frame_height})")

    print()

    # --- Load detector + aggregator ---
    print("  Loading ObjectDetector (YOLOv8n)...")
    detector = ObjectDetector(confidence_threshold=confidence)
    aggregator = TemporalWindowAggregator(
        window_seconds=10.0,
        persistence_ratio=0.30,
        confidence_threshold=confidence,
    )
    print(f"  YOLO model: {detector.model.model_name if hasattr(detector.model, 'model_name') else 'yolov8n'}")
    print()

    # --- Warmup ---
    print(f"  Running {n_warmup} warmup frames (discarded)...")
    for _ in range(n_warmup):
        if use_camera:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                frame = _synthetic_frame(frame_width, frame_height)
        else:
            frame = _synthetic_frame(frame_width, frame_height)
        detector.detect(frame)
    print("  Warmup complete.\n")

    # --- Benchmark ---
    print(f"  Benchmarking {n_frames} frames...")

    inference_latencies_ms = []
    capture_latencies_ms = []
    pipeline_start = time.perf_counter()

    for i in range(n_frames):
        # Camera capture timing
        t_cap_start = time.perf_counter()
        if use_camera:
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                frame = _synthetic_frame(frame_width, frame_height)
        else:
            frame = _synthetic_frame(frame_width, frame_height)
        capture_latencies_ms.append((time.perf_counter() - t_cap_start) * 1000.0)

        # YOLO + aggregator
        detection_output = detector.detect(frame, frame_index=i)
        detection_output["frame_bgr"] = frame
        aggregator.update(detection_output)

        inference_latencies_ms.append(detection_output["inference_time_ms"])

        if (i + 1) % 25 == 0:
            print(f"    ... frame {i + 1}/{n_frames}")

    pipeline_elapsed = time.perf_counter() - pipeline_start
    pipeline_fps = n_frames / pipeline_elapsed

    # --- Cleanup ---
    if cam:
        cam.release()

    # --- Results ---
    mean_infer = statistics.mean(inference_latencies_ms)
    p50_infer = _percentile(inference_latencies_ms, 50)
    p95_infer = _percentile(inference_latencies_ms, 95)
    p99_infer = _percentile(inference_latencies_ms, 99)
    mean_capture = statistics.mean(capture_latencies_ms)

    results = {
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "gpu_name": gpu_name,
        "model": "yolov8n",
        "input_resolution": f"{frame_width}x{frame_height}",
        "confidence_threshold": confidence,
        "warmup_frames": n_warmup,
        "benchmark_frames": n_frames,
        "inference_latency_mean_ms": round(mean_infer, 2),
        "inference_latency_p50_ms": round(p50_infer, 2),
        "inference_latency_p95_ms": round(p95_infer, 2),
        "inference_latency_p99_ms": round(p99_infer, 2),
        "capture_latency_mean_ms": round(mean_capture, 2),
        "pipeline_fps": round(pipeline_fps, 2),
        "pipeline_elapsed_s": round(pipeline_elapsed, 2),
    }

    print()
    print("=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Python          : {results['python_version']}")
    print(f"  OpenCV          : {results['opencv_version']}")
    print(f"  Ultralytics     : {results['ultralytics_version']}")
    print(f"  PyTorch         : {results['torch_version']}")
    print(f"  Device          : {results['device']}" +
          (f"  ({results['gpu_name']})" if results["gpu_name"] else ""))
    print(f"  Model           : {results['model']}")
    print(f"  Input resolution: {results['input_resolution']}")
    print(f"  Conf threshold  : {results['confidence_threshold']}")
    print()
    print(f"  YOLO inference latency:")
    print(f"    mean  = {results['inference_latency_mean_ms']:.2f} ms")
    print(f"    p50   = {results['inference_latency_p50_ms']:.2f} ms")
    print(f"    p95   = {results['inference_latency_p95_ms']:.2f} ms")
    print(f"    p99   = {results['inference_latency_p99_ms']:.2f} ms")
    print()
    print(f"  Camera capture latency (mean): {results['capture_latency_mean_ms']:.2f} ms")
    print(f"  Pipeline FPS (end-to-end)    : {results['pipeline_fps']:.1f} FPS")
    print(f"  Total time ({n_frames} frames): {results['pipeline_elapsed_s']:.1f} s")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="P3 Environment Pipeline Benchmark")
    parser.add_argument("--frames", type=int, default=100,
                        help="Number of benchmark frames (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup frames to discard (default: 10)")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Use synthetic frames instead of webcam")
    parser.add_argument("--conf", type=float, default=0.40,
                        help="Confidence threshold (default: 0.40)")
    args = parser.parse_args()

    run_benchmark(
        n_warmup=args.warmup,
        n_frames=args.frames,
        camera_index=args.camera_index,
        use_camera=not args.no_camera,
        confidence=args.conf,
    )


if __name__ == "__main__":
    main()
