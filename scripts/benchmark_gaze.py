"""
Performance Benchmark for Gaze Estimation Component.
Measures execution time and FPS on synthetic CPU frame streams.
"""

import sys
import os
import time
import numpy as np

from exam_proctoring.gaze.gaze_estimator import GazeEstimator
from exam_proctoring.gaze.feature_extractor import GazeFeatureExtractor


def benchmark_gaze_pipeline(num_frames: int = 100):
    print(f"[Benchmark] Initializing GazeEstimator on CPU...")
    estimator = GazeEstimator()
    extractor = GazeFeatureExtractor()

    # Generate synthetic 640x480 RGB image with random noise
    synthetic_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]

    print(f"[Benchmark] Processing {num_frames} frames...")
    start_time = time.time()
    latencies = []

    for frame in synthetic_frames:
        t0 = time.time()
        res = estimator.process_frame(frame)
        extractor.update(res)
        latencies.append((time.time() - t0) * 1000.0)

    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = num_frames / total_time

    print("\n" + "=" * 40)
    print("GAZE PIPELINE PERFORMANCE REPORT")
    print("=" * 40)
    print(f"Total Frames Processed : {num_frames}")
    print(f"Total Runtime          : {total_time:.3f} s")
    print(f"Throughput             : {fps:.2f} FPS")
    print(f"Avg Latency per Frame  : {avg_latency:.2f} ms (+/- {std_latency:.2f} ms)")
    print("=" * 40)

    return {
        "num_frames": num_frames,
        "fps": fps,
        "avg_latency_ms": avg_latency,
        "total_time": total_time
    }


if __name__ == "__main__":
    benchmark_gaze_pipeline(100)
