"""
Calibration Module for Gaze Estimation.

Maps raw eye/head features to normalized screen-space coordinates [0, 1] x [0, 1]
using a 3x3 grid polynomial mapping. Includes interactive OpenCV calibration UI.
"""

import os
import json
import time
import numpy as np
import cv2

# Standard 3x3 grid points in normalized screen space
GRID_POINTS_3X3 = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),   # Top row
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),   # Middle row
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),   # Bottom row
]


def get_default_calibration_path() -> str:
    """Returns absolute path to default calibration file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    calib_path = os.path.join(base_dir, "models", "gaze", "calibration.json")
    return calib_path


class GazeCalibrator:
    """
    Calibrates raw gaze observations (raw_x, raw_y, yaw, pitch) to normalized screen space.
    Uses polynomial feature mapping fit via Ridge / Least-Squares regression.
    """

    def __init__(self, calib_file: str = None):
        self.calib_file = calib_file if calib_file else get_default_calibration_path()
        self.is_fitted = False
        self.coef_x = None
        self.coef_y = None

        # Attempt auto-load if calibration file exists
        if os.path.exists(self.calib_file):
            try:
                self.load(self.calib_file)
            except Exception:
                self.is_fitted = False

    def _extract_polynomial_features(self, raw_x: float, raw_y: float, yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
        """Extracts 2nd-order polynomial feature vector."""
        x = float(raw_x)
        y = float(raw_y)
        y_angle = (float(yaw) if yaw is not None else 0.0) / 45.0
        p_angle = (float(pitch) if pitch is not None else 0.0) / 45.0

        return np.array([
            1.0,
            x,
            y,
            y_angle,
            p_angle,
            x ** 2,
            y ** 2,
            x * y,
            x * y_angle,
            y * p_angle
        ], dtype=np.float64)

    def filter_samples(self, samples: list, min_confidence: float = 0.3) -> list:
        """Filters out invalid or outlier samples from calibration collection."""
        valid = []
        for s in samples:
            if not s.get("face_detected", False):
                continue
            if s.get("gaze_confidence", 0.0) < min_confidence:
                continue
            if s.get("raw_gaze_x") is None or s.get("raw_gaze_y") is None:
                continue
            valid.append(s)

        if not valid:
            return []

        # Remove statistical outliers using Median Absolute Deviation (MAD)
        raw_xs = np.array([s["raw_gaze_x"] for s in valid])
        raw_ys = np.array([s["raw_gaze_y"] for s in valid])

        med_x = float(np.median(raw_xs))
        med_y = float(np.median(raw_ys))
        mad_x = float(np.median(np.abs(raw_xs - med_x)))
        mad_y = float(np.median(np.abs(raw_ys - med_y)))

        thresh_x = max(3.5 * mad_x, 0.25)
        thresh_y = max(3.5 * mad_y, 0.25)

        filtered = []
        for s in valid:
            if (abs(s["raw_gaze_x"] - med_x) <= thresh_x and
                    abs(s["raw_gaze_y"] - med_y) <= thresh_y):
                filtered.append(s)

        return filtered if filtered else valid

    def fit(self, observations: list, target_points: list, alpha: float = 1e-3) -> bool:
        """
        Fits calibration regression model from pairs of raw observations and target screen points.

        Args:
            observations: list of dicts with keys 'raw_gaze_x', 'raw_gaze_y', 'head_yaw', 'head_pitch'
            target_points: list of tuples (target_x, target_y) corresponding to each observation
        """
        if len(observations) < 6 or len(observations) != len(target_points):
            return False

        X = []
        Y_x = []
        Y_y = []

        for obs, tgt in zip(observations, target_points):
            feat = self._extract_polynomial_features(
                obs["raw_gaze_x"],
                obs["raw_gaze_y"],
                obs.get("head_yaw", 0.0),
                obs.get("head_pitch", 0.0)
            )
            X.append(feat)
            Y_x.append(tgt[0])
            Y_y.append(tgt[1])

        X = np.array(X, dtype=np.float64)
        Y_x = np.array(Y_x, dtype=np.float64)
        Y_y = np.array(Y_y, dtype=np.float64)

        # Ridge regression solve: (X^T X + alpha * I)^-1 X^T Y
        n_features = X.shape[1]
        I = np.eye(n_features, dtype=np.float64)
        I[0, 0] = 0.0  # Do not penalize bias term

        try:
            A = np.dot(X.T, X) + alpha * I
            self.coef_x = np.linalg.solve(A, np.dot(X.T, Y_x))
            self.coef_y = np.linalg.solve(A, np.dot(X.T, Y_y))
            self.is_fitted = True
            return True
        except np.linalg.LinAlgError:
            self.is_fitted = False
            return False

    def predict(self, raw_x: float, raw_y: float, yaw: float = 0.0, pitch: float = 0.0) -> tuple:
        """
        Predicts calibrated screen coordinates (calib_x, calib_y) in [0, 1].
        """
        if not self.is_fitted or self.coef_x is None or self.coef_y is None:
            # Fallback to uncalibrated raw values
            return float(np.clip(raw_x, 0.0, 1.0)), float(np.clip(raw_y, 0.0, 1.0))

        feat = self._extract_polynomial_features(raw_x, raw_y, yaw, pitch)
        pred_x = float(np.dot(feat, self.coef_x))
        pred_y = float(np.dot(feat, self.coef_y))

        return float(np.clip(pred_x, 0.0, 1.0)), float(np.clip(pred_y, 0.0, 1.0))

    def save(self, filepath: str = None):
        """Saves calibration coefficients to JSON file."""
        if filepath is None:
            filepath = self.calib_file

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "is_fitted": self.is_fitted,
            "coef_x": self.coef_x.tolist() if self.coef_x is not None else None,
            "coef_y": self.coef_y.tolist() if self.coef_y is not None else None,
            "timestamp": time.time()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[GazeCalibrator] Calibration saved to {filepath}")

    def load(self, filepath: str = None):
        """Loads calibration coefficients from JSON file."""
        if filepath is None:
            filepath = self.calib_file

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Calibration file {filepath} not found.")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.is_fitted = data.get("is_fitted", False)
        self.coef_x = np.array(data["coef_x"], dtype=np.float64) if data.get("coef_x") else None
        self.coef_y = np.array(data["coef_y"], dtype=np.float64) if data.get("coef_y") else None


def run_calibration_ui(samples_per_point: int = 30, camera_id: int = 0, save_path: str = None):
    """
    Runs OpenCV interactive calibration UI over 3x3 grid points.
    """
    from gaze_estimator import GazeEstimator

    print("[Calibration UI] Initializing GazeEstimator...")
    estimator = GazeEstimator()
    calibrator = GazeCalibrator(calib_file=save_path)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Calibration UI] Error: Could not open camera {camera_id}.")
        return False

    cv2.namedWindow("Gaze Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    all_observations = []
    all_targets = []

    print("[Calibration UI] Starting calibration sequence. Look at red dots.")

    for pt_idx, target_pt in enumerate(GRID_POINTS_3X3):
        samples_collected = []
        start_time = time.time()

        while len(samples_collected) < samples_per_point:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural UI view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Process gaze
            result = estimator.process_frame(frame)

            # Target position in pixels
            target_pixel_x = int(target_pt[0] * w)
            target_pixel_y = int(target_pt[1] * h)

            # Draw UI
            frame_disp = frame.copy()

            # Background dim
            frame_disp = cv2.addWeighted(frame_disp, 0.4, np.zeros_like(frame_disp), 0.6, 0)

            # Target Dot
            color = (0, 0, 255) if result["face_detected"] else (0, 165, 255)
            cv2.circle(frame_disp, (target_pixel_x, target_pixel_y), 24, color, -1)
            cv2.circle(frame_disp, (target_pixel_x, target_pixel_y), 6, (255, 255, 255), -1)

            # Status overlay
            progress_pct = int((len(samples_collected) / samples_per_point) * 100)
            status_txt = f"Point {pt_idx + 1}/9: Look at the dot ({progress_pct}%)"
            cv2.putText(frame_disp, status_txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if not result["face_detected"]:
                cv2.putText(frame_disp, "No Face Detected!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Gaze Calibration", frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print("[Calibration UI] Calibration aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            # Collect sample if face is detected and confidence is good
            if result["face_detected"] and result["gaze_confidence"] > 0.2 and (time.time() - start_time > 0.5):
                samples_collected.append(result)

        # Filter point samples
        filtered = calibrator.filter_samples(samples_collected)
        all_observations.extend(filtered)
        all_targets.extend([target_pt] * len(filtered))

    cap.release()
    cv2.destroyAllWindows()

    print(f"[Calibration UI] Total samples collected: {len(all_observations)}")
    success = calibrator.fit(all_observations, all_targets)
    if success:
        calibrator.save()
        print("[Calibration UI] Calibration completed successfully!")
        return True
    else:
        print("[Calibration UI] Calibration fit failed!")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Gaze Calibration UI")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples per point")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    args = parser.parse_args()

    run_calibration_ui(samples_per_point=args.samples, camera_id=args.camera)
