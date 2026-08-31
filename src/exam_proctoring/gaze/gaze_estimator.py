"""
Gaze Estimator using MediaPipe Face Landmarker.

Extracts facial and iris landmarks, estimates iris positions relative to eye geometry,
computes head yaw and pitch angles, and produces raw/calibrated gaze predictions
along with confidence metrics.
"""

import os
import sys
import time
import urllib.request
import math
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Landmark Indices for MediaPipe Face Mesh (478 total landmarks)
# Left Eye
LEFT_IRIS_CENTER = 468
LEFT_IRIS_PERIMETER = [469, 470, 471, 472]
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

# Right Eye
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_PERIMETER = [474, 475, 476, 477]
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Canonical 3D Face Model Points for solvePnP head pose estimation (in mm)
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
    (0.0, -330.0, -65.0),        # Chin (landmark 152)
    (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 33)
    (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 263)
    (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 61)
    (150.0, -150.0, -125.0)      # Right mouth corner (landmark 291)
], dtype=np.float64)

FACE_2D_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def get_default_model_path() -> str:
    """Returns absolute path to default face landmarker task model file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(base_dir, "models", "pretrained", "gaze", "face_landmarker.task")
    return model_path


def ensure_model_file(model_path: str = None) -> str:
    """Ensures face_landmarker.task exists, downloading if necessary."""
    if model_path is None:
        model_path = get_default_model_path()
    
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"[GazeEstimator] Downloading face landmarker model to {model_path}...")
        try:
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
            print("[GazeEstimator] Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe model from {DEFAULT_MODEL_URL}: {e}")
    return model_path


class GazeEstimator:
    """
    Estimates eye gaze coordinates and head pose from webcam or image frames
    using MediaPipe Face Landmarker Tasks API.
    """

    def __init__(self, model_path: str = None, num_faces: int = 1, calibrator=None):
        self.model_path = ensure_model_file(model_path)
        self.calibrator = calibrator

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> dict:
        """
        Process a single BGR image frame.

        Returns dictionary with:
            - timestamp: float
            - gaze_x: float | None (calibrated or raw screen x in [0, 1])
            - gaze_y: float | None (calibrated or raw screen y in [0, 1])
            - raw_gaze_x: float | None
            - raw_gaze_y: float | None
            - gaze_confidence: float in [0.0, 1.0]
            - head_yaw: float | None (degrees)
            - head_pitch: float | None (degrees)
            - face_detected: bool
        """
        if timestamp is None:
            timestamp = time.time()

        empty_result = {
            "timestamp": timestamp,
            "gaze_x": None,
            "gaze_y": None,
            "raw_gaze_x": None,
            "raw_gaze_y": None,
            "gaze_confidence": 0.0,
            "head_yaw": None,
            "head_pitch": None,
            "face_detected": False,
            "landmarks_3d": None,
        }

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return empty_result

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return empty_result

        # Convert BGR to RGB for MediaPipe Image
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.landmarker.detect(mp_image)
        except Exception:
            return empty_result

        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return empty_result

        # Process primary face landmarks
        face_landmarks = detection_result.face_landmarks[0]
        num_landmarks = len(face_landmarks)

        # Convert landmarks to list of 3D tuples (x, y, z)
        landmarks_3d = [(lm.x, lm.y, lm.z) for lm in face_landmarks]

        # 1. Compute Head Pose (Yaw and Pitch in degrees)
        head_yaw, head_pitch = self._estimate_head_pose(landmarks_3d, w, h, detection_result)

        # 2. Extract Eye Geometry & Iris Ratios
        left_eye_info = self._process_eye(
            landmarks_3d, LEFT_EYE_OUTER, LEFT_EYE_INNER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_IRIS_CENTER
        ) if num_landmarks > LEFT_IRIS_CENTER else None

        right_eye_info = self._process_eye(
            landmarks_3d, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_IRIS_CENTER
        ) if num_landmarks > RIGHT_IRIS_CENTER else None

        # Combine eye observations
        raw_gaze_x, raw_gaze_y, eye_confidence = self._combine_eye_ratios(left_eye_info, right_eye_info, head_yaw, head_pitch)

        # 3. Overall Confidence Calculation
        confidence = self._compute_confidence(
            face_detected=True,
            eye_confidence=eye_confidence,
            head_yaw=head_yaw,
            head_pitch=head_pitch
        )

        if raw_gaze_x is None or raw_gaze_y is None or confidence < 0.1:
            gaze_x, gaze_y = None, None
            raw_gaze_x, raw_gaze_y = None, None
        else:
            # Clamp raw observation to [0, 1]
            raw_gaze_x = float(np.clip(raw_gaze_x, 0.0, 1.0))
            raw_gaze_y = float(np.clip(raw_gaze_y, 0.0, 1.0))

            # Apply calibrator if present and fitted
            if self.calibrator is not None and getattr(self.calibrator, "is_fitted", False):
                cal_x, cal_y = self.calibrator.predict(raw_gaze_x, raw_gaze_y, head_yaw, head_pitch)
                gaze_x = float(np.clip(cal_x, 0.0, 1.0))
                gaze_y = float(np.clip(cal_y, 0.0, 1.0))
            else:
                gaze_x = raw_gaze_x
                gaze_y = raw_gaze_y

        return {
            "timestamp": timestamp,
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "raw_gaze_x": raw_gaze_x,
            "raw_gaze_y": raw_gaze_y,
            "gaze_confidence": float(confidence),
            "head_yaw": float(head_yaw) if head_yaw is not None else None,
            "head_pitch": float(head_pitch) if head_pitch is not None else None,
            "face_detected": True,
            "landmarks_3d": landmarks_3d,
        }

    def _estimate_head_pose(self, landmarks_3d: list, img_w: int, img_h: int, detection_result) -> tuple:
        """Estimates head yaw and pitch angles in degrees."""
        # Try matrix option first if available
        if (hasattr(detection_result, "facial_transformation_matrixes") and
                detection_result.facial_transformation_matrixes and
                len(detection_result.facial_transformation_matrixes) > 0):
            try:
                matrix = np.array(detection_result.facial_transformation_matrixes[0])
                r_mat = matrix[:3, :3]
                yaw = math.atan2(r_mat[0, 2], r_mat[2, 2]) * 180.0 / math.pi
                pitch = math.atan2(-r_mat[1, 2], math.sqrt(r_mat[1, 1]**2 + r_mat[1, 0]**2)) * 180.0 / math.pi
                return yaw, pitch
            except Exception:
                pass

        # Fallback / robust solvePnP
        try:
            image_points = []
            for idx in FACE_2D_LANDMARK_IDS:
                lm = landmarks_3d[idx]
                image_points.append([lm[0] * img_w, lm[1] * img_h])
            image_points = np.array(image_points, dtype=np.float64)

            focal_length = img_w
            center = (img_w / 2.0, img_h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                FACE_3D_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                r_mat, _ = cv2.Rodrigues(rot_vec)
                # Compute Euler angles
                sy = math.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    pitch = math.atan2(r_mat[2, 1], r_mat[2, 2]) * 180.0 / math.pi
                    yaw = math.atan2(-r_mat[2, 0], sy) * 180.0 / math.pi
                else:
                    pitch = math.atan2(-r_mat[1, 2], r_mat[1, 1]) * 180.0 / math.pi
                    yaw = math.atan2(-r_mat[2, 0], sy) * 180.0 / math.pi
                return yaw, pitch
        except Exception:
            pass

        return 0.0, 0.0

    def _process_eye(self, landmarks_3d: list, left_idx: int, right_idx: int, top_idx: int, bottom_idx: int, iris_idx: int) -> dict:
        """Calculates normalized iris ratio and eye aspect ratio (EAR)."""
        p_left = np.array(landmarks_3d[left_idx][:2])
        p_right = np.array(landmarks_3d[right_idx][:2])
        p_top = np.array(landmarks_3d[top_idx][:2])
        p_bottom = np.array(landmarks_3d[bottom_idx][:2])
        p_iris = np.array(landmarks_3d[iris_idx][:2])

        eye_width = np.linalg.norm(p_right - p_left)
        eye_height = np.linalg.norm(p_bottom - p_top)

        if eye_width < 1e-5 or eye_height < 1e-5:
            return None

        ear = eye_height / eye_width

        # Project iris center onto eye bounding vectors
        # Horizontal ratio along left-to-right vector
        lr_vec = p_right - p_left
        iris_lr_vec = p_iris - p_left
        horiz_ratio = np.dot(iris_lr_vec, lr_vec) / (eye_width ** 2)

        # Vertical ratio along top-to-bottom vector
        tb_vec = p_bottom - p_top
        iris_tb_vec = p_iris - p_top
        vert_ratio = np.dot(iris_tb_vec, tb_vec) / (eye_height ** 2)

        return {
            "ratio_x": float(horiz_ratio),
            "ratio_y": float(vert_ratio),
            "ear": float(ear),
        }

    def _combine_eye_ratios(self, left_eye: dict, right_eye: dict, yaw: float, pitch: float) -> tuple:
        """Combines left and right eye iris ratios into a normalized gaze observation."""
        ratios_x = []
        ratios_y = []
        ears = []

        if left_eye is not None and left_eye["ear"] > 0.12:
            ratios_x.append(left_eye["ratio_x"])
            ratios_y.append(left_eye["ratio_y"])
            ears.append(left_eye["ear"])

        if right_eye is not None and right_eye["ear"] > 0.12:
            ratios_x.append(right_eye["ratio_x"])
            ratios_y.append(right_eye["ratio_y"])
            ears.append(right_eye["ear"])

        if not ratios_x:
            return None, None, 0.0

        mean_ratio_x = float(np.mean(ratios_x))
        mean_ratio_y = float(np.mean(ratios_y))
        eye_confidence = min(1.0, float(np.mean(ears)) * 4.0)

        # Combine iris displacement with head orientation offset
        # Head yaw offset shift: ~ 15 deg yaw corresponds to ~ 0.3 screen offset
        head_yaw_offset = (yaw if yaw is not None else 0.0) / 45.0
        head_pitch_offset = -(pitch if pitch is not None else 0.0) / 45.0

        raw_x = mean_ratio_x + 0.5 * head_yaw_offset
        raw_y = mean_ratio_y + 0.5 * head_pitch_offset

        return raw_x, raw_y, eye_confidence

    def _compute_confidence(self, face_detected: bool, eye_confidence: float, head_yaw: float, head_pitch: float) -> float:
        """Computes frame-level gaze confidence score in [0.0, 1.0]."""
        if not face_detected:
            return 0.0

        conf = eye_confidence

        # Deduct confidence for extreme head rotation
        if head_yaw is not None and abs(head_yaw) > 35.0:
            conf *= max(0.2, 1.0 - (abs(head_yaw) - 35.0) / 30.0)
        if head_pitch is not None and abs(head_pitch) > 30.0:
            conf *= max(0.2, 1.0 - (abs(head_pitch) - 30.0) / 30.0)

        return float(np.clip(conf, 0.0, 1.0))
