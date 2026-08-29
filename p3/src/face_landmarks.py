import cv2
import mediapipe as mp

# Defined constants for MediaPipe 478-landmark FaceMesh model (with refined iris landmarks)
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]

LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
LEFT_IRIS_CENTER_INDEX = 468

RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
RIGHT_IRIS_CENTER_INDEX = 473

# Common key facial feature points (useful for visualization or head pose)
KEY_FACE_INDICES = {
    "nose_tip": 4,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}


class FaceLandmarkDetector:
    """
    Real-time face and iris landmark detector using MediaPipe FaceMesh.
    Currently configured for single-examinee tracking (max_num_faces = 1).
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.detector = self.mp_face_mesh.FaceMesh(
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        """
        Process a single OpenCV BGR image frame.

        Returns:
            dict with fields:
                - 'face_detected': bool
                - 'num_faces': int (total faces detected by model)
                - 'landmarks': list of (x, y, z) normalized coordinates for primary face, or None
                - 'raw_results': raw MediaPipe output object
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "face_detected": False,
                "num_faces": 0,
                "landmarks": None,
                "raw_results": None,
            }

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Performance optimization: set writeable flag to False
        frame_rgb.flags.writeable = False
        results = self.detector.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            return {
                "face_detected": False,
                "num_faces": 0,
                "landmarks": None,
                "raw_results": results,
            }

        num_faces = len(results.multi_face_landmarks)
        # Use primary (first) face landmarks
        primary_face = results.multi_face_landmarks[0]

        # Convert landmarks to list of tuples (x, y, z)
        landmarks = [(lm.x, lm.y, lm.z) for lm in primary_face.landmark]

        return {
            "face_detected": True,
            "num_faces": num_faces,
            "landmarks": landmarks,
            "raw_results": results,
        }

    def get_eye_and_iris_landmarks(self, landmarks, frame_width: int, frame_height: int):
        """
        Extract pixel coordinates for left eye, right eye, left iris, right iris.
        """
        if landmarks is None or len(landmarks) == 0:
            return None

        def to_pixel(idx):
            if idx < len(landmarks):
                x, y, _ = landmarks[idx]
                return (int(x * frame_width), int(y * frame_height))
            return None

        left_eye_pts = [to_pixel(idx) for idx in LEFT_EYE_INDICES]
        right_eye_pts = [to_pixel(idx) for idx in RIGHT_EYE_INDICES]
        left_iris_pts = [to_pixel(idx) for idx in LEFT_IRIS_INDICES]
        right_iris_pts = [to_pixel(idx) for idx in RIGHT_IRIS_INDICES]

        left_iris_center = to_pixel(LEFT_IRIS_CENTER_INDEX)
        right_iris_center = to_pixel(RIGHT_IRIS_CENTER_INDEX)

        return {
            "left_eye": [pt for pt in left_eye_pts if pt is not None],
            "right_eye": [pt for pt in right_eye_pts if pt is not None],
            "left_iris": [pt for pt in left_iris_pts if pt is not None],
            "right_iris": [pt for pt in right_iris_pts if pt is not None],
            "left_iris_center": left_iris_center,
            "right_iris_center": right_iris_center,
        }

    def draw_landmarks(self, frame_bgr, detection_result, draw_iris_highlight=True):
        """
        Draw face mesh tesselation and highlight iris landmarks on frame_bgr.
        """
        if not detection_result["face_detected"] or detection_result["raw_results"] is None:
            return frame_bgr

        annotated_frame = frame_bgr.copy()
        h, w, _ = frame_bgr.shape

        for face_landmarks in detection_result["raw_results"].multi_face_landmarks:
            # Draw facial mesh mesh contours
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )
            # Draw eye contours
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
            )

            if self.refine_landmarks:
                # Draw iris connections
                self.mp_drawing.draw_landmarks(
                    image=annotated_frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        if draw_iris_highlight and detection_result["landmarks"]:
            landmarks = detection_result["landmarks"]
            eye_iris_info = self.get_eye_and_iris_landmarks(landmarks, w, h)
            if eye_iris_info:
                # Draw explicit bright circles on left and right iris centers
                if eye_iris_info["left_iris_center"]:
                    cv2.circle(
                        annotated_frame, eye_iris_info["left_iris_center"], 3, (0, 255, 255), -1
                    )
                if eye_iris_info["right_iris_center"]:
                    cv2.circle(
                        annotated_frame, eye_iris_info["right_iris_center"], 3, (0, 255, 255), -1
                    )

        return annotated_frame

    def close(self):
        """Release MediaPipe FaceMesh resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
