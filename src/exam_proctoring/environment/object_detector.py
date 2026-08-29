import os
import time
import cv2
import torch
from ultralytics import YOLO

# Exam environment relevant COCO classes
DEFAULT_RELEVANT_CLASSES = {
    "cell phone",
    "book",
    "laptop",
    "person",
    "backpack",
    "keyboard",
    "mouse",
    "remote",
    "tv",
    "bottle",
}


class ObjectDetector:
    """
    Real-time Object Detector wrapping Ultralytics YOLO for online exam proctoring.
    Performs CPU/GPU inference, formats structured detection dictionaries,
    and draws clean bounding box overlays.
    """

    def __init__(
        self,
        model_path: str = "models/pretrained/environment/yolov8n.pt",
        confidence_threshold: float = 0.40,
        relevant_classes: set = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.relevant_classes = (
            relevant_classes if relevant_classes is not None else DEFAULT_RELEVANT_CLASSES
        )

        # Detect GPU/CPU hardware
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve fallback to standard yolov8n.pt
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"

        print(
            f"Loading ObjectDetector model from '{model_path}' on device '{self.device}' (conf={self.confidence_threshold})..."
        )
        self.model = YOLO(model_path)

    def detect(self, frame_bgr, frame_index: int = 0):
        """
        Run object detection on an OpenCV BGR frame.

        Returns:
            dict with fields:
                - 'timestamp': float
                - 'frame_index': int
                - 'inference_time_ms': float
                - 'detections': list of detection dicts:
                    [
                        {
                            "class_id": int,
                            "class_name": str,
                            "confidence": float,
                            "bbox": [x1, y1, x2, y2],
                            "is_relevant": bool,
                            "timestamp": float,
                            "frame_index": int
                        }, ...
                    ]
        """
        start_time = time.time()
        timestamp = time.time()

        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "inference_time_ms": 0.0,
                "detections": [],
            }

        # Run YOLO inference
        results = self.model(
            frame_bgr, conf=self.confidence_threshold, device=self.device, verbose=False
        )

        inference_time_ms = (time.time() - start_time) * 1000.0
        detections = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0].item())
                    # bbox coordinates: [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    bbox = [int(val) for val in xyxy]

                    is_relevant = cls_name.lower() in [c.lower() for c in self.relevant_classes]

                    detections.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 4),
                            "bbox": bbox,
                            "is_relevant": is_relevant,
                            "timestamp": timestamp,
                            "frame_index": frame_index,
                        }
                    )

        return {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "inference_time_ms": round(inference_time_ms, 2),
            "detections": detections,
        }

    def draw_detections(self, frame_bgr, detection_output):
        """
        Draw bounding boxes and class labels onto a copy of frame_bgr.
        """
        if frame_bgr is None:
            return None

        annotated_frame = frame_bgr.copy()
        detections = detection_output.get("detections", [])

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Distinct colors: Red/Orange for phone/book/relevant objects, Cyan for others
            if det["class_name"].lower() in ["cell phone", "book"]:
                color = (0, 0, 255)  # Red
                thickness = 3
            elif det["is_relevant"]:
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (255, 255, 0)  # Cyan
                thickness = 2

            # Bounding box rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Label box background and text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated_frame,
                (x1, max(0, y1 - text_height - 6)),
                (x1 + text_width + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 3, max(text_height + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated_frame
