import cv2


class Camera:
    """
    A simple, reusable camera interface using OpenCV VideoCapture.
    """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        """
        Open the webcam and verify initialization.
        Returns True if successful, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"Error: Unable to open camera at index {self.camera_index}")
                return False
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def read_frame(self):
        """
        Reads a frame from the webcam.
        Returns (ret, frame) where ret is boolean indicating success and frame is the image frame.
        """
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera is not open.")
            return False, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("Warning: Failed to capture frame.")
                return False, None
            return ret, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None

    def get_info(self):
        """
        Returns resolution (width, height) and FPS of the camera stream.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return {"width": width, "height": height, "fps": fps}

    def release(self):
        """
        Cleanly releases the VideoCapture resource.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
