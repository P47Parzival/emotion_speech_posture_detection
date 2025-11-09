import dlib
import numpy as np

class LandmarkExtractor:
    def __init__(self, face_detector, landmark_predictor, upsample=0):
        self.detector = face_detector
        self.predictor = landmark_predictor
        self.upsample = upsample

    def detect_landmarks(self, gray_image):
        rects = self.detector(gray_image, self.upsample)
        if len(rects) == 0:
            return None
        # use the largest face
        rect = max(rects, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray_image, rect)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
        return pts
