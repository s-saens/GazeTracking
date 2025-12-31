from __future__ import division
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request


class GazeTracking(object):

    # MediaPipe Face Mesh Landmarks' Indexes
    LEFT_EYE_INDEXES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_INDEXES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_IRIS_INDEXES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDEXES = [473, 474, 475, 476, 477]
    
    MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    def __init__(self, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.frame = None
        self.frame_height = None
        self.frame_width = None
        self.left_pupil = None
        self.right_pupil = None
        self.left_eye_region = None
        self.right_eye_region = None
        self.landmarks = None
        
        self._download_model()
        
        base_options = python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=max_faces,
            min_face_detection_confidence=detection_confidence,
            min_face_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @property
    def pupils_located(self):
        return self.left_pupil is not None and self.right_pupil is not None

    def _download_model(self):
        if not os.path.exists(self.MODEL_PATH):
            try:
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            except Exception as e:
                raise

    def _analyze(self):
        if not self.landmarks:
            return
        
        self.left_pupil = self._get_iris_center(self.LEFT_IRIS_INDEXES)
        self.right_pupil = self._get_iris_center(self.RIGHT_IRIS_INDEXES)
        self.left_eye_region = self._get_eye_region(self.LEFT_EYE_INDEXES)
        self.right_eye_region = self._get_eye_region(self.RIGHT_EYE_INDEXES)

    def _get_iris_center(self, iris_indexes):
        try:
            points = []
            for idx in iris_indexes:
                landmark = self.landmarks[idx]
                x = int(landmark.x * self.frame_width)
                y = int(landmark.y * self.frame_height)
                points.append((x, y))
            
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            return (center_x, center_y)
        except:
            return None

    def _get_eye_region(self, eye_indexes):
        try:
            points = []
            for idx in eye_indexes:
                landmark = self.landmarks[idx]
                x = int(landmark.x * self.frame_width)
                y = int(landmark.y * self.frame_height)
                points.append((x, y))
            
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            return (min(xs), min(ys), max(xs), max(ys))
        except:
            return None

    def refresh(self, frame):
        self.frame = frame
        self.frame_height, self.frame_width = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        
        self.left_pupil = None
        self.right_pupil = None
        self.left_eye_region = None
        self.right_eye_region = None
        self.landmarks = None
        
        if detection_result.face_landmarks:
            self.landmarks = detection_result.face_landmarks[0]
            self._analyze()

    def pupil_left_coords(self):
        return self.left_pupil

    def pupil_right_coords(self):
        return self.right_pupil

    def horizontal_ratio(self):
        if not self.pupils_located or not self.left_eye_region or not self.right_eye_region:
            return None
        
        left_eye_width = self.left_eye_region[2] - self.left_eye_region[0]
        left_pupil_offset = self.left_pupil[0] - self.left_eye_region[0]
        left_ratio = left_pupil_offset / left_eye_width if left_eye_width > 0 else 0.5
        
        right_eye_width = self.right_eye_region[2] - self.right_eye_region[0]
        right_pupil_offset = self.right_pupil[0] - self.right_eye_region[0]
        right_ratio = right_pupil_offset / right_eye_width if right_eye_width > 0 else 0.5
        
        return (left_ratio + right_ratio) / 2

    def vertical_ratio(self):
        if not self.pupils_located or not self.left_eye_region or not self.right_eye_region:
            return None
        
        left_eye_height = self.left_eye_region[3] - self.left_eye_region[1]
        left_pupil_offset = self.left_pupil[1] - self.left_eye_region[1]
        left_ratio = left_pupil_offset / left_eye_height if left_eye_height > 0 else 0.5
        
        right_eye_height = self.right_eye_region[3] - self.right_eye_region[1]
        right_pupil_offset = self.right_pupil[1] - self.right_eye_region[1]
        right_ratio = right_pupil_offset / right_eye_height if right_eye_height > 0 else 0.5
        
        return (left_ratio + right_ratio) / 2

    def is_right(self):
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.38

    def is_left(self):
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.62

    def is_center(self):
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_up(self):
        if self.pupils_located:
            return self.vertical_ratio() <= 0.42

    def is_down(self):
        if self.pupils_located:
            return self.vertical_ratio() >= 0.58

    def is_blinking(self):
        if not self.landmarks:
            return False
        
        try:
            left_ratio = self._eye_aspect_ratio(self.LEFT_EYE_INDEXES[:6])
            right_ratio = self._eye_aspect_ratio(self.RIGHT_EYE_INDEXES[:6])
            avg_ratio = (left_ratio + right_ratio) / 2
            return avg_ratio < 0.2
        except:
            return False

    def _eye_aspect_ratio(self, eye_indexes):
        points = []
        for idx in eye_indexes:
            landmark = self.landmarks[idx]
            x = landmark.x * self.frame_width
            y = landmark.y * self.frame_height
            points.append((x, y))
        
        vertical1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        vertical2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        ear = (vertical1 + vertical2) / (2.0 * horizontal) if horizontal > 0 else 0
        return ear

    def annotated_frame(self):
        frame = self.frame.copy()
        
        if self.pupils_located:
            if self.left_pupil:
                cv2.circle(frame, self.left_pupil, 3, (0, 255, 0), -1)
                cv2.circle(frame, self.left_pupil, 15, (0, 255, 0), 1)
            
            if self.right_pupil:
                cv2.circle(frame, self.right_pupil, 3, (0, 255, 0), -1)
                cv2.circle(frame, self.right_pupil, 15, (0, 255, 0), 1)
        
        if self.left_eye_region:
            x1, y1, x2, y2 = self.left_eye_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        if self.right_eye_region:
            x1, y1, x2, y2 = self.right_eye_region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        return frame

    def get_gaze_info(self):
        return {
            'pupils_located': self.pupils_located,
            'left_pupil': self.left_pupil,
            'right_pupil': self.right_pupil,
            'horizontal_ratio': self.horizontal_ratio(),
            'vertical_ratio': self.vertical_ratio(),
            'is_right': self.is_right(),
            'is_left': self.is_left(),
            'is_center': self.is_center(),
            'is_blinking': self.is_blinking()
        }

    def __del__(self):
        if hasattr(self, 'detector'):
            self.detector.close()

