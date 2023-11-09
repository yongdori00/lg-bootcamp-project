import numpy as np
import mediapipe as mp
import cv2

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class Cal:
    def __init__(self, width_grid, height_grid):
        self.width_grid, self.height_grid = width_grid, height_grid

    def calculate_cell_size(self, width, height, row, col):
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        x1 = col * cell_size_x
        x2 = (col + 1) * cell_size_x
        y1 = row * cell_size_y
        y2 = (row + 1) * cell_size_y

        return x1, x2, y1, y2
    
    def calculate_direction(self, row, col):
        directions = []
        if 0 <= row / self.height_grid and row / self.height_grid < 0.33:
            directions.append("down")
        elif 0.66 < row / self.height_grid and row / self.height_grid <= 1:
            directions.append("up")
        if 0 <= col / self.width_grid and col / self.width_grid < 0.33:
            directions.append("left")
        elif 0.66 < col / self.width_grid and col / self.width_grid <= 1:
            directions.append("right")

        return directions
    
    def output(self, frame=None, model_out=None):
        #crop 및 결과
        # row, col = self.crop_image(frame, model_out)
        
        return 0, 0
        #return row, col
    
    def crop_image(self, frame, model_out):
        frame_copy = frame.copy()
        right_eye_image, left_eye_image = None, None
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=1
            ) as face_detection:
            frame_copy.flags.writeable = False
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_copy)

            frame_copy.flags.writeable = True
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    # mp_drawing.draw_detection(frame_copy, detection)

                    keypoints = detection.location_data.relative_keypoints
                    right_eye = keypoints[0]
                    left_eye = keypoints[1]

                    h, w, _ = frame_copy.shape
                    right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                    left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                    
                    right_eye_image = frame_copy[right_eye[1] - 30:right_eye[1] + 30, right_eye[0] - 40:right_eye[0] + 40]
                    left_eye_image = frame_copy[left_eye[1] - 30:left_eye[1] + 30, left_eye[0] - 40:left_eye[0] + 40]
                    print(left_eye, right_eye)

        if right_eye_image is None or left_eye_image is None:
            return 0, 0
        
        # 두 이미지를 수평으로 합칩니다.
        combined_image = cv2.hconcat([right_eye_image, left_eye_image])
        
        if combined_image is None:
            print('Image load failed')
            return 0, 0

        expanded_combined_image = np.expand_dims(combined_image, axis=0)
        y_out = model_out.predict(expanded_combined_image)
        # cv2.imshow('Grid Webcam', combined_image)

        # row = y_out[0]
        row = 0
        col = y_out[0]
        # col = y_out[1]

        # 그리드 하이라이트 위치
        # row = np.argmax(row)
        row = 0
        col = np.argmax(col)
        
        return row, col