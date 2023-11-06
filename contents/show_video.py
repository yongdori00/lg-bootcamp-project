import random
import cv2
from calculator import Cal
import numpy as np

class ShowVideo:
    def __init__(self, cal: Cal = None, model_out=None):
        self.cal = cal
        self.model_out = model_out
        
    def show_frame(self, frame, row, col, ret):
        if not ret:
            return
        
        
        self._add_grid(frame, frame.shape[0], frame.shape[1])
        # self._add_number_in_grid(frame, frame.shape[0], frame.shape[1])
        self._add_ellipse(frame, frame.shape[0], frame.shape[1])

        # 그리드 하이라이트
        self.highlight_grid(frame, row, col)  # 함수 호출로 그리드 하이라이트
        cv2.imshow('Grid Webcam', frame)
            
    
    def create_fullscreen_window(self):
        cv2.namedWindow('Grid Webcam', cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_TOPMOST, 1)

    def _add_grid(self, frame, height, width):
        cell_size_x = width // self.cal.width_grid
        cell_size_y = height // self.cal.height_grid
        # 수평선 그리기
        for i in range(1, self.cal.height_grid):
            cv2.line(frame, (0, i * cell_size_y), (width, i * cell_size_y), (0, 255, 0), 1)

        # 수직선 그리기
        for i in range(1, self.cal.width_grid):
            cv2.line(frame, (i * cell_size_x, 0), (i * cell_size_x, height), (0, 255, 0), 1)

    def _add_ellipse(self, frame, height, width):
        center_x = width // 2
        center_y = height // 2
        major_axis = width // 6
        minor_axis = height // 3
        angle = 0
        cv2.ellipse(frame, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, (255, 0, 0), 2)

    def highlight_grid(self, frame, row, col, color=(0, 0, 255, 128)):
        height, width, _ = frame.shape
        x1, x2, y1, y2 = self.cal.calculate_cell_size(width, height, row, col)

        # 투명도를 조절하기 위해 알파 채널 추가
        alpha_channel = color[3]
        color_bgr = color[:3]
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)  # 빨간색 사각형 그리기
        cv2.addWeighted(overlay, alpha_channel / 255, frame, 1 - alpha_channel / 255, 0, frame)

    def _add_number_in_grid(self, frame, height, width):
        cell_size_x = width // self.cal.width_grid
        cell_size_y = height // self.cal.height_grid
        # 격자에 번호를 추가합니다
        for i in range(self.cal.height_grid):
            for j in range(self.cal.width_grid):
                # 격자의 중심 좌표 계산
                center_x = j * cell_size_x + cell_size_x // 2
                center_y = i * cell_size_y + cell_size_y // 2

                # 격자 번호를 표시할 위치 계산
                font_scale = 0.5
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'{i * self.cal.width_grid + j}'
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                # cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    def _random_integer_between_0_and_num(self):
        num = (self.cal.height_grid*self.cal.width_grid) - 1
        if num < 0:
            return None  # 유효하지 않은 입력 처리

        random_num = random.randint(0, num)
        return random_num