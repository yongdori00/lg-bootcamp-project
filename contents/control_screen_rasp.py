import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
import random
from models.params import Params


class Controller:
    def __init__(self, camera_num: int = 0, width_grid: int = 0, height_grid: int = 0):
        self.cap = cv2.VideoCapture(camera_num)
        self.width_grid, self.height_grid = width_grid, height_grid
        self.params = Params()
        self.model_path = "./tfliteConvert/MobileNetV2_tfliteModel.tflite"
        self.interpreter=tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

    def create_fullscreen_window(self):
        cv2.namedWindow('Grid Webcam', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_TOPMOST, 1)
    
    def _add_grid(self, frame, height, width):
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        # 수평선 그리기
        for i in range(1, self.height_grid):
            cv2.line(frame, (0, i * cell_size_y), (width, i * cell_size_y), (0, 255, 0), 1)

        # 수직선 그리기
        for i in range(1, self.width_grid):
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
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        x1 = col * cell_size_x
        x2 = (col + 1) * cell_size_x
        y1 = row * cell_size_y
        y2 = (row + 1) * cell_size_y

        # 투명도를 조절하기 위해 알파 채널 추가
        alpha_channel = color[3]
        color_bgr = color[:3]
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)  # 빨간색 사각형 그리기
        cv2.addWeighted(overlay, alpha_channel / 255, frame, 1 - alpha_channel / 255, 0, frame)

    def _add_number_in_grid(self, frame, height, width):
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        # 격자에 번호를 추가합니다
        for i in range(self.height_grid):
            for j in range(self.width_grid):
                # 격자의 중심 좌표 계산
                center_x = j * cell_size_x + cell_size_x // 2
                center_y = i * cell_size_y + cell_size_y // 2

                # 격자 번호를 표시할 위치 계산
                font_scale = 0.5
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f'{i * self.width_grid + j}'
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                # cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    def _random_integer_between_0_and_num(self):
        num = (self.height_grid*self.width_grid) - 1
        if num < 0:
            return None  # 유효하지 않은 입력 처리

        random_num = random.randint(0, num)
        return random_num

    def control_screen(self, init_label):
        highlighted_grid = init_label
        # print("highlighted_grid:", highlighted_grid)
        cnt = 0
        while True:
        # for _ in range(3):
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                continue
            print('next frame')
            frame_copy = frame.copy()
            height, width, _ = frame_copy.shape
            center_x = width // 2
            center_y = height // 2
            major_axis = width // 6       # Adjust the major axis length
            minor_axis = height // 3  # Adjust the minor axis length
            angle = 0  # Adjust the angle if needed
            input_image = frame_copy[center_y - minor_axis: center_y + minor_axis, center_x - major_axis:center_x + major_axis]
            #print('org', input_image.shape)
            input_image = cv2.resize(input_image,(self.width, self.height))
            input_image = np.expand_dims(input_image, axis=0)
            if self.floating_model == True :     # 트레이닝된 모델이 float32면 해당 타입으로 캐스팅
                input_image = (np.float32(input_image))
            #print('height', self.height)
            #print('width', self.width)
            #print('A', input_image.shape)
            #print('B', self.input_details[0])
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            #print(self.input_details[0])
            self.interpreter.invoke()
            #print(self.output_details[0])
            col_out = self.interpreter.get_tensor(self.output_details[0]['index'])
            #print(y_out1.shape)
            row_out = self.interpreter.get_tensor(self.output_details[0]['index']+1)
            #print(y_out2.shape)
            col = col_out[0]
            #print(col)
            row = row_out[0]
            
            self._add_grid(frame, frame.shape[0], frame.shape[1])
            # self._add_number_in_grid(frame, frame.shape[0], frame.shape[1])
            self._add_ellipse(frame, frame.shape[0], frame.shape[1])

            # 그리드 하이라이트
            row = np.argmax(row)
            col = np.argmax(col)
            self.highlight_grid(frame, row, col)  # 함수 호출로 그리드 하이라이트

            cv2.imshow('Grid Webcam', frame)
            # cnt += 1
            # if cnt > 30:
            #     cnt = 0
            #     # highlighted_grid += 1
            #     highlighted_grid = self._random_integer_between_0_and_num()
            #     print("highlighted_grid:", highlighted_grid)
            #     if highlighted_grid >= self.height_grid*self.width_grid:
            #         break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grid_webcam = Controller(camera_num=0, width_grid=8, height_grid=6)
    grid_webcam.create_fullscreen_window()
    grid_webcam.control_screen(0)
