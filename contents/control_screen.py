import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import time
from mouse_tracker import MouseTracker
from show_video import ShowVideo
from calculator_media import Cal
from models.params import Params
from models.create_MobileNetV2_model import Create_model
from keyboard_controller import KeyboardController

class Controller:
    def __init__(self, camera_num: int = 0, width_grid: int = 0, height_grid: int = 0):
        self.cap = cv2.VideoCapture(camera_num)
        self.cal = Cal(width_grid=width_grid, height_grid=height_grid)
        self.params = Params()
        self.model_path = "./checkpoints/checkpoint-0093.ckpt"
        self.model_creator = Create_model()
        self.model_out = self.model_creator.create_model()
        # self.model_out.load_weights(self.model_path)

    def control_screen(self):
        start_time = time.time()
        cur_col, cur_row = 0, 0

        video = ShowVideo(cal=self.cal, model_out=self.model_out)
        # video.create_fullscreen_window()
        cursor = MouseTracker(self.cal)
        keyboard_controller = KeyboardController(self.cal)
        
        row, col = 0, 0
        cnt = 0
        while True:
            current_time = time.time()
            # 촬영 시작
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            # 추론 및 row, col return
            row, col = self.cal.output(frame=frame, model_out=self.model_out)

            # 화면 출력 및 마우스 출력
            video.show_frame(frame, row, col, ret)
            # keyboard_controller.move(row, col)


            #여기는 마우스 컨트롤
            # cursor.move_cursor(row, col)

            # 가만히 있지 않으면 시간 초기화
            if cur_col != col or cur_row != row:
                start_time = current_time
            keyboard_controller.move_binary(row, col)

            # 3초 이상 머물러 있으면 동작
            if current_time - start_time >= 3 and (cur_col == col and cur_row == row):
                # 3초 이상 머물러 있으면 클릭
                cursor.click_cursor(row, col)
                # 3초 이상 머물러 있으면 양방향 중 하나로 이동
                keyboard_controller.move_binary(row, col)
                start_time = current_time
            cur_row, cur_col = row, col

            col = (col + 1) % 3
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grid_webcam = Controller(camera_num=0, width_grid=3, height_grid=1)
    grid_webcam.control_screen()
