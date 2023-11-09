import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import time
from show_video import ShowVideo
from calculator_media import Cal
from models.params import Params
from models.create_MobileNetV2_model import Create_model
from keyboard_controller import KeyboardController
import pyautogui

class Controller:
    def __init__(self, camera_num: int = 0, width_grid: int = 0, height_grid: int = 0):
        self.cap = cv2.VideoCapture(camera_num)
        self.cal = Cal(width_grid=width_grid, height_grid=height_grid)
        self.params = Params()
        self.model_path = "./checkpoints/checkpoint-0093.ckpt"
        # self.model_creator = Create_model()
        # self.model_out = self.model_creator.create_model()
        # self.model_out.load_weights(self.model_path)

    def control_screen(self):
        start_time = time.time()

        video = ShowVideo(cal=self.cal)
        # video.create_fullscreen_window()
        keyboard_controller = KeyboardController(self.cal)
        
        # for i in range(10):
        #     time.sleep(1)        
        #     pyautogui.press('enter')

        row, col = 0, 0
        cnt = 0
        while True:
            current_time = time.time()
            # 촬영 시작
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            # 추론 및 row, col return
            row, col = self.cal.output(frame=frame)

            # 화면 출력 및 마우스 출력
            video.show_frame(frame, row, col, ret)
            keyboard_controller.moving_for_galaga(row, col)


            col = (col + 1) % 3
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grid_webcam = Controller(camera_num=0, width_grid=3, height_grid=1)
    grid_webcam.control_screen()
