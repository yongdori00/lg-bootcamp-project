import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
from mouse_tracker import MouseTracker
from show_video import ShowVideo
from calculator import Cal
from models.params import Params
from models.create_model import Create_model

class Controller:
    def __init__(self, camera_num: int = 0, width_grid: int = 0, height_grid: int = 0):
        self.cap = cv2.VideoCapture(camera_num)
        self.cal = Cal(width_grid=width_grid, height_grid=height_grid)
        self.params = Params()
        self.model_path = "./data/checkpoint-0030.ckpt.index"
        self.model_creator = Create_model()
        self.model_out = self.model_creator.create_model()
        # self.model_out.load_weights(self.model_path)

    def control_screen(self):
        cursor = MouseTracker(self.cal)
        video = ShowVideo(cal=self.cal, model_out=self.model_out)
        video.create_fullscreen_window()
        
        highlighted_grid = 0
        print("highlighted_grid:", highlighted_grid)
        cnt = 0
        while True:
            # 촬영 시작
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            # 추론 및 row, col return
            row, col = self.cal.output(frame, model_out=self.model_out)

            # 화면 출력 및 마우스 출력
            video.show_frame(frame, row, col, ret)
            cursor.move_cursor(row, col)
            
            cnt += 1
            if cnt > 30:
                cnt = 0
                # highlighted_grid += 1
                highlighted_grid = video._random_integer_between_0_and_num()
                print("highlighted_grid:", highlighted_grid)
                if highlighted_grid >= self.cal.height_grid*self.cal.width_grid:
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grid_webcam = Controller(camera_num=1, width_grid=8, height_grid=6)
    grid_webcam.control_screen()
