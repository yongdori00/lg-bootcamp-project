from calculator import Param
import pyautogui
import cv2

class MouseTracker:
    def __init__(self, cal: Param):
        self.cal = cal

    def move_cursor(self, row, col):
        screen_width, screen_height = pyautogui.size()
        x1, x2, y1, y2 = self.cal.calculate_cell_size(screen_width, screen_height, row, col)

        # 마우스 커서 이동
        pyautogui.moveTo((x2 + x1) / 2, (y2 + y1) / 2, duration=0.05)

        # 이동 후의 마우스 좌표 출력
        # print("마우스 현재 위치:", current_x, current_y)

    # def click_mouse():

    # def 