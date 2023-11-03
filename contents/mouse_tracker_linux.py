from calculator import Param
from Xlib import X, display

class MouseTracker:
    def __init__(self, cal: Param):
        self.cal = cal
        self.disp = display.Display()

    def move_cursor(self, row, col):
        screen_width, screen_height = self.root.get_geometry().width, root.get_geometry().height
        x1, x2, y1, y2 = self.cal.calculate_cell_size(screen_width, screen_height, row, col)

        self.disp = display.Display()

        # 현재 마우스 위치 가져오기
        root = self.disp.screen().root
        root.warp_pointer((x2 + x1) / 2, (y2 + y1) / 2)  # (x, y)로 마우스 이동
        self.disp.sync()

        # 이동 후의 마우스 좌표 출력
        # print("마우스 현재 위치:", current_x, current_y)

    def click_mouse():
        # 마우스 왼쪽 버튼 클릭
        root.button_press(1)
        disp.sync()

        # 마우스 왼쪽 버튼 떼기
        root.button_release(1)
        disp.sync()

    def 