import cv2
from mouse_tracker import MouseTracker
from show_video import ShowVideo
from calculator import Param

class Controller:
    def __init__(self, width_grid: int = 0, height_grid: int = 0):
        self.cal = Param(width_grid=width_grid, height_grid=height_grid)

    def control_screen(self):
        cursor = MouseTracker(self.cal)
        video = ShowVideo(camera_num=1, cal=self.cal)
        video.create_fullscreen_window()
        
        highlighted_grid = 0
        print("highlighted_grid:", highlighted_grid)
        cnt = 0
        while True:
            row = highlighted_grid // self.cal.width_grid
            col = highlighted_grid % self.cal.width_grid

            video.show_frame(row, col)
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

        video.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    grid_webcam = Controller(width_grid=8, height_grid=6)
    grid_webcam.control_screen()
