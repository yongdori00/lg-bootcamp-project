import cv2
import os
import time
import numpy as np
from plyer import notification


class GridWebcamCapture:
    def __init__(self, camera_num: int=0, width_grid: int=0, height_grid: int=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.frame_count = 0
        self.capture_interval = 1       # 1초에 한 번씩 캡처
        self.width_grid, self.height_grid = width_grid, height_grid
        self._check_and_get_grid_dimensions()
        self._output_directory = ""

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 0.5
        self._font_color = (0, 255, 255)  # 텍스트 색상 (파란색)
        self._thickness = 2  # 텍스트 두께
        
    def _check_and_get_grid_dimensions(self):
        if self.width_grid==0 and self.height_grid==0:
            self.width_grid, self.height_grid = list(map(int, input("가로 그리드 갯수, 세로 그리드 갯수를 입력하세요: ").split()))
        else:
            pass

    def create_fullscreen_window(self):
        cv2.namedWindow('Grid Webcam', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_TOPMOST, 1)

    def initial_screen(self, wait_time: int=2):
        for i in range(wait_time, 0, -1):
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            text = "Camera Loading..."
            text_size = cv2.getTextSize(text, self._font, self._font_scale, self._thickness)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            text_y = (image.shape[0] + text_size[1]) // 2
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow('Grid Webcam', image)
            cv2.waitKey(1000)       # 1초 대기
    
    def show_adjust_face_position(self, wait_time: int=3):
        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            height, width, _ = frame.shape
            frame = cv2.flip(frame, 1)  # 좌우반전
            self._add_ellipse(frame, height, width)
            
            elapsed_time = time.time() - start_time
            remaining_time = int(wait_time - elapsed_time)+1
            text = "Starts after {} seconds".format(remaining_time)
            text_size = cv2.getTextSize(text, self._font, self._font_scale, self._thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow('Grid Webcam', frame)
            if cv2.waitKey(1) & 0xFF != 255:  # 사용자가 아무 키나 누르면 종료
                break
            if time.time() - start_time >= wait_time:
                break

    def _set_dir(self):
        self._output_directory = "images/" + str(self.frame_count % (self.width_grid * self.height_grid) + 1)
        os.makedirs(self._output_directory, exist_ok=True)

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
                text = f'{i * self.width_grid + j + 1}'
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2

                # 격자 중심에 번호를 추가합니다
                if self.frame_count == self.width_grid * i + (j + 1):
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)
                else:
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

    def _add_grid(self, frame, height, width):
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        # 수평선 그리기
        for i in range(1, self.height_grid):
            cv2.line(frame, (0, i * cell_size_y), (width, i * cell_size_y), (0, 255, 0), 1)

        # 수직선 그리기
        for i in range(1, self.width_grid):
            cv2.line(frame, (i * cell_size_x, 0), (i * cell_size_x, height), (0, 255, 0), 1)

    def _add_circle(self, frame, height, width):
        center_x = width // 2
        center_y = height // 2
        radius = width // 4
        cv2.circle(frame, (center_x, center_y), radius, (255, 0, 0), 2)

    def _add_ellipse(self, frame, height, width):
        center_x = width // 2
        center_y = height // 2
        major_axis = width // 6     # Adjust the major axis length
        minor_axis = height // 3    # Adjust the minor axis length
        angle = 0                   # Adjust the angle if needed
        cv2.ellipse(frame, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, (255, 0, 0), 2)

    def capture_frames(self, capture_delay_time: float=1):
        is_waited = False
        start_time = time.time()
        while True:
            self._set_dir()
            current_time = time.time()
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)  # 좌우반전
            if not ret:
                continue
            frame_copy = frame.copy()
            height, width, _ = frame.shape
            self._add_grid(frame, height, width)
            self._add_number_in_grid(frame, height, width)
            # self._add_circle(frame, height, width)
            self._add_ellipse(frame, height, width)
        
            # 화면에 프레임을 표시합니다.
            cv2.imshow('Grid Webcam', frame)

            # 캡처 하기
            if is_waited:
                if current_time - start_time >= capture_delay_time:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_filename = os.path.join(self._output_directory, f"frame_{timestamp}.png")
                    cv2.imwrite(output_filename, frame_copy)
                    print(f"Captured frame saved: {output_filename}")

                    notification_title = "Frame Captured"
                    notification_message = f"Captured frame saved: {output_filename}"
                    notification.notify(
                        title=notification_title,
                        message=notification_message,
                        app_name='Grid Webcam App',
                        timeout=1
                    )
                    start_time = current_time
                    is_waited = False
                    print("capture")
            else:
                if current_time - start_time >= capture_delay_time:
                    start_time = current_time
                    self.frame_count += 1
                    is_waited = True
                    print("wait")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if self.frame_count == (self.width_grid * self.height_grid):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    grid_webcam = GridWebcamCapture(camera_num=1, width_grid=8, height_grid=6)
    # grid_webcam.create_fullscreen_window()
    grid_webcam.initial_screen()
    grid_webcam.show_adjust_face_position(5)
    grid_webcam.capture_frames(capture_delay_time=2)
