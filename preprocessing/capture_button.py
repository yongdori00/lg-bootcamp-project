import cv2
import os
import time
import numpy as np

from datetime import datetime
from plyer import notification


class GridWebcamCapture:
    def __init__(self, camera_num: int=0, width_grid: int=0, height_grid: int=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.num_zone = 0
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
        for _ in range(wait_time, 0, -1):
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
        h = self.num_zone//self.width_grid
        w = self.num_zone%self.width_grid
        self._output_directory = "images/" + str("{}/{}".format(h, w))
        print("set_dir", self._output_directory)
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
                text = f'{i * self.width_grid + j}'
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2

                # 격자 중심에 번호를 추가합니다
                if self.num_zone == self.width_grid * i + j:
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

    def _add_ellipse(self, frame, height, width):
        center_x = width // 2
        center_y = height // 2
        major_axis = width // 6     # Adjust the major axis length
        minor_axis = height // 3    # Adjust the minor axis length
        angle = 0                   # Adjust the angle if needed
        cv2.ellipse(frame, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, (255, 0, 0), 2)

    def _get_time(self):  
        # 현재 시간을 초 단위로 가져오기
        current_time = time.time()
        # 현재 시간을 datetime 객체로 변환
        current_datetime = datetime.fromtimestamp(current_time)

        # 년도, 월, 일, 시, 분, 초 가져오기
        year = current_datetime.year
        month = current_datetime.month
        day = current_datetime.day
        hour = current_datetime.hour
        minute = current_datetime.minute
        second = current_datetime.second
        # 밀리초 계산 (밀리초 단위로 반올림)
        milliseconds = int(current_time % 1 * 10000)

        # 파일 이름 생성
        timestamp = f"{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}_{milliseconds:03d}"
        return timestamp


    def capture_frames(self, name: str, capture_idx:int):
        self.num_zone = capture_idx
        if self.num_zone >= self.height_grid*self.width_grid:
            return print("그리드 갯수-1 보다 작은 값을 입력하시오.(0~{})".format(self.width_grid*self.height_grid-1))
        elif self.num_zone < 0:
            return print("0보다 큰 값을 입력하시오.")
        self._set_dir()
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)  # 좌우반전
            if not ret:
                continue
            frame_copy = frame.copy()
            height, width, _ = frame.shape
            self._add_grid(frame, height, width)
            self._add_number_in_grid(frame, height, width)
            self._add_ellipse(frame, height, width)
        
            # 화면에 프레임을 표시합니다.
            cv2.imshow('Grid Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print(self._get_time())
                timestamp = self._get_time()
                output_filename = os.path.join(self._output_directory, f"{name}_{timestamp}.png")
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
                print("capture {}".format(self.num_zone))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if cv2.waitKey(1) & 0xFF == ord('u'):
                self.num_zone += 1
                self._set_dir()

            if self.num_zone == (self.width_grid * self.height_grid):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    grid_webcam = GridWebcamCapture(camera_num=1, width_grid=8, height_grid=6)
    grid_webcam.create_fullscreen_window()
    # grid_webcam.initial_screen()
    # grid_webcam.show_adjust_face_position(2)
    grid_webcam.capture_frames(name="hyungoo", capture_idx=33)
