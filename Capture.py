import cv2
import os
import time
import numpy as np
from plyer import notification  # plyer 라이브러리를 임포트합니다

# 웹캠을 엽니다
cap = cv2.VideoCapture(1)

frame_count = 0
capture_interval = 1  # 1초에 한 번씩 캡처합니다
width_grid, height_grid = list(map(int, input("가로 그리드 갯수, 세로 그리드 갯수를 입력하세요.").split()))

start_time = time.time()

cv2.namedWindow('Grid Webcam', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty('Grid Webcam', cv2.WND_PROP_TOPMOST, 1)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 255, 255)  # 텍스트 색상 (파란색)
thickness = 2  # 텍스트 두께

for i in range(3, 0, -1):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "starts after {} seconds".format(i)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)
    cv2.imshow('Grid Webcam', image)
    cv2.waitKey(1000)  # 1초 대기

while True:
    current_time = time.time()
    # 웹캠에서 프레임을 읽어옵니다
    ret, frame = cap.read()
    #빈 화면을 복사합니다.
    frame_copy = frame.copy()
    # 10x10 격자를 그립니다]
    height, width, _ = frame.shape
    cell_size_x = width // width_grid
    cell_size_y = height // height_grid

    # 프레임을 저장할 디렉토리를 지정합니다
    output_directory = "images/" + str(frame_count % (width_grid * height_grid) + 1)
    os.makedirs(output_directory, exist_ok=True)

    # 격자에 번호를 추가합니다
    for i in range(height_grid):
        for j in range(width_grid):
            # 격자의 중심 좌표 계산
            center_x = j * cell_size_x + cell_size_x // 2
            center_y = i * cell_size_y + cell_size_y // 2

            # 격자 번호를 표시할 위치 계산
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'{i * width_grid + j + 1}'
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

            # 격자 중심에 번호를 추가합니다
            if frame_count == width_grid * i + (j + 1):
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, lineType=cv2.LINE_AA)
            else:
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)


    # 수평선 그리기
    for i in range(1, height_grid):
        cv2.line(frame, (0, i * cell_size_y), (width, i * cell_size_y), (0, 255, 0), 1)

    # 수직선 그리기
    for i in range(1, width_grid):
        cv2.line(frame, (i * cell_size_x, 0), (i * cell_size_x, height), (0, 255, 0), 1)

    # 화면에 프레임을 표시합니다.

    cv2.imshow('Grid Webcam', frame)


    # 프레임을 캡처해서 저장합니다 (1초에 한 번)
    if current_time - start_time >= 1:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_directory, f"frame_{timestamp}.png")
        cv2.imwrite(output_filename, frame_copy)
        print(f"Captured frame saved: {output_filename}")

        # 캡처될 때마다 알림을 띄웁니다
        notification_title = "Frame Captured"
        notification_message = f"Captured frame saved: {output_filename}"
        notification.notify(
            title=notification_title,
            message=notification_message,
            app_name='Grid Webcam App',  # 알림에서 표시될 앱 이름
            timeout=1  # 알림이 표시될 시간 (초)
        )
        start_time = current_time
        frame_count += 1

    # 'q' 키를 누르면 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count == (width_grid * height_grid):
        break

# 웹캠을 해제하고 창을 닫습니다
cap.release()
cv2.destroyAllWindows()
