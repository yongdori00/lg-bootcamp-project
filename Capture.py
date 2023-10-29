import cv2
import os
import time
from plyer import notification  # plyer 라이브러리를 임포트합니다

# 웹캠을 엽니다
cap = cv2.VideoCapture(0)

# 프레임을 저장할 디렉토리를 지정합니다
output_directory = "captured_frames"
os.makedirs(output_directory, exist_ok=True)

frame_count = 0
capture_interval = 1  # 1초에 한 번씩 캡처합니다

while True:
    # 웹캠에서 프레임을 읽어옵니다
    ret, frame = cap.read()
    # 10x10 격자를 그립니다
    grid_size = 10
    height, width, _ = frame.shape
    cell_size_x = width // grid_size
    cell_size_y = height // grid_size

    # 격자에 번호를 추가합니다
    for i in range(grid_size):
        for j in range(grid_size):
            # 격자의 중심 좌표 계산
            center_x = j * cell_size_x + cell_size_x // 2
            center_y = i * cell_size_y + cell_size_y // 2

            # 격자 번호를 표시할 위치 계산
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f'{i * grid_size + j}'
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2

            # 격자 중심에 번호를 추가합니다
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)


    # 수평선 그리기
    for i in range(1, grid_size):
        cv2.line(frame, (0, i * cell_size_y), (width, i * cell_size_y), (0, 255, 0), 1)

    # 수직선 그리기
    for i in range(1, grid_size):
        cv2.line(frame, (i * cell_size_x, 0), (i * cell_size_x, height), (0, 255, 0), 1)

    # 화면에 프레임을 표시합니다
    cv2.imshow('Grid Webcam', frame)

    # 프레임을 캡처해서 저장합니다 (1초에 한 번)
    if frame_count % capture_interval == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_directory, f"frame_{timestamp}.png")
        cv2.imwrite(output_filename, frame)
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

    frame_count += 1

    time.sleep(1)

    # 'q' 키를 누르면 종료합니다
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠을 해제하고 창을 닫습니다
cap.release()
cv2.destroyAllWindows()
