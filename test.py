import cv2

# 얼굴과 눈을 검출하는 Haar 캐스케이드 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 웹캠에서 비디오 캡처를 시작
cap = cv2.VideoCapture(1)

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4)
    
    # 검출된 얼굴 주변에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 눈 검출
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # 양쪽 눈의 좌표 얻기
        left_eye = None
        right_eye = None
        for (ex, ey, ew, eh) in eyes:
            if ex < w // 2:  # 양쪽 눈 중 왼쪽 눈
                left_eye = (x + ex, y + ey, ew, eh)
            else:  # 양쪽 눈 중 오른쪽 눈
                right_eye = (x + ex, y + ey, ew, eh)
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # 양쪽 눈과 그 사이 전체 영역을 이미지로 따기
        if left_eye is not None and right_eye is not None:
            x, y, w, h = left_eye[0], left_eye[1], right_eye[0] + right_eye[2] - left_eye[0], left_eye[3]
            eye_area = frame[y:y+h, x:x+w]
            
            # 캡처된 영역을 이미지로 저장
            cv2.imwrite('eye_area.png', eye_area)
    
    # 결과 화면에 표시
    cv2.imshow('Eye Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체와 창 닫기
cap.release()
cv2.destroyAllWindows()