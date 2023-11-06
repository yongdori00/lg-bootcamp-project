import cv2
import os
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 특정 경로에서 이미지 불러오기
directory_path = 'images/'  # 이미지 파일 경로를 지정합니다.
cropped_path = 'cropped_eye/'
image_directory = [f for f in os.listdir(directory_path)]
print(image_directory)

for height_directory_path in image_directory:
    height_directory = [f for f in os.listdir(directory_path + '/' + height_directory_path)]
    for width_directory_path in height_directory:
        width_directory = [f for f in os.listdir(directory_path + '/' + str(height_directory_path) + '/' + str(width_directory_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        # print(image_files)
        for i in width_directory:
            image = cv2.imread(directory_path + '/' + str(height_directory_path) + '/' + str(width_directory_path) + '/' + i)
            # 얼굴 검출
            if image is None:
                print('이미지를 불러올 수 없습니다.')
            else:
                with mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.7
                 ) as face_detection:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detections:
                        for detection in results.detections:
                            mp_drawing.draw_detection(image, detection)

                    cv2.imshow("whow", cv2.resize(image, None, fx=0.5, fy=0.5))

                    keypoints = detection.location_data.relative_keypoints
                    right_eye = keypoints[0]
                    left_eye = keypoints[1]

                    h, w, _ = image.shape
                    right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                    left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                    
                    right_eye_image = image[right_eye[1] - 10:right_eye[1] + 10, right_eye[0] - 25:right_eye[0] + 25]
                    left_eye_image = image[left_eye[1] - 10:left_eye[1] + 10, left_eye[0] - 25:left_eye[0] + 25]

                    print(right_eye_image.shape)
                # cv2.rectangle(image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

                # 두 이미지를 수평으로 합칩니다.
                combined_image = cv2.hconcat([right_eye_image, left_eye_image])

                os.makedirs(cropped_path + str(height_directory_path) + "/" + str(width_directory_path), exist_ok=True)
                # 처리된 이미지를 저장할 경로와 파일명을 지정합니다.
                output_path = cropped_path + str(height_directory_path) + "/" + str(width_directory_path) + "/" + i
        
                # 캡처된 영역을 이미지로 저장
                cv2.imwrite(output_path, combined_image)


                