import cv2
import os

# 특정 경로에서 이미지 불러오기
directory_path = 'images/'  # 이미지 파일 경로를 지정합니다.
cropped_path = 'cropped/'
image_directory = [f for f in os.listdir(directory_path)]
print(image_directory)

for height_directory_path in image_directory:
    height_directory = [f for f in os.listdir(directory_path + '/' + height_directory_path)]
    for width_directory_path in height_directory:
        width_directory = [f for f in os.listdir(directory_path + '/' + str(height_directory_path) + '/' + str(width_directory_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        # print(image_files)
        for i in width_directory:
            image = cv2.imread(directory_path + '/' + str(height_directory_path) + '/' + str(width_directory_path) + '/' + i)
            if image is None:
                print('이미지를 불러올 수 없습니다.')
            else:
                height, width, _ = image.shape
                center_x = width // 2
                center_y = height // 2
                major_axis = width // 6       # Adjust the major axis length
                minor_axis = height // 3  # Adjust the minor axis length
                angle = 0  # Adjust the angle if needed
                # 이미지 처리 (예: 자르기, 변환 등)
                cropped_image = image[center_y - minor_axis: center_y + minor_axis, center_x - major_axis:center_x + major_axis]

                os.makedirs(cropped_path + str(height_directory_path) + "/" + str(width_directory_path), exist_ok=True)
                # 처리된 이미지를 저장할 경로와 파일명을 지정합니다.
                output_path = cropped_path + str(height_directory_path) + "/" + str(width_directory_path) + "/" + i
                
                # 자른 이미지를 저장합니다.
                cv2.imwrite(output_path, cropped_image)