import os
import shutil

# 이동할 디렉토리 리스트
list_ = [[0,1,2],[0,1,2]]

# 원본 디렉토리 경로 설정
source_directory = './cropped'

# 대상 디렉토리 경로 설정 (복사해서 붙여넣을 디렉토리)
destination_directory = './new_cropped'

# 원본 디렉토리의 파일 및 디렉토리 리스트 불러오기
file_list = []
dir_list = []

for height_path in os.listdir(source_directory):
    ind_x = 0
    ind_y = 0
    for width_path in os.listdir(source_directory + "/" + height_path):
        height = int(height_path)
        width = int(width_path)
        if 0 <= height and height < 3:
            ind_y = 0
        elif 3 <= height and height < 6:
            ind_y = 1
        else:
            print("세로가 0~5 중에 하나가 아닙니다.")

        if 0 <= width and width < 3:
            ind_x = 0
        elif 3 <= width and width < 5:
            ind_x = 1
        elif 6 <= width and width < 8:
            ind_x = 2
        else:
            print("가로가 0~7 중에 하나가 아닙니다.")
        print("height: ", height, "width: ", width)
        print("ind_y: ", ind_y, "ind_x: ", ind_x, end="\n\n")

        source_full_path = os.path.join(source_directory, height_path, width_path)
        destination_full_path = os.path.join(destination_directory, str("{}/{}".format(ind_y, ind_x)))
        print("set_dir", destination_full_path)
        os.makedirs(destination_full_path, exist_ok=True)

        # 대상 디렉토리에 파일 복사
        for file_name in os.listdir(source_full_path):
            source_file = os.path.join(source_full_path, file_name)
            destination_file = os.path.join(destination_full_path, file_name)
            shutil.copy2(source_file, destination_file)

print("파일 및 디렉토리가 복사되었습니다.")