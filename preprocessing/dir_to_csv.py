import os
import csv

# 디렉토리 구조를 스캔하고 CSV 파일로 저장하는 함수
def create_csv_from_directory(directory_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['name', 'row', 'column'])  # CSV 파일 헤더 작성

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory_path)
                    path_parts = relative_path.split(os.path.sep)

                    if len(path_parts) >= 3:
                        first_subdir = path_parts[0]
                        second_subdir = path_parts[1]
                        png_file = path_parts[2]

                        csv_writer.writerow([f"{first_subdir}/{second_subdir}/{png_file}", first_subdir, second_subdir])

if __name__ == '__main__':
    directory_path = 'new_cropped'  # 디렉토리 경로를 변경하세요.
    output_csv = 'output.csv'  # CSV 파일 이름을 변경하세요.

    create_csv_from_directory(directory_path, output_csv)
    print(f"CSV 파일이 생성되었습니다: {output_csv}")
