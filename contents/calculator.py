import numpy as np

class Cal:
    def __init__(self, width_grid, height_grid):
        self.width_grid, self.height_grid = width_grid, height_grid

    def calculate_cell_size(self, width, height, row, col):
        cell_size_x = width // self.width_grid
        cell_size_y = height // self.height_grid
        x1 = col * cell_size_x
        x2 = (col + 1) * cell_size_x
        y1 = row * cell_size_y
        y2 = (row + 1) * cell_size_y

        return x1, x2, y1, y2
    
    def output(self, frame, model_out):
        #crop 및 결과
        frame_copy = frame.copy()
        height, width, _ = frame_copy.shape
        center_x = width // 2
        center_y = height // 2
        major_axis = width // 6       # Adjust the major axis length
        minor_axis = height // 3  # Adjust the minor axis length
        # angle = 0  # Adjust the angle if needed
        input_image = frame_copy[center_y - minor_axis: center_y + minor_axis, center_x - major_axis:center_x + major_axis]
        input_image = np.expand_dims(input_image, axis=0)
        y_out = model_out.predict(input_image)
        row = y_out[0]
        col = y_out[1]

        # 그리드 하이라이트 위치
        row = np.argmax(row)
        col = np.argmax(col)

        return row, col
    
