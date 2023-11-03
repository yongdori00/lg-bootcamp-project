class Param:
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