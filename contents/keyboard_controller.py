from calculator_media import Cal
import pyautogui

class KeyboardController:
    def __init__(self, cal: Cal):
        self.cal = cal

    def move(self, row, col):
        directions = self.cal.calculate_direction(row, col)
        print(directions)
        for direction in directions:
            pyautogui.press(direction)

    
    def move_binary(self, row, col):
        directions = self.cal.calculate_direction(row, col)
        print(directions)
        for direction in directions:
            if direction == "left" or direction == "right":
                pyautogui.press(direction)

    def moving_for_galaga(self, row, col):
        directions = self.cal.calculate_direction(row, col)
        print(directions)
        pyautogui.press('z')
        for direction in directions:
            if direction == "left" or direction == "right":
                pyautogui.press(direction)