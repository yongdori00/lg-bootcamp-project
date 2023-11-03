import pyautogui

# # 현재 마우스의 위치를 출력
# print(pyautogui.position())

# # (x, y) 좌표로 마우스 이동
# pyautogui.moveTo(100, 100, duration=1)  # 1초 동안 (100, 100) 좌표로 이동

# # 마우스 클릭 (왼쪽 버튼)
# pyautogui.click(100, 100)  # (100, 100) 좌표에서 마우스 클릭

# # 오른쪽 버튼 클릭
# pyautogui.rightClick(100, 100)

# # 더블 클릭
# pyautogui.doubleClick(100, 100)

# # 마우스 드래그 (시작 좌표에서 끝 좌표까지)
# pyautogui.dragTo(200, 200, duration=1)  # 1초 동안 (100, 100)에서 (200, 200)으로 드래그

# 마우스 스크롤
while True:
    pyautogui.hscroll(10)  # 위로 10만큼 스크롤
pyautogui.scroll(-10)  # 아래로 10만큼 스크롤
