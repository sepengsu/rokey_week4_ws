import cv2
 # 이미지 불러오

img_path =r'C:\Users\na062\Desktop\week4\project\52.jpg'
img = cv2.imread(img_path)
 # 박스 칠 좌표 설정 (예: 좌측 상단 (50, 50), 우측 하단 (200, 200))
start_point = (50, 50)  # 박스 시작 좌표 (x, y)
end_point = (200, 200)  # 박스 끝 좌표 (x, y)
color = (0, 255, 0)  # BGR 색상 (초록색)
thickness = 2  # 박스 선의 두께
# 박스 그리기
cv2.rectangle(img, start_point, end_point, color, thickness)
 # 텍스트 설정
text = "Hello, OpenCV!"  # 추가할 텍스트
position = (50, 50)  # 텍스트 시작 위치 (x, y)
font = cv2.FONT_HERSHEY_SIMPLEX  # 글꼴 설정
font_scale = 1  # 글자 크기
color = (0, 255, 0)  # BGR 색상 (초록색)
thickness = 2  # 글자 두께
# 텍스트 추가
cv2.putText(img, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
     # 이미지 출력
cv2.imshow('image', img)
cv2.waitKey(0)