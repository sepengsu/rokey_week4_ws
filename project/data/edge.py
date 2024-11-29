import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread(r"C:\Users\na062\Desktop\week4\project\data\4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 블러링과 밝은 회색 조정
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 밝은 회색으로 변환
alpha = 1.5  # 밝기 계수 (1.0보다 크면 밝아짐)
beta = 50    # 픽셀 값에 추가할 상수
bright_blur = cv2.addWeighted(blur, alpha, blur, 0, beta)

# 에지 검출
edges = cv2.Canny(bright_blur, 50, 150)

# Hough Line Transform으로 선 검출
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# 이미지 중심 계산
height, width = image.shape[:2]
center_x, center_y = width // 2, height // 2
ignore_radius = 200  # 중심 근처를 무시하는 반경 (조정 가능)

# 검출된 선을 원본 이미지 위에 그림
output_image = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # 선 길이 계산

        # 선의 중점 계산
        line_center_x = (x1 + x2) // 2
        line_center_y = (y1 + y2) // 2

        # 중심 근처 선 필터링
        distance_to_center = np.sqrt((line_center_x - center_x) ** 2 + (line_center_y - center_y) ** 2)
        if distance_to_center > ignore_radius and length > 150:  # 중심 근처 제외 및 긴 선 필터링
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 녹색 선으로 표시

# 결과 시각화
cv2.imshow("Long Lines", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
