import cv2
import numpy as np

# 이미지 읽기
image = cv2.imread(r"C:\Users\na062\Desktop\week4\project\test_data\9.jpg")

# HSV 색공간 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 피코의 HSV 범위 설정 (밝은 녹색)
lower_pico = np.array([35,80, 80])  # 조정 가능
upper_pico = np.array([85, 195, 195])  # 조정 가능

# 마스크 생성 (피코 영역 추출)
mask_pico = cv2.inRange(hsv, lower_pico, upper_pico)

# Morphological Transformations으로 노이즈 제거
kernel = np.ones((10, 10), np.uint8)  # 더 큰 커널 적용
mask_pico = cv2.morphologyEx(mask_pico, cv2.MORPH_CLOSE, kernel)
mask_pico = cv2.morphologyEx(mask_pico, cv2.MORPH_OPEN, kernel)

# 윤곽선 검출
contours, _ = cv2.findContours(mask_pico, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 큰 윤곽선 병합
min_x, min_y = float('inf'), float('inf')
max_x, max_y = float('-inf'), float('-inf')

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # 작은 윤곽선 무시
    if w * h > 1000:  # 최소 넓이 조건 (조정 가능)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

# 원본 이미지 위에 병합된 박스 그리기
if min_x < max_x and min_y < max_y:  # 유효한 박스가 있을 경우
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)

# 결과 출력 (원본 이미지 위에 박스 표시)
cv2.imshow("Pico Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
