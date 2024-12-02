import cv2
import numpy as np


# 피코의 HSV 범위 설정 (밝은 녹색)
class Cropper:
    def __init__(self):
        self.lower_pico = np.array([35, 100,100])  # 조정 가능
        self.upper_pico = np.array([85, 190, 190])  # 조정 가능
    
    def __call__(self,image):
        self.image = image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.mask_pico = cv2.inRange(hsv, self.lower_pico, self.upper_pico)
        self.cropping()

    def cropping(self):
        image = self.image
        kernel = np.ones((10, 10), np.uint8)  # 더 큰 커널 적용
        mask_pico = cv2.morphologyEx(self.mask_pico, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(mask_pico, 50, 150)  # 임계값 조정 가능

        # 윤곽선 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 이미지 크기 가져오기
        height, width = image.shape[:2]

        # 큰 윤곽선 병합
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # 작은 윤곽선 무시 및 외곽 제외
            if w * h > 1000:  # 최소 넓이 조건 (조정 가능)
                # 외곽에 닿아 있는 윤곽선 제외
                if x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height:
                    continue

                # 박스를 병합하기 위해 최소/최대 좌표 계산
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

        # 원본 이미지 위에 병합된 박스 그리기 (+10 확장)
        if min_x < max_x and min_y < max_y:  # 유효한 박스가 있을 경우
            min_x = max(min_x - 30, 0)  # 좌표 확장 (좌측 경계)
            min_y = max(min_y - 30, 0)  # 좌표 확장 (상단 경계)
            max_x = min(max_x + 30, width)  # 좌표 확장 (우측 경계)
            max_y = min(max_y + 30, height)  # 좌표 확장 (하단 경계)
            cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
        cropped_image = image[min_x:max_x,min_y:max_y]
        return cropped_image
