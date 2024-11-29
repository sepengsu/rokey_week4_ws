import numpy as np

def align_coordinates(coords1, coords2):
    # 중심 계산
    center1 = np.mean(coords1, axis=0)
    center2 = np.mean(coords2, axis=0)
    coords1_centered = coords1 - center1
    coords2_centered = coords2 - center2

    # 스케일 계산 및 정규화
    scale1 = np.sqrt(np.sum(coords1_centered**2) / coords1.shape[0])
    scale2 = np.sqrt(np.sum(coords2_centered**2) / coords2.shape[0])
    coords1_normalized = coords1_centered / scale1
    coords2_normalized = coords2_centered / scale2

    # 회전 계산 (SVD)
    H = coords1_normalized.T @ coords2_normalized
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    # 변환 적용
    coords1_transformed = scale2 * (coords1_normalized @ R.T) + center2
    return coords1_transformed

# Test
