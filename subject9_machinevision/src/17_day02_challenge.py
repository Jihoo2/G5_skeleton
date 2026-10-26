from pathlib import Path

import cv2
import numpy as np

image_path = Path(
    "samples/day02/challenge.jpg"
)

output_dir = Path(
    "outputs/day02/challenge"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# TODO
#
# 1. 이미지의 문제를 판단합니다.
# 2. 필요한 전처리를 선택합니다.
# 3. Parameter를 직접 정합니다.
# 4. result에 최종 결과를 저장합니다.
# --------------------------------------------------



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# CLAHE 파라미터 비교
for clip_limit in [1.0, 2.0, 3.0]:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    trial = clahe.apply(gray)

    cv2.imwrite(
        str(output_dir / f"clahe_{clip_limit}.jpg"),
        trial
    )

    _, mask = cv2.threshold(
        trial, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        str(output_dir / f"otsu_clahe_{clip_limit}.jpg"),
        mask
    )


# --------------------------------------------------
# 추가 실험 1: Gamma Correction
# --------------------------------------------------

gamma_results = {}

for gamma in [1.2, 1.5]:

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)

    gamma_result = cv2.LUT(gray, table)
    gamma_results[gamma] = gamma_result

    cv2.imwrite(
        str(output_dir / f"gamma_{gamma}.jpg"),
        gamma_result
    )

    _, mask = cv2.threshold(
        gamma_result, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    cv2.imwrite(
        str(output_dir / f"otsu_gamma_{gamma}.jpg"),
        mask
    )


# --------------------------------------------------
# 추가 실험 2: Gaussian Blur
# --------------------------------------------------

gaussian_result = cv2.GaussianBlur(
    gray, (5, 5), 0
)

cv2.imwrite(
    str(output_dir / "gaussian_5.jpg"),
    gaussian_result
)

_, mask = cv2.threshold(
    gaussian_result, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(output_dir / "otsu_gaussian_5.jpg"),
    mask
)

# 최종 판단: 전처리 미사용
# 이유: 실험한 전처리 모두 병뚜껑과 배경의
# Otsu 분리를 충분히 개선하지 못함
# Gaussian 5x5는 3일차 추가 확인 후보

result = image.copy()



# --------------------------------------------------
# 전처리 결과 저장
# --------------------------------------------------

result_path = (
    output_dir
    / "challenge_result.jpg"
)

cv2.imwrite(
    str(result_path),
    result,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

def to_gray(source):

    if len(source.shape) == 2:
        return source

    return cv2.cvtColor(
        source,
        cv2.COLOR_BGR2GRAY,
    )


original_gray = to_gray(
    image
)

result_gray = to_gray(
    result
)


# --------------------------------------------------
# 전처리 전 Otsu
#
# 이번 Challenge에서는
# 밝은 병뚜껑을 흰색 객체로 확인합니다.
# --------------------------------------------------

_, original_otsu = cv2.threshold(
    original_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


# --------------------------------------------------
# 전처리 후 Otsu
# --------------------------------------------------

_, result_otsu = cv2.threshold(
    result_gray,
    0,
    255,
    cv2.THRESH_BINARY
    + cv2.THRESH_OTSU,
)


cv2.imwrite(
    str(
        output_dir
        / "challenge_original_otsu.jpg"
    ),
    original_otsu,
)

cv2.imwrite(
    str(
        output_dir
        / "challenge_result_otsu.jpg"
    ),
    result_otsu,
)


print("입력 :", image_path)
print("결과 :", result_path)
print("Challenge 완료")