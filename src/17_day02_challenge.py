''' 의사코드
1. challenge.jpg(ROI로 자른 이미지)를 읽는다.

2. [후보 비교] 여러 전처리 후보를 같은 이미지에 적용한다.
   none / Brightness / Gamma / CLAHE / Median

3. 각 후보에 Otsu(THRESH_BINARY)를 적용하고
   Otsu 값과 흰색 비율을 출력한다.

4. 후보 결과를 한 장(challenge_candidates.png)으로 저장한다.

5. 결과를 보고 METHOD에 최종 선택을 적는다.  ← 학생이 직접 결정

6. 선택한 전처리 결과와
   전처리 전·후 Otsu 결과를 저장한다.

이번 Challenge의 병뚜껑은 밝은 색이므로
THRESH_BINARY로 밝은 병뚜껑을 흰색 객체로 본다.
'''

from pathlib import Path

import cv2
import numpy as np


image_path = Path("samples/day02/challenge.jpg")
output_dir = Path("outputs/day02/challenge")
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 전처리 함수
# --------------------------------------------------

def to_gray(source):
    if len(source.shape) == 2:
        return source
    return cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)


def adjust_gamma(source, gamma):
    # 11_gamma.py와 같은 규칙: gamma > 1 → 밝아짐
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((v / 255.0) ** inv_gamma) * 255 for v in np.arange(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(source, table)


def apply_clahe(source, clip):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return clahe.apply(to_gray(source))


def otsu(source):
    value, binary = cv2.threshold(
        to_gray(source),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    white_ratio = float(np.count_nonzero(binary)) / binary.size
    return value, binary, white_ratio


# --------------------------------------------------
# 비교할 후보 (필요하면 값을 바꾸거나 추가)
# --------------------------------------------------

candidates = {
    "none":       lambda img: img.copy(),
    "beta_30":    lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=30),
    "gamma_1.2":  lambda img: adjust_gamma(img, 1.2),
    "gamma_1.5":  lambda img: adjust_gamma(img, 1.5),
    "gamma_2.0":  lambda img: adjust_gamma(img, 2.0),
    "clahe_2.0":  lambda img: apply_clahe(img, 2.0),
    "clahe_3.0":  lambda img: apply_clahe(img, 3.0),
    "median_3":   lambda img: cv2.medianBlur(img, 3),
}


# --------------------------------------------------
# TODO — 후보 비교 결과를 보고 최종 선택을 적는다.
#
# 1. 이미지의 문제를 판단합니다.
# 2. 필요한 전처리를 선택합니다.
# 3. Parameter를 직접 정합니다.
# 4. 아래 METHOD에 candidates의 키 이름을 적습니다.
#    (전처리가 필요 없다고 판단하면 "none")
# --------------------------------------------------

METHOD = "none"


# --------------------------------------------------
# 후보 비교 — Otsu 값과 흰색 비율 출력
# --------------------------------------------------

view_size = (240, 180)
tiles = []

print()
print(f"{'후보':<11} {'Otsu':>6} {'흰색비율':>8}")
print("-" * 28)

for name, func in candidates.items():
    processed = func(image)
    value, binary, ratio = otsu(processed)

    print(f"{name:<11} {value:>6.1f} {ratio:>8.3f}")

    top = cv2.cvtColor(
        cv2.resize(to_gray(processed), view_size, interpolation=cv2.INTER_AREA),
        cv2.COLOR_GRAY2BGR,
    )
    bottom = cv2.cvtColor(
        cv2.resize(binary, view_size, interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    tile = cv2.vconcat([top, bottom])
    cv2.rectangle(tile, (0, 0), (view_size[0], 24), (0, 0, 0), -1)
    cv2.putText(
        tile,
        f"{name}  T={value:.0f}",
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    tiles.append(tile)

# 4개씩 한 줄로 배치
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))

rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
candidates_path = output_dir / "challenge_candidates.png"
cv2.imwrite(str(candidates_path), cv2.vconcat(rows))

print("-" * 28)
print("후보 비교 저장:", candidates_path)
print("  위: 전처리 결과 / 아래: Otsu 결과")


# --------------------------------------------------
# 선택한 전처리 적용 및 결과 저장
# --------------------------------------------------

if METHOD not in candidates:
    raise ValueError(
        f"METHOD '{METHOD}'가 candidates에 없습니다: {list(candidates)}"
    )

result = candidates[METHOD](image)

result_path = output_dir / "challenge_result.jpg"
cv2.imwrite(str(result_path), result)

original_value, original_otsu, original_ratio = otsu(image)
result_value, result_otsu, result_ratio = otsu(result)

cv2.imwrite(str(output_dir / "challenge_original_otsu.jpg"), original_otsu)
cv2.imwrite(str(output_dir / "challenge_result_otsu.jpg"), result_otsu)

print()
print("선택한 전처리 :", METHOD)
print(f"전처리 전 Otsu : {original_value:.1f}  흰색비율 {original_ratio:.3f}")
print(f"전처리 후 Otsu : {result_value:.1f}  흰색비율 {result_ratio:.3f}")
print("입력 :", image_path)
print("결과 :", result_path)
print("Challenge 완료")
