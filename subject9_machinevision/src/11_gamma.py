''' 의사코드
1. Gamma 결과를 저장할 폴더를 준비한다.

2. dark.jpg를 읽어본다.

3. dark.jpg가 존재하는지 확인한다.

4. dark.jpg가 없다면
   cap_on_reference.jpg를 읽는다.

5. 기준 이미지의 밝기를 낮춰
   dark.jpg를 자동으로 만든다.

6. Gamma Correction 함수를 준비한다.

7. Gamma 값이 0보다 큰지 확인한다.

8. 0~255의 픽셀값을
   Gamma 공식으로 변환하는
   Lookup Table을 만든다.

9. Gamma 값을 여러 개 준비한다.

   0.6
   0.8
   1.0
   1.2
   1.5
   2.0

10. 첫 번째 Gamma 값을
    dark.jpg에 적용한다.

11. 결과 이미지를 저장한다.

12. 나머지 Gamma 값도
    같은 방법으로 반복한다.

13. 모든 결과 이미지를 비교한다.

14. 어두운 영역은 잘 보이면서
    밝은 영역의 정보가
    지나치게 사라지지 않는지 확인한다.
'''

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

dark_path = Path(
    "samples/day02/dark.jpg"
)

output_dir = Path(
    "outputs/day02/gamma"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# dark.jpg 확인
# --------------------------------------------------

image = cv2.imread(
    str(dark_path)
)


# dark.jpg가 없다면
# 기준 ROI 이미지에서 자동으로 생성
if image is None:

    print(
        "dark.jpg가 없어 기준 이미지에서 생성합니다."
    )

    reference = cv2.imread(
        str(reference_path)
    )

    if reference is None:
        raise FileNotFoundError(
            f"기준 이미지를 읽지 못했습니다: "
            f"{reference_path}"
        )

    image = cv2.convertScaleAbs(
        reference,
        alpha=0.55,
        beta=0,
    )

    cv2.imwrite(
        str(dark_path),
        image,
    )

    print(
        "문제 이미지 생성:",
        dark_path,
    )


# --------------------------------------------------
# Gamma Correction 함수
# --------------------------------------------------

def adjust_gamma(
    source: np.ndarray,
    gamma: float,
) -> np.ndarray:

    if gamma <= 0:
        raise ValueError(
            "gamma는 0보다 커야 합니다."
        )

    inv_gamma = 1.0 / gamma

    table = np.array(
        [
            (
                (value / 255.0)
                ** inv_gamma
            ) * 255
            for value in np.arange(256)
        ],
        dtype=np.uint8,
    )

    return cv2.LUT(
        source,
        table,
    )


# --------------------------------------------------
# 여러 Gamma 값 비교
# --------------------------------------------------

gamma_values = [
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    2.0,
]


for gamma in gamma_values:

    result = adjust_gamma(
        image,
        gamma,
    )

    output_path = (
        output_dir
        / f"gamma_{gamma}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "저장:",
        output_path,
    )