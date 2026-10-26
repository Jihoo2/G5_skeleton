''' 의사코드
1. 1일차에서 만든
   cap_on_reference.jpg를 읽는다.

2. 기준 이미지가 정상적으로
   읽혔는지 확인한다.

3. 기준 이미지의 Contrast를 낮춘다.

4. 저대비 결과를
   low_contrast.jpg로 저장한다.

5. 기준 이미지와
   저대비 이미지를 Gray로 변환한다.

6. 두 이미지를 저장하여
   Contrast가 실제로 낮아졌는지 비교한다.

7. CLAHE에서 사용할
   clipLimit 값을 준비한다.

   1.0
   2.0
   3.0
   4.0

8. tileGridSize는 8×8로 고정한다.

9. low_contrast_gray에
   첫 번째 CLAHE 설정을 적용한다.

10. 결과 이미지를 저장한다.

11. 나머지 clipLimit도
    같은 방법으로 반복한다.

12. 저대비 원본과
    CLAHE 결과들을 비교한다.

13. 객체의 경계가 좋아졌는지 확인한다.

14. 동시에 배경 Texture나 Noise도
    지나치게 강조되지 않았는지 확인한다.

15. 다음 Threshold/Otsu에
    어떤 결과가 더 유리한지 이후 비교한다.
'''

from pathlib import Path

import cv2


# --------------------------------------------------
# 사용할 이미지 경로
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

low_contrast_path = Path(
    "samples/day02/low_contrast.jpg"
)

output_dir = Path(
    "outputs/day02/clahe"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 기준 ROI 이미지 읽기
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: "
        f"{reference_path}"
    )


# --------------------------------------------------
# 실습용 저대비 이미지 만들기
# --------------------------------------------------

low_contrast = cv2.convertScaleAbs(
    reference,
    alpha=0.35,
    beta=80,
)

cv2.imwrite(
    str(low_contrast_path),
    low_contrast,
)

print(
    "문제 이미지 생성:",
    low_contrast_path,
)


# --------------------------------------------------
# Gray 변환
# --------------------------------------------------

reference_gray = cv2.cvtColor(
    reference,
    cv2.COLOR_BGR2GRAY,
)

low_contrast_gray = cv2.cvtColor(
    low_contrast,
    cv2.COLOR_BGR2GRAY,
)


# 기준 이미지와 저대비 이미지 저장
cv2.imwrite(
    str(
        output_dir
        / "reference_gray.jpg"
    ),
    reference_gray,
)

cv2.imwrite(
    str(
        output_dir
        / "low_contrast_gray.jpg"
    ),
    low_contrast_gray,
)


# --------------------------------------------------
# CLAHE 설정
# --------------------------------------------------

settings = [
    (1.0, (8, 8)),
    (2.0, (8, 8)),
    (3.0, (8, 8)),
    (4.0, (8, 8)),
]


# --------------------------------------------------
# 여러 clipLimit 비교
# --------------------------------------------------

for clip_limit, tile_size in settings:

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_size,
    )

    result = clahe.apply(
        low_contrast_gray
    )

    output_path = (
        output_dir
        / f"clahe_clip_{clip_limit}.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print(
        "저장:",
        output_path,
    )