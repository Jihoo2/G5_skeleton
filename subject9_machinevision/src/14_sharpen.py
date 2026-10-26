''' 의사코드
[1. 이미지 경로 설정]

1. 사용할 이미지와 결과 저장 경로를 정한다.

   기준 이미지:
   samples/day02/cap_on_reference.jpg

   Blur 이미지:
   samples/day02/blur.jpg

   결과 저장 폴더:
   outputs/day02/sharpen/


[2. 출력 폴더 생성]

2. Sharpening 결과를 저장할 폴더를 생성한다.

   폴더가 없으면 새로 만든다.

   폴더가 이미 있다면 그대로 사용한다.


[3. 기준 이미지 읽기]

3. 1일차에서 만든 병뚜껑 기준 ROI 이미지를 읽는다.

   cv2.imread() 사용

   읽은 이미지를 reference 변수에 저장한다.


[4. 이미지 오류 확인]

4. 기준 이미지를 정상적으로 읽었는지 확인한다.

   reference가 None인가?

   예
   → FileNotFoundError 발생

   아니요
   → 다음 단계로 진행


[5. Gaussian Blur 적용]

5. 기준 ROI 이미지에 Gaussian Blur를 적용한다.

   Kernel 크기 = 7×7

   Sigma = 자동 계산

   이미지의 경계와 세부 형태를 흐리게 만든다.

   결과를 blurred 변수에 저장한다.


[6. Blur 이미지 저장]

6. 흐리게 만든 이미지를 저장한다.

   저장 파일:
   samples/day02/blur.jpg

   이 이미지는 이후 Sharpening 실습에 사용한다.


[7. Sharpening Kernel 생성]

7. 이미지 경계를 강조할 3×3 Kernel을 만든다.

        0  -1   0
       -1   5  -1
        0  -1   0

   중심 픽셀과 주변 픽셀의 차이를 이용하여
   경계를 강조하는 Kernel이다.

   생성한 Kernel을 kernel 변수에 저장한다.


[8. Sharpening 적용]

8. Blur 이미지에 Sharpening Kernel을 적용한다.

   cv2.filter2D() 사용

   입력 이미지:
   blurred

   적용할 Kernel:
   kernel

   결과를 sharpened 변수에 저장한다.


[9. 원본 이미지 저장]

9. 전처리를 적용하지 않은 기준 ROI 이미지를 저장한다.

   저장 파일:
   outputs/day02/sharpen/reference.jpg


[10. Blur 이미지 저장]

10. Gaussian Blur를 적용한 이미지를 저장한다.

    저장 파일:
    outputs/day02/sharpen/blurred.jpg


[11. Sharpening 이미지 저장]

11. Sharpening을 적용한 이미지를 저장한다.

    저장 파일:
    outputs/day02/sharpen/sharpened.jpg


[12. 결과 확인 및 분석]

12. 저장 완료 메시지를 출력한다.

    저장된 이미지 3장을 직접 비교한다.

    reference.jpg
         ↓
    원본 병뚜껑의 경계와 형태 확인

    blurred.jpg
         ↓
    Blur로 경계가 흐려진 정도 확인

    sharpened.jpg
         ↓
    Sharpening 후 경계 강조 효과 확인

    최종적으로 다음을 판단한다.

    병뚜껑 외곽이 더 잘 보이는가?

    병뚜껑의 세로 홈이 유지되는가?

    원본과 동일하게 복원된 것인가?

    Noise까지 함께 강조되지는 않았는가?

    다음 Otsu 이진화에 도움이 될 수 있는가?
'''

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 사용할 이미지와 결과 저장 경로를 정한다.
#
#    reference_path → 1일차 병뚜껑 기준 ROI
#    blur_path      → 흐리게 만든 이미지 저장 경로
#    output_dir     → Sharpening 비교 결과 폴더
# --------------------------------------------------

reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

blur_path = Path(
    "samples/day02/blur.jpg"
)

output_dir = Path(
    "outputs/day02/sharpen"
)


# --------------------------------------------------
# 2. Sharpening 결과를 저장할 폴더를 생성한다.
#
#    parents=True → 상위 폴더도 필요하면 생성
#    exist_ok=True → 폴더가 이미 있어도 오류 없음
# --------------------------------------------------

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 3. 기준 ROI 이미지를 읽는다.
#
#    cv2.imread()로 병뚜껑 이미지를 불러온다.
# --------------------------------------------------

reference = cv2.imread(
    str(reference_path)
)


# --------------------------------------------------
# 4. 이미지를 정상적으로 읽었는지 확인한다.
#
#    reference가 None이면 파일을 읽지 못한 것이다.
#    오류 메시지를 출력하고 실행을 중단한다.
# --------------------------------------------------

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: "
        f"{reference_path}"
    )


# --------------------------------------------------
# 5. 기준 이미지에 Gaussian Blur를 적용한다.
#
#    (7, 7) → 7×7 Kernel 사용
#    0      → Sigma 값을 OpenCV가 자동 계산
#
#    병뚜껑의 경계와 세부 형태를 일부러 흐리게 만든다.
# --------------------------------------------------

blurred = cv2.GaussianBlur(
    reference,
    (7, 7),
    0,
)


# --------------------------------------------------
# 6. 흐리게 만든 이미지를 blur.jpg로 저장한다.
#
#    이후 Sharpening 효과를 비교할 문제 이미지이다.
# --------------------------------------------------

cv2.imwrite(
    str(blur_path),
    blurred,
)

print(
    "문제 이미지 생성:",
    blur_path,
)


# --------------------------------------------------
# 7. Sharpening Kernel을 만든다.
#
#    3×3 행렬을 사용한다.
#
#       0  -1   0
#      -1   5  -1
#       0  -1   0
#
#    중심 픽셀의 값을 강조하고
#    주변 픽셀과의 차이를 이용하여 경계를 강조한다.
#
#    dtype=np.float32 → 실수형 Kernel
# --------------------------------------------------

kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)


# --------------------------------------------------
# 8. Blur 이미지에 Sharpening을 적용한다.
#
#    cv2.filter2D() → Kernel을 이미지 전체에 적용
#
#    blurred → 입력 이미지
#    -1      → 출력 이미지의 자료형 깊이를 입력과 동일하게 유지
#    kernel  → 7번에서 만든 Sharpening Kernel
#
#    결과를 sharpened에 저장한다.
# --------------------------------------------------

sharpened = cv2.filter2D(
    blurred,
    -1,
    kernel,
)


# --------------------------------------------------
# 9. 기준 ROI 이미지를 reference.jpg로 저장한다.
#
#    Blur와 Sharpening을 적용하지 않은
#    원래 병뚜껑 이미지이다.
# --------------------------------------------------

cv2.imwrite(
    str(
        output_dir / "reference.jpg"
    ),
    reference,
)


# --------------------------------------------------
# 10. Blur 이미지를 blurred.jpg로 저장한다.
#
#     5번에서 Gaussian Blur를 적용한 결과이다.
#     기준 이미지와 비교하여 경계가 얼마나
#     흐려졌는지 확인할 수 있다.
# --------------------------------------------------

cv2.imwrite(
    str(
        output_dir / "blurred.jpg"
    ),
    blurred,
)


# --------------------------------------------------
# 11. Sharpening 결과를 sharpened.jpg로 저장한다.
#
#     8번에서 경계를 강조한 결과이다.
#     Blur 이미지보다 경계가 강조되었는지 비교한다.
# --------------------------------------------------

cv2.imwrite(
    str(
        output_dir / "sharpened.jpg"
    ),
    sharpened,
)


# --------------------------------------------------
# 12. 결과 저장 완료 메시지를 출력한다.
#
#     저장된 이미지 3장을 직접 비교한다.
#
#     reference.jpg
#     blurred.jpg
#     sharpened.jpg
#
#     확인할 내용:
#     - Blur 후 병뚜껑 경계가 흐려졌는가?
#     - Sharpening으로 경계가 다시 강조되었는가?
#     - 원본과 동일하게 복원되었는가?
#     - Noise나 불필요한 경계도 강조되었는가?
# --------------------------------------------------

print(
    "Sharpening 결과 저장 완료"
)