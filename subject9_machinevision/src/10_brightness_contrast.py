''' 의사코드
1. 1일차에서 만든 병뚜껑 기준 ROI 이미지의 경로를 정한다.

2. 실습용으로 만들 어두운 이미지의 저장 경로를 정한다.

3. Brightness 비교 결과를 저장할 폴더를 준비한다.

4. 기준 ROI 이미지를 읽는다.

5. 이미지가 정상적으로 읽히지 않았다면 오류를 발생시킨다.

6. 기준 이미지의 밝기를 약 55% 수준으로 낮춘다.

7. 어둡게 만든 이미지를 dark.jpg로 저장한다.

8. Brightness를 비교할 beta 값을 준비한다.

   alpha는 1.0으로 고정하고

   beta = 0
   beta = 20
   beta = 50
   beta = 80

   순서로 사용한다.

9. dark.jpg에 첫 번째 alpha, beta 값을 적용한다.

10. 변환된 결과 이미지를 저장한다.

11. 나머지 beta 값도 같은 방식으로 반복 적용한다.

12. 모든 결과 이미지를 각각 다른 파일명으로 저장한다.

13. 저장된 결과를 비교하여
    beta가 커질수록 이미지 전체가 어떻게 밝아지는지 확인한다.

14. 너무 밝게 만든 경우
    병뚜껑의 경계나 밝은 영역의 세부정보가 사라지는지도 함께 확인한다.
'''

from pathlib import Path

import cv2


# 1. 1일차에서 만든 기준 ROI
reference_path = Path(
    "samples/day02/cap_on_reference.jpg"
)

# 2. 오늘 사용할 문제 이미지
dark_path = Path(
    "samples/day02/dark.jpg"
)

# 3. 결과 저장 위치
output_dir = Path(
    "outputs/day02/brightness"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# 기준 이미지 읽기
reference = cv2.imread(
    str(reference_path)
)

if reference is None:
    raise FileNotFoundError(
        f"기준 이미지를 읽지 못했습니다: {reference_path}"
    )


# --------------------------------------------------
# 실습용 어두운 이미지 만들기
# --------------------------------------------------

dark = cv2.convertScaleAbs(
    reference,
    alpha=0.55,
    beta=0,
)

cv2.imwrite(
    str(dark_path),
    dark,
)

print("문제 이미지 생성:", dark_path)


# --------------------------------------------------
# Brightness / Contrast 비교
# --------------------------------------------------

settings = [
    (1.0, 0),
    (1.0, 20),
    (1.0, 50),
    (1.0, 80),
]


for alpha, beta in settings:

    result = cv2.convertScaleAbs(
        dark,
        alpha=alpha,
        beta=beta,
    )

    filename = (
        f"alpha_{alpha}_beta_{beta}.jpg"
    )

    output_path = (
        output_dir / filename
    )

    cv2.imwrite(
        str(output_path),
        result,
    )

    print("저장:", output_path)