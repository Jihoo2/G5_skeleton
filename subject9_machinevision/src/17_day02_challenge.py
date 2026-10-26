# ==================================================
# 17_day02_challenge.py
#
# 2일차 Mini Challenge (교재 28~33번)
#
# 목적
#   새로운 challenge.jpg를 보고 문제를 정의한 뒤,
#   전처리 후보 여러 개를 같은 Otsu 기준으로 비교하여
#   병뚜껑/배경 분리에 실제로 도움이 되는 방법을 선택한다.
#
# 원본 템플릿과의 차이
#   - result 하나만 시험 → 후보 6개를 한 번에 비교
#   - 후보별 결과 파일 + 비교 화면(challenge_compare.png) 저장
#   - Otsu 기준값과 흰색 픽셀 비율을 함께 출력
#   - 최종 산출물 파일명(challenge_result*.jpg)은 그대로 유지
# ==================================================

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. 입력·출력 경로 설정
#
# challenge.jpg는 roi_temp/crop_challenge.py로
# 병뚜껑 주변만 잘라낸 ROI 이미지이다.
# (원본은 roi_temp/challenge_original.jpg에 백업됨)
# --------------------------------------------------

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


# --------------------------------------------------
# 2. 이미지 읽기 + 예외 처리
#
# cv2.imread()는 실패해도 오류를 내지 않고
# None을 반환하므로 직접 검사해야 한다.
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )


# --------------------------------------------------
# 3. 문제 정의 (교재 28번)
#
# 관찰
#   - 병뚜껑이 흰색이고 위쪽 배경도 밝은 회색이다.
#   - 전체 밝기가 어둡지는 않다.
#   - 홈과 외곽선이 약간 뿌옇다.
#   - 아래쪽 링과 투명 병목도 밝게 보인다.
#
# 문제
#   병뚜껑과 배경이 모두 밝은 계열이라 밝기 차이가 작고,
#   외곽 경계가 약간 흐려 Otsu에서 배경이 함께
#   선택될 수 있다.
#
# 후보 선택 (교재 29번)
#   - CLAHE      : 지역 대비를 키워 뚜껑/배경 차이 확대
#   - Gamma < 1  : 중간 밝기(배경)를 더 많이 낮춰
#                  밝은 뚜껑과의 간격 확대
#   - original   : 전처리 없이도 충분한지 확인하는 기준
#
# 제외
#   - Brightness / Gamma > 1 (밝게 만들기)
#     → 밝은 픽셀이 255 근처로 몰려(포화)
#       뚜껑과 배경 차이가 오히려 줄어들 수 있음
# --------------------------------------------------


# --------------------------------------------------
# 4. 공통 함수
# --------------------------------------------------

def to_gray(source):
    """
    컬러(BGR)면 Gray로 변환하고,
    이미 Gray(2차원 배열)면 그대로 반환한다.

    CLAHE 결과처럼 이미 Gray인 이미지와
    Gamma 결과처럼 컬러인 이미지를
    같은 방식으로 처리하기 위해 사용한다.
    """

    if len(source.shape) == 2:
        return source

    return cv2.cvtColor(
        source,
        cv2.COLOR_BGR2GRAY,
    )


def adjust_gamma(source, gamma):
    """
    Gamma Correction (11_gamma.py와 동일한 구현)

    output = (input / 255) ** (1 / gamma) * 255

    이 구현 기준
      gamma < 1.0 → 어두워짐
      gamma = 1.0 → 변화 없음
      gamma > 1.0 → 밝아짐

    gamma < 1이면 중간 밝기는 많이 내려가고,
    아주 밝은 값은 덜 내려간다.
      예) gamma 0.6
          배경 200 → 약 170
          뚜껑 240 → 약 225
          차이 40  → 약 55 로 확대

    0~255 모든 값의 변환 결과를 표(Lookup Table)로
    미리 만들어 두고 cv2.LUT로 한 번에 적용한다.
    """

    inv_gamma = 1.0 / gamma

    table = np.array(
        [
            ((value / 255.0) ** inv_gamma) * 255
            for value in np.arange(256)
        ],
        dtype=np.uint8,
    )

    return cv2.LUT(
        source,
        table,
    )


def apply_clahe(source, clip_limit):
    """
    CLAHE (12_clahe.py와 동일한 방식)

    이미지를 8×8 타일로 나누어 지역별로 대비를 높인다.

    clipLimit
      작을수록 → 부드럽게, 개선 효과 약함
      클수록   → 대비 강함, 배경 Texture·Noise도 강조됨

    CLAHE는 1채널에 적용하므로 Gray로 변환 후 사용하며,
    반환값도 Gray 이미지이다.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        to_gray(source)
    )


def apply_otsu(source):
    """
    Otsu 이진화

    THRESH_BINARY
      밝은 병뚜껑 → 흰색(255) 객체
      어두운 영역 → 검정(0)

      ※ 앞 실습의 검은 병뚜껑은 THRESH_BINARY_INV를 썼다.
        이번 뚜껑은 밝은 색이라 방향이 반대다. (교재 30번)

    반환
      value  : Otsu가 자동으로 선택한 기준값
      binary : 이진화 결과 이미지

    원본 템플릿은 value를 '_'로 버렸지만,
    전처리에 따라 기준값이 어떻게 달라지는지
    비교하기 위해 함께 반환한다.
    """

    value, binary = cv2.threshold(
        to_gray(source),
        0,          # Otsu 사용 시 무시됨 (자동 계산)
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU,
    )

    return value, binary


# --------------------------------------------------
# 5. 후보 전처리 생성 (교재 30~31번)
#
# 처음 고른 값이 정답이라고 가정하지 않고
# 기술별로 Parameter를 2~3개 바꾸어 비교한다.
#
# 모든 후보에 "같은 Otsu"를 적용해야
# 전처리 효과만 공정하게 비교할 수 있다.
# --------------------------------------------------

candidates = {
    "original": image,                    # 비교 기준 (전처리 없음)
    "clahe_1.0": apply_clahe(image, 1.0),
    "clahe_2.0": apply_clahe(image, 2.0),
    "clahe_3.0": apply_clahe(image, 3.0),
    "gamma_0.6": adjust_gamma(image, 0.6),
    "gamma_0.8": adjust_gamma(image, 0.8),
}


# --------------------------------------------------
# 6. 비교 화면용 타일 함수
#
# 16_preprocess_compare.py의 make_view와 같은 방식.
# 모든 결과를 같은 크기로 맞추고 제목을 붙인다.
#
# Binary 이미지는 INTER_NEAREST로 축소한다.
#   → 0과 255 사이의 중간값(회색)이 생기지 않아
#     이진화 결과가 그대로 보인다.
# --------------------------------------------------

tile_size = (320, 270)   # (width, height), ROI 비율에 가깝게


def make_tile(source, title, is_binary=False):

    interpolation = (
        cv2.INTER_NEAREST
        if is_binary
        else cv2.INTER_AREA
    )

    tile = cv2.resize(
        source,
        tile_size,
        interpolation=interpolation,
    )

    # hconcat / vconcat은 채널 수가 같아야 하므로
    # Gray 이미지를 3채널로 맞춘다.
    if len(tile.shape) == 2:
        tile = cv2.cvtColor(
            tile,
            cv2.COLOR_GRAY2BGR,
        )

    # 제목 배경(검정 띠) + 흰 글씨
    cv2.rectangle(
        tile,
        (0, 0),
        (tile_size[0], 32),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        tile,
        title,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return tile


# --------------------------------------------------
# 7. 후보별 Otsu 적용 + 결과 저장 + 수치 출력 (교재 32번)
#
# Otsu    : 자동으로 선택된 기준값
# white % : Otsu 결과에서 흰색(객체) 픽셀 비율
#
# white % 해석 (original 대비)
#   크게 증가 → 배경까지 객체로 잡혔을 가능성
#   크게 감소 → 병뚜껑 일부가 잘렸을 가능성
#
# ※ 수치는 판단 보조용이다.
#   최종 판단은 반드시 비교 화면을 눈으로 확인한다.
# --------------------------------------------------

print()
print(f"{'method':<11}{'Otsu':>8}{'white %':>10}")
print("-" * 29)

rows = []

for name, processed in candidates.items():

    value, binary = apply_otsu(processed)

    white_ratio = (
        cv2.countNonZero(binary)
        / binary.size
        * 100
    )

    # 후보마다 다른 파일명으로 저장 → 덮어쓰기 없음
    # CSV 기록(교재 34번)에 바로 사용 가능
    cv2.imwrite(
        str(output_dir / f"{name}.jpg"),
        processed,
    )

    cv2.imwrite(
        str(output_dir / f"{name}_otsu.jpg"),
        binary,
    )

    print(f"{name:<11}{value:>8.1f}{white_ratio:>9.1f}%")

    # 한 행 = [전처리 결과 | Otsu 결과]
    rows.append(
        cv2.hconcat(
            [
                make_tile(processed, name),
                make_tile(
                    binary,
                    f"{name} + Otsu",
                    is_binary=True,
                ),
            ]
        )
    )


# 모든 행을 세로로 이어 붙여 한 장으로 저장
compare_path = (
    output_dir / "challenge_compare.png"
)

cv2.imwrite(
    str(compare_path),
    cv2.vconcat(rows),
)


# --------------------------------------------------
# 8. 최종 선택 (교재 33번)
#
# challenge_compare.png를 보고
# 아래 기준으로 가장 좋은 후보를 고른 뒤
# SELECTED 값을 바꾸어 다시 실행한다.
#
#   - 병뚜껑이 끊기지 않은 흰 덩어리로 나오는가?
#   - 위쪽 배경이 흰색으로 같이 잡히지 않는가?
#   - 아래 링·병목이 많이 섞이지 않는가?
#
# 선택 가능한 값
#   "original", "clahe_1.0", "clahe_2.0",
#   "clahe_3.0", "gamma_0.6", "gamma_0.8"
#
# 전처리가 도움이 되지 않았다면
# "original"(미사용)이 올바른 결론이다. (교재 25번)
# --------------------------------------------------

SELECTED = "clahe_1.0"

result = candidates[SELECTED]


# --------------------------------------------------
# 9. 최종 산출물 저장
#
# 원본 템플릿과 같은 파일명으로 저장하여
# 교재 41번 결과 확인 구조를 그대로 유지한다.
#
#   challenge_result.jpg        : 선택한 전처리 결과
#   challenge_original_otsu.jpg : 전처리 전 Otsu
#   challenge_result_otsu.jpg   : 전처리 후 Otsu
# --------------------------------------------------

result_path = (
    output_dir / "challenge_result.jpg"
)

cv2.imwrite(
    str(result_path),
    result,
)

original_value, original_otsu = apply_otsu(image)
result_value, result_otsu = apply_otsu(result)

cv2.imwrite(
    str(output_dir / "challenge_original_otsu.jpg"),
    original_otsu,
)

cv2.imwrite(
    str(output_dir / "challenge_result_otsu.jpg"),
    result_otsu,
)


# --------------------------------------------------
# 10. 실행 결과 요약
# --------------------------------------------------

print()
print("입력      :", image_path)
print("비교 화면 :", compare_path)
print("최종 선택 :", SELECTED)
print(f"Otsu 전/후 : {original_value:.1f} / {result_value:.1f}")
print("Challenge 완료")