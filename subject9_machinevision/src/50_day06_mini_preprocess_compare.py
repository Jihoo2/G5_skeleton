
import csv
from pathlib import Path

import cv2

from day06_ocr_utils import (
    PROJECT_ROOT,
    load_rules,
    make_preprocess_variants,
    run_ocr,
)
from day06_ocr_validation import clean_ocr_text

# 검사할 이미지
IMAGE_PATH = (
    PROJECT_ROOT
    / "samples/day06/mini_challenge/product"
    / "0386316000696_side.png"
)

EXPECTED = "TI776224-2D"

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/day06/mini_preprocess"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "reports/day06_mini_preprocess_compare.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 기존 OCR Rule은 메모리에서만 수정
rules = load_rules().copy()
rules["ocr_psm"] = 7
rules["whitelist"] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

# 원본 이미지 열기
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

# 실패 이미지의 Ch.-B. 코드만 다시 선택
window = "Select Ch-B Code"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, 1100, 800)

print("TI776224-2D 영역만 선택하고 ENTER를 누르세요.")

x, y, w, h = cv2.selectROI(
    window,
    image,
    fromCenter=False,
    showCrosshair=True,
)

cv2.destroyAllWindows()

if w == 0 or h == 0:
    raise RuntimeError("ROI 선택이 취소되었습니다.")

roi = image[y:y+h, x:x+w]

# 전처리 후보 5종
variants = make_preprocess_variants(roi, rules)

rows = []

for name, processed in variants.items():

    # 동일하게 4배 확대
    enlarged = cv2.resize(
        processed,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    # 실제 OCR 입력 이미지 보관
    cv2.imwrite(
        str(OUTPUT_DIR / f"{name}.png"),
        enlarged,
    )

    raw_text = run_ocr(enlarged, rules)
    ocr_text = clean_ocr_text(raw_text).upper()

    exact_match = ocr_text == EXPECTED

    row = {
        "preprocess": name,
        "expected_text": EXPECTED,
        "ocr_text": ocr_text,
        "exact_match": exact_match,
    }

    rows.append(row)

    print(
        f"{name:12s} | "
        f"OCR: {ocr_text:15s} | "
        f"Exact: {exact_match}"
    )

# 결과 CSV 저장
with REPORT_PATH.open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n결과 저장:", REPORT_PATH)
