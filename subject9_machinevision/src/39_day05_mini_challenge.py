
from pathlib import Path
import csv

from day05_logo_features import (
    prepare_reference,
    measure_target,
)

# --------------------------------------
# 1. 경로 설정
# --------------------------------------
ROOT = Path("samples/day05/mini_challenge")

REFERENCE_PATH = ROOT / "reference/reference_label.jpg"
CALIBRATION_DIR = ROOT / "calibration"

OUTPUT_CSV = Path(
    "reports/day05_mini_calibration.csv"
)

# --------------------------------------
# 2. Reference 준비
# --------------------------------------
reference_data = prepare_reference(
    REFERENCE_PATH
)

print(
    "Reference Keypoints:",
    len(reference_data["keypoints"])
)

# --------------------------------------
# 3. Calibration 이미지 목록
# --------------------------------------
image_infos = []

for image_path in sorted(
    (CALIBRATION_DIR / "ok").glob("*.jpg")
):
    image_infos.append((image_path, "OK"))

for image_path in sorted(
    (CALIBRATION_DIR / "ng").glob("*.jpg")
):
    image_infos.append((image_path, "NG"))

print("Calibration 이미지:", len(image_infos))

if len(image_infos) != 6:
    raise RuntimeError(
        "Calibration 이미지가 6장인지 확인하세요."
    )

# --------------------------------------
# 4. 각 이미지 특징값 측정
# --------------------------------------
rows = []

for image_path, actual in image_infos:

    result = measure_target(
        image_path,
        reference_data
    )

    row = {
        "file": image_path.name,
        "actual": actual,
        "template_score": round(
            result["template_score"], 4
        ),
        "sift_good_matches":
            result["sift_good_matches"],
        "homography_found":
            result["homography_found"],
        "homography_inliers":
            result["homography_inliers"],
        "inlier_ratio": round(
            result["inlier_ratio"], 4
        ),
    }

    rows.append(row)
    print(row)

# --------------------------------------
# 5. Calibration 결과 CSV 저장
# --------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "actual",
            "template_score",
            "sift_good_matches",
            "homography_found",
            "homography_inliers",
            "inlier_ratio",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print("Calibration 측정 완료")
print("결과 저장:", OUTPUT_CSV)
