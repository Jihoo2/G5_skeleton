
from pathlib import Path
import csv
import json

from day05_logo_features import prepare_reference, measure_target

# 1. Calibration으로 결정한 Rule
rule = {
    "version": "day05-vitamin-v1.0",
    "sift_good_matches_min": 50,
    "inlier_ratio_min": 0.82,
    "homography_found": 1,
}

config_path = Path("configs/day05_vitamin_rules.json")
config_path.parent.mkdir(parents=True, exist_ok=True)

# 이미 동결한 Rule이 있으면 그대로 사용
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as f:
        rule = json.load(f)
else:
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(rule, f, indent=4, ensure_ascii=False)

print("사용할 Rule:", rule)

# 2. Reference 준비
root = Path("samples/day05/mini_challenge")
reference = prepare_reference(
    root / "reference/reference_label.jpg"
)

# 3. Test 이미지 목록
test_images = []

for path in sorted((root / "test/ok").glob("*.jpg")):
    test_images.append((path, "OK"))

for path in sorted((root / "test/ng").glob("*.jpg")):
    test_images.append((path, "NG"))

if len(test_images) != 4:
    raise RuntimeError(
        f"Test 이미지는 4장이어야 합니다. 현재 {len(test_images)}장"
    )

# 4. Test 실행
rows = []
counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

for path, actual in test_images:
    result = measure_target(path, reference)

    passed = (
        result["homography_found"] == 1
        and result["sift_good_matches"]
        >= rule["sift_good_matches_min"]
        and result["inlier_ratio"]
        >= rule["inlier_ratio_min"]
    )

    predicted = "OK" if passed else "NG"

    if actual == "NG" and predicted == "NG":
        error_type = "TP"
    elif actual == "OK" and predicted == "OK":
        error_type = "TN"
    elif actual == "OK" and predicted == "NG":
        error_type = "FP"
    else:
        error_type = "FN"

    counts[error_type] += 1

    row = {
        "file": path.name,
        "actual": actual,
        "predicted": predicted,
        "error_type": error_type,
        "sift_good_matches": result["sift_good_matches"],
        "homography_found": result["homography_found"],
        "inlier_ratio": round(result["inlier_ratio"], 4),
    }

    rows.append(row)
    print(row)

# 5. CSV 저장
csv_path = Path("reports/day05_mini_test_results.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# 6. 평가 지표
tp, tn = counts["TP"], counts["TN"]
fp, fn = counts["FP"], counts["FN"]

accuracy = (tp + tn) / len(rows)
precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

print("\n=== Test 결과 ===")
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)
print("Accuracy:", round(accuracy, 4))
print("NG Precision:", round(precision, 4))
print("NG Recall:", round(recall, 4))
print("CSV 저장:", csv_path)
