
from pathlib import Path
import csv
import cv2
import numpy as np

# --------------------------------------
# 1. 파일 경로
# --------------------------------------
root = Path("samples/day04/my_challenge")

reference_path = root / "full_front.jpg"

challenge_files = [
    root / "shift.jpg",
    root / "rotate.jpg",
    root / "distance.jpg",
    root / "tilt_glare.jpg",
]

output_dir = Path("outputs/day04/my_challenge")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# 2. 기준 이미지에서 ROI 직접 선택
# --------------------------------------
reference_full = cv2.imread(str(reference_path))

if reference_full is None:
    raise FileNotFoundError(reference_path)

# 마우스로 파란색 글자 + 주변 라벨 선택
x, y, w, h = cv2.selectROI(
    "Select Reference ROI",
    reference_full,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("Select Reference ROI")

if w <= 0 or h <= 0:
    raise ValueError("ROI를 선택하지 않았습니다.")

reference = reference_full[y:y+h, x:x+w]

cv2.imwrite(
    str(output_dir / "reference_label.jpg"),
    reference
)

print("기준 로고 저장 완료")

# --------------------------------------
# 3. SIFT 특징점 추출
# --------------------------------------
sift = cv2.SIFT_create()

reference_gray = cv2.cvtColor(
    reference, cv2.COLOR_BGR2GRAY
)

kp_ref, des_ref = sift.detectAndCompute(
    reference_gray, None
)

if des_ref is None:
    raise RuntimeError("기준 이미지에서 특징점을 찾지 못했습니다.")

matcher = cv2.BFMatcher(cv2.NORM_L2)

rows = []

# --------------------------------------
# 4. 실제 촬영 사진 검사
# --------------------------------------
for target_path in challenge_files:

    target = cv2.imread(str(target_path))

    if target is None:
        print("파일 읽기 실패:", target_path)
        continue

    gray = cv2.cvtColor(
        target, cv2.COLOR_BGR2GRAY
    )

    kp_target, des_target = sift.detectAndCompute(
        gray, None
    )

    good_matches = []
    inliers = 0
    status = "NOT_FOUND"

    preview = target.copy()

    if des_target is not None:

        matches = matcher.knnMatch(
            des_ref, des_target, k=2
        )

        # Ratio Test
        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # --------------------------------------
    # 5. Homography + RANSAC
    # --------------------------------------
    if len(good_matches) >= 4:

        src_pts = np.float32([
            kp_ref[m.queryIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp_target[m.trainIdx].pt
            for m in good_matches
        ]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if H is not None and mask is not None:

            inliers = int(mask.sum())

            if inliers >= 4:

                ref_h, ref_w = reference_gray.shape

                corners = np.float32([
                    [0, 0],
                    [ref_w, 0],
                    [ref_w, ref_h],
                    [0, ref_h]
                ]).reshape(-1, 1, 2)

                projected = cv2.perspectiveTransform(
                    corners, H
                )

                if np.isfinite(projected).all():
                    cv2.polylines(
                        preview,
                        [np.int32(projected)],
                        True,
                        (0, 255, 0),
                        3
                    )

                    status = "FOUND"

    inlier_ratio = (
        inliers / len(good_matches)
        if good_matches else 0.0
    )

    # --------------------------------------
    # 6. 결과 이미지 저장
    # --------------------------------------
    output_path = output_dir / f"{target_path.stem}_result.jpg"

    cv2.imwrite(str(output_path), preview)

    rows.append({
        "filename": target_path.name,
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "status": status
    })

    print(
        target_path.name,
        "Good Match:", len(good_matches),
        "Inlier:", inliers,
        "Inlier Ratio:", round(inlier_ratio, 4),
        "Status:", status
    )

# --------------------------------------
# 7. CSV 저장
# --------------------------------------
report_path = Path("reports/day04_challenge_results.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)

with report_path.open("w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "good_matches",
            "inliers",
            "inlier_ratio",
            "status"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)

print("Mini Challenge 코드 실행 완료")
print("결과 CSV:", report_path)
