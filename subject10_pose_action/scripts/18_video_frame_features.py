from __future__ import annotations

# ==================================================
# 18_video_frame_features.py
# --------------------------------------------------
# MP4의 각 Frame에 YOLO Pose를 적용하고
# 34개 정규화 좌표 + 4개 관절각도 = 38-D Feature를
# 시간 순서대로 CSV에 저장합니다.
#
# 출력: reports/day03/tables/{영상이름}_frame_features.csv
#   1행 = 1 Frame
#   frame_id, time_sec, valid_pose, people,
#   mean_keypoint_conf, low_keypoint_count,
#   hip_center_x, hip_center_y, shoulder_width,
#   + 38개 Feature
# ==================================================

# ==================================================
# [01] 필요한 라이브러리 가져오기
# ==================================================
import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==================================================
# [02] 프로젝트 ROOT 경로 설정
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ==================================================
# [03] 공통 함수 가져오기
# ==================================================
from src.common import (
    load_config,
    resolve_device,
)
from src.feature_utils import (
    normalize_by_shoulder_width,
    selected_joint_angles,
)
from src.pose_utils import (
    COCO_KEYPOINT_NAMES,
)


# ==================================================
# [04] 38개 Feature 이름 (16번과 같은 순서)
# ==================================================
ANGLE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
]


def make_feature_names() -> list[str]:
    names = []
    for keypoint_name in COCO_KEYPOINT_NAMES:
        names.append(f"{keypoint_name}_x")
        names.append(f"{keypoint_name}_y")
    names.extend(ANGLE_NAMES)
    return names


FEATURE_NAMES = make_feature_names()


# ==================================================
# [05] 실행 옵션
# ==================================================
# --source     : 분석할 MP4 (기본: data/day03/videos/feature_demo.mp4)
# --kp-conf    : LOW Keypoint 판단 기준 (기본 0.50)
# --max-frames : 최대 처리 Frame 수 (0이면 영상 끝까지)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=str(
            ROOT
            / "data"
            / "day03"
            / "videos"
            / "feature_demo.mp4"
        ),
        help="분석할 MP4 파일",
    )

    parser.add_argument(
        "--kp-conf",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="처리할 최대 Frame 수. 0이면 영상 끝까지 처리",
    )

    return parser.parse_args()


# ==================================================
# [06] Frame 한 장에서 Feature 계산
# ==================================================
# 성공하면 Feature 정보 딕셔너리를 반환하고,
# Pose가 없거나 정규화·각도 계산이 불가능하면
# valid_pose = 0 인 행을 반환합니다.

def frame_to_feature(result, kp_threshold: float) -> dict:

    empty = {
        "valid_pose": 0,
        "people": 0,
        "mean_keypoint_conf": "",
        "low_keypoint_count": "",
        "hip_center_x": "",
        "hip_center_y": "",
        "shoulder_width": "",
        **{name: "" for name in FEATURE_NAMES},
    }

    people = (
        len(result.boxes)
        if result.boxes is not None
        else 0
    )
    empty["people"] = people

    if (
        people == 0
        or result.keypoints is None
        or result.keypoints.xy is None
        or len(result.keypoints.xy) == 0
    ):
        return empty

    # 대표 사람: BBox Confidence가 가장 높은 사람 (2일차 10번과 같은 기준)
    box_conf = result.boxes.conf.cpu().numpy()
    main_index = int(np.argmax(box_conf))

    keypoints = result.keypoints.xy[main_index].cpu().numpy()

    if result.keypoints.conf is not None:
        confidence = result.keypoints.conf[main_index].cpu().numpy()
    else:
        confidence = np.ones(17, dtype=np.float32)

    # 정규화 (어깨 폭이 0에 가까우면 ValueError)
    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
    except ValueError:
        return empty

    angles = selected_joint_angles(keypoints)

    # 각도 계산이 불가능(NaN)하면 유효 Feature로 사용하지 않음
    if any(math.isnan(angles[name]) for name in ANGLE_NAMES):
        return empty

    values = []
    for point in normalized:
        values.extend([float(point[0]), float(point[1])])
    values.extend(float(angles[name]) for name in ANGLE_NAMES)

    row = {
        "valid_pose": 1,
        "people": people,
        "mean_keypoint_conf": round(float(np.mean(confidence)), 4),
        "low_keypoint_count": int(np.sum(confidence < kp_threshold)),
        "hip_center_x": round(float(center[0]), 4),
        "hip_center_y": round(float(center[1]), 4),
        "shoulder_width": round(float(scale), 4),
    }

    for name, value in zip(FEATURE_NAMES, values):
        row[name] = round(value, 6)

    return row


def main() -> None:

    # ==============================================
    # [07] 입력 영상 확인 및 모델 준비
    # ==============================================
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(f"영상이 없습니다: {source}")

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = fps if fps > 0 else 30.0

    # ==============================================
    # [08] Frame마다 Pose → 38-D Feature
    # ==============================================
    rows = []
    frame_id = 0

    while True:
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

        success, frame = capture.read()
        if not success:
            break

        frame_id += 1

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        row = {
            "frame_id": frame_id,
            "time_sec": round((frame_id - 1) / output_fps, 3),
        }
        row.update(frame_to_feature(result, args.kp_conf))
        rows.append(row)

        if frame_id % 30 == 0:
            print(f"Processed frame: {frame_id}")

    capture.release()

    if not rows:
        raise RuntimeError("영상에서 Frame을 읽지 못했습니다.")

    # ==============================================
    # [09] CSV 저장
    # ==============================================
    output_path = (
        ROOT
        / "reports"
        / "day03"
        / "tables"
        / f"{source.stem}_frame_features.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # [10] 결과 요약 출력
    # ==============================================
    valid_count = sum(row["valid_pose"] for row in rows)

    print()
    print(f"Source: {source.name}")
    print(f"FPS: {output_fps:.2f}")
    print(f"Total frames: {len(rows)}")
    print(f"Valid pose frames: {valid_count}")
    print(f"Feature count per frame: {len(FEATURE_NAMES)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
