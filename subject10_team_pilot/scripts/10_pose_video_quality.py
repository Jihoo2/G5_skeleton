"""
교과 10 2일차 10_pose_video_quality.py와 같은 처리 (저장 위치만 reports/pilot/)

MP4 → Frame별 YOLO Pose →
  reports/pilot/videos/{이름}_pose.mp4          : Skeleton이 그려진 결과 영상
  reports/pilot/tables/{이름}_frame_quality.csv  : Frame별 품질 요약
  reports/pilot/tables/{이름}_keypoints.csv      : Frame별 17관절 좌표·Confidence
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--kp-conf", type=float, default=0.50)
    p.add_argument("--max-frames", type=int, default=0, help="0이면 영상 끝까지")
    return p.parse_args()


def main() -> None:
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

    success, frame = capture.read()
    if not success:
        raise RuntimeError("첫 Frame을 읽지 못했습니다.")
    height, width = frame.shape[:2]

    video_dir = ROOT / "reports" / "pilot" / "videos"
    table_dir = ROOT / "reports" / "pilot" / "tables"
    video_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(video_dir / f"{source.stem}_pose.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height),
    )

    frame_rows, keypoint_rows = [], []
    frame_id = 0
    while success:
        frame_id += 1
        time_sec = round((frame_id - 1) / output_fps, 3)
        result = model.predict(
            source=frame, imgsz=config["image_size"],
            conf=config["person_confidence"], device=device, verbose=False,
        )[0]
        writer.write(result.plot())

        people = len(result.boxes) if result.boxes is not None else 0
        mean_kp, low_kp, lowest_name, lowest_conf = 0.0, 17, "NO_PERSON", 0.0

        if people > 0 and result.keypoints is not None and result.keypoints.conf is not None:
            box_conf = result.boxes.conf.cpu().numpy()
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy()
            main_i = int(np.argmax(box_conf))          # 대표 사람 = BBox Confidence 최대
            main_xy, main_conf = xy[main_i], conf[main_i]
            mean_kp = float(np.mean(main_conf))
            low_kp = int(np.sum(main_conf < args.kp_conf))
            lowest_id = int(np.argmin(main_conf))
            lowest_name, lowest_conf = COCO_KEYPOINT_NAMES[lowest_id], float(main_conf[lowest_id])

            for k, name in enumerate(COCO_KEYPOINT_NAMES):
                keypoint_rows.append({
                    "frame_id": frame_id, "time_sec": time_sec,
                    "keypoint_id": k, "keypoint_name": name,
                    "pixel_x": round(float(main_xy[k, 0]), 2),
                    "pixel_y": round(float(main_xy[k, 1]), 2),
                    "confidence": round(float(main_conf[k]), 4),
                })

        frame_rows.append({
            "frame_id": frame_id, "time_sec": time_sec, "people": people,
            "mean_keypoint_conf": round(mean_kp, 4), "low_keypoint_count": low_kp,
            "lowest_keypoint": lowest_name, "lowest_keypoint_conf": round(lowest_conf, 4),
            "keypoint_threshold": args.kp_conf,
        })

        if frame_id % 30 == 0:
            print(f"Processed frame: {frame_id}")
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break
        success, frame = capture.read()

    capture.release()
    writer.release()

    for path, rows in [
        (table_dir / f"{source.stem}_frame_quality.csv", frame_rows),
        (table_dir / f"{source.stem}_keypoints.csv", keypoint_rows),
    ]:
        if not rows:
            print(f"저장할 데이터 없음: {path.name}")
            continue
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {path}")

    print(f"Frames processed: {frame_id}")


if __name__ == "__main__":
    main()
