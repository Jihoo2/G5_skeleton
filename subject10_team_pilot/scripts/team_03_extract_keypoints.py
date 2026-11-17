"""
[팀 03] Manifest의 모든 영상에서 YOLO Pose Keypoint 추출 (교과 2일차 10번과 같은 형식)

출력: reports/team_projects/<team>/keypoints/<clip_id>_keypoints.csv
      frame_id, time_sec, keypoint_id, keypoint_name, pixel_x, pixel_y, confidence, people

- 여러 명이 잡히면 BBox Confidence가 가장 높은 사람을 사용합니다 (교과와 동일).
- 이미 만든 파일은 건너뜁니다. 다시 만들려면 --overwrite
- review_status 가 EXCLUDE 인 영상은 건너뜁니다.

실행
  python scripts/team_03_extract_keypoints.py
  python scripts/team_03_extract_keypoints.py --clip T05_S01_A008_T01_side
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402
from src.team_feature_utils import load_team_config, team_paths  # noqa: E402

FIELDS = ["frame_id", "time_sec", "keypoint_id", "keypoint_name",
          "pixel_x", "pixel_y", "confidence", "people"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", help="특정 clip_id만 처리")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def extract(model, video: Path, out: Path, config: dict, device) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, frame_id = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        r = model.predict(source=frame, imgsz=config["image_size"],
                          conf=config["person_confidence"], device=device, verbose=False)[0]
        people = len(r.boxes) if r.boxes is not None else 0
        if people and r.keypoints is not None and r.keypoints.conf is not None:
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            xy = r.keypoints.xy.cpu().numpy()[i]
            cf = r.keypoints.conf.cpu().numpy()[i]
            for k, name in enumerate(COCO_KEYPOINT_NAMES):
                rows.append([frame_id, round((frame_id - 1) / fps, 3), k, name,
                             round(float(xy[k, 0]), 2), round(float(xy[k, 1]), 2),
                             round(float(cf[k]), 4), people])
    cap.release()
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    return frame_id


def main() -> None:
    args = parse_args()
    cfg = load_team_config()
    p = team_paths(cfg)
    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    if args.clip:
        man = man[man["clip_id"] == args.clip]
    man = man[man["review_status"] != "EXCLUDE"]

    for n, (_, r) in enumerate(man.iterrows(), 1):
        out = p["keypoints"] / f"{r['clip_id']}_keypoints.csv"
        if out.exists() and not args.overwrite:
            print(f"[{n}/{len(man)}] skip (이미 있음) {r['clip_id']}")
            continue
        frames = extract(model, ROOT / r["video_file"], out, config, device)
        print(f"[{n}/{len(man)}] {r['clip_id']}: {frames} frames → {out.name}")


if __name__ == "__main__":
    main()
