"""
[팀 04] Keypoint CSV → Clip별 Sequence CSV (교과 4일차 STEP 4, 22번 코드의 팀 확장)

출력: reports/team_projects/<team>/sequences/<clip_id>_sequence.csv
  - 출처: clip_id, subject_id, action_label, view, source_type
  - 시간: frame_id, time_sec
  - 품질: valid_pose, mean_keypoint_conf, low_keypoint_count, <관절>_conf (17개)
  - 교과 38-D: <관절>_x, <관절>_y (34개) + 팔꿈치·무릎 각도 4개
  - 정규화 기준: norm_scale_mode (shoulder=교과 방식 / torso=측면용), norm_scale_px
  - 팀 Feature: left/right_hip_angle, left/right_leg_ratio, leg_ratio, left/right_body_line_offset

clip_id별로 저장하므로 같은 동작을 여러 번 찍어도 덮어쓰지 않습니다.

실행
  python scripts/team_04_build_sequences.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import (  # noqa: E402
    build_frame_features, load_keypoints_csv, load_team_config, team_paths,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--scale", default="auto", choices=["auto", "shoulder", "torso"],
                    help="auto: 측면은 몸통 길이, 그 외는 교과 방식(어깨 폭)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_team_config()
    p = team_paths(cfg)
    kp_conf = cfg["keypoint_confidence"]
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    if args.clip:
        man = man[man["clip_id"] == args.clip]
    man = man[man["review_status"] != "EXCLUDE"]

    done = 0
    for _, r in man.iterrows():
        kp_path = p["keypoints"] / f"{r['clip_id']}_keypoints.csv"
        if not kp_path.exists():
            print(f"skip (keypoints 없음): {r['clip_id']}")
            continue
        fid, t, xy, conf = load_keypoints_csv(kp_path)
        seq = build_frame_features(fid, t, xy, conf, r["view"], kp_conf, args.scale)
        for i, c in enumerate(["clip_id", "subject_id", "action_label", "view", "source_type"]):
            seq.insert(i, c, r[c])
        out = p["sequences"] / f"{r['clip_id']}_sequence.csv"
        seq.to_csv(out, index=False, encoding="utf-8-sig")
        done += 1
        print(f"{r['clip_id']}: {len(seq)} frames, valid {seq['valid_pose'].mean():.0%}, "
              f"scale={seq['norm_scale_mode'].iloc[0]} → {out.name}")
    print(f"\nSequence CSV {done}개 생성: {p['sequences']}")


if __name__ == "__main__":
    main()
