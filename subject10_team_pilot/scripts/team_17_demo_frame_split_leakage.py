"""
[팀 17] 교과 5일차 32번 — 일부러 잘못 나눈 Random Frame Split 으로 Leakage 확인 (교육용 비교)

입력  reports/.../day05/manifests/approved_clip_manifest.csv 의 Keypoint CSV
처리  모든 Frame을 한 표로 합침 → 섞어서 80% / 20% (random_state=42) → Subject·Clip 겹침 계산
      팀 데이터는 한 Clip 안에 여러 회 반복이 있으므로 '같은 1회 동작'이 양쪽에 나뉜 수도 보여 줌
      (1회 구간 = 같은 Clip 안에서 1초 간격 덩어리로 근사)
출력  화면 + reports/.../day05/tables/random_frame_split_demo.csv
      실제 학습·평가용 split 파일은 만들지 않고, team_15 결과도 건드리지 않음

실행
  python scripts/team_17_demo_frame_split_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import day05_paths  # noqa: E402
from src.team_feature_utils import load_team_config  # noqa: E402


def main() -> None:
    cfg = load_team_config()
    d = day05_paths(cfg)
    if not d["approved"].exists():
        raise SystemExit("[중지] approved_clip_manifest.csv 없음 → team_14 먼저 실행")
    man = pd.read_csv(d["approved"], dtype=str).fillna("")

    frames = []
    for _, r in man.iterrows():
        df = pd.read_csv(ROOT / r["keypoint_csv"], usecols=["clip_id", "frame_id", "time_sec", "subject"])
        df["second"] = df["clip_id"] + "@" + np.floor(df["time_sec"]).astype(int).astype(str)
        frames.append(df)
    all_frames = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(all_frames))
    cut = int(len(all_frames) * 0.8)
    train, test = all_frames.iloc[idx[:cut]], all_frames.iloc[idx[cut:]]

    subj = sorted(set(train["subject"]) & set(test["subject"]))
    clips = set(train["clip_id"]) & set(test["clip_id"])
    secs = set(train["second"]) & set(test["second"])

    print("Random Frame Split Demo (팀 데이터)\n")
    print(f"Clips        : {man['clip_id'].nunique()}개 / Frames {len(all_frames)}개")
    print(f"Train frames : {len(train)}")
    print(f"Test frames  : {len(test)}\n")
    print(f"Subject overlap    : {len(subj)}명 / 전체 {all_frames['subject'].nunique()}명")
    print(f"Clip overlap       : {len(clips)}개 / 전체 {all_frames['clip_id'].nunique()}개")
    print(f"같은 1초 구간 overlap: {len(secs)}개 / 전체 {all_frames['second'].nunique()}개 "
          "(바로 옆 Frame이 Train·Test로 갈라진 구간)\n")
    if subj:
        print("[LEAKAGE] 같은 Subject가 Train과 Test에 존재")
    if clips:
        print("[LEAKAGE] 같은 Clip의 Frame이 Train과 Test에 존재")
    print("\n이 Random Frame Split은 최종 평가용으로 사용하지 않습니다. (team_15 Subject-wise Split 사용)")

    pd.DataFrame([dict(train_frames=len(train), test_frames=len(test),
                       subject_overlap=len(subj), subjects_total=all_frames["subject"].nunique(),
                       clip_overlap=len(clips), clips_total=all_frames["clip_id"].nunique(),
                       second_overlap=len(secs), seconds_total=all_frames["second"].nunique())]
                 ).to_csv(d["tables"] / "random_frame_split_demo.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
