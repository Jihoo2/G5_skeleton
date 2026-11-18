"""
[팀 13] 교과 5일차 27번 — Clip별 자동 Keypoint QA (AUTO_PASS / CHECK) + Human QA 작업표

입력  reports/.../day05/manifests/keypoint_dataset_manifest.csv  (team_12)
출력  reports/.../day05/qa/keypoint_clip_qa.csv       자동 QA 결과
      reports/.../day05/qa/keypoint_clip_review.csv   Human QA 작업표

자동 QA 기준 (교재와 동일)
  pose_ratio < 0.90 / single_person_ratio < 0.90 / mean_keypoint_conf < 0.50 /
  low_critical_ratio > 0.20  →  하나라도 해당하면 CHECK, 아니면 AUTO_PASS
  low_critical_ratio = 동작별 중요 관절 Confidence 중 keypoint_confidence 미만인 비율

Human QA 작업표
  교재는 매번 UNREVIEWED로 덮어쓰지만, 팀은 사람 판정을 Manifest의 review_status에 기록해 왔으므로
  그 값을 human_decision으로 옮겨 적습니다 → 다시 실행해도 사람 판정이 지워지지 않습니다.
    PASS / RESHOOT / EXCLUDE → 그대로,  REVIEW / PENDING / 빈칸 → UNREVIEWED
  사람 판정을 바꿀 때는 Manifest(configs/team05_clip_manifest.csv)를 고치고 이 스크립트를 다시 실행하세요.

용어 주의: 여기의 CHECK(자동 QA: 사람이 먼저 볼 Clip)는
          team_09 판정의 CHECK(1회 구간 측정 불가)와 다른 뜻입니다.

실행
  python scripts/team_13_keypoint_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import day05_paths  # noqa: E402
from src.team_feature_utils import load_team_config, team_paths  # noqa: E402

LEG = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SHOULDER = ["left_shoulder", "right_shoulder"]
ACTION_CRITICAL_KEYPOINTS = {
    "burpee": SHOULDER + LEG,               # 몸통 일직선(어깨–골반–발목) + 다리 굽힘
    "lunge": LEG,                           # 앞무릎·고관절·정강이
    "squat_jump": LEG,                      # 깊이·무릎·무릎 벌어짐
    "plank_kick_through": SHOULDER + LEG,   # 범위 제외 동작이지만 정의는 남김
}
DECISION_MAP = {"PASS": "PASS", "RESHOOT": "RESHOOT", "EXCLUDE": "EXCLUDE"}


def inspect(csv_path: Path, action_label: str, threshold: float) -> dict:
    df = pd.read_csv(csv_path)
    names = ACTION_CRITICAL_KEYPOINTS.get(action_label)
    if names is None:
        raise SystemExit(f"[중지] 중요 관절 정의가 없는 동작: {action_label}")
    if df.empty:
        return dict(qa_status="CHECK", frame_count=0, pose_ratio=0.0, single_person_ratio=0.0,
                    mean_keypoint_conf=0.0, low_critical_ratio=1.0, missing_pose_frames=0,
                    critical_keypoints="|".join(names), qa_reason="빈 CSV")
    pose = pd.to_numeric(df["pose_detected"], errors="coerce").fillna(0)
    people = pd.to_numeric(df["person_count"], errors="coerce").fillna(0)
    conf_cols = [c for c in df.columns if c.endswith("_conf") and c != "selected_box_conf"]
    conf = df[conf_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    crit = df[[f"{n}_conf" for n in names]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(crit)

    r = dict(
        frame_count=len(df),
        pose_ratio=float(pose.mean()),
        single_person_ratio=float((people == 1).mean()),
        mean_keypoint_conf=float(np.nanmean(conf)) if np.isfinite(conf).any() else 0.0,
        low_critical_ratio=float(((crit < threshold) & valid).sum() / valid.sum()) if valid.any() else 1.0,
        missing_pose_frames=int((pose == 0).sum()),
    )
    reasons = []
    if r["pose_ratio"] < 0.90:
        reasons.append("pose_ratio<0.90")
    if r["single_person_ratio"] < 0.90:
        reasons.append("single_person_ratio<0.90")
    if r["mean_keypoint_conf"] < 0.50:
        reasons.append("mean_keypoint_conf<0.50")
    if r["low_critical_ratio"] > 0.20:
        reasons.append("low_critical_ratio>0.20")
    r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    return dict(qa_status="CHECK" if reasons else "AUTO_PASS", **r,
                critical_keypoints="|".join(names), qa_reason="; ".join(reasons))


def main() -> None:
    cfg = load_team_config()
    d = day05_paths(cfg)
    if not d["keypoint_manifest"].exists():
        raise SystemExit("[중지] keypoint_dataset_manifest.csv 없음 → team_12 먼저 실행")
    man = pd.read_csv(d["keypoint_manifest"], dtype=str).fillna("")
    notes = pd.read_csv(team_paths(cfg)["manifest"], dtype=str).fillna("").set_index("clip_id")["note"]
    threshold = float(cfg.get("keypoint_confidence", 0.50))

    rows = []
    for _, item in man.iterrows():
        csv_path = ROOT / item["keypoint_csv"]
        if not csv_path.exists():
            raise SystemExit(f"[중지] Keypoint CSV 없음: {csv_path}")
        qa = inspect(csv_path, item["action_label"], threshold)
        rows.append(dict(
            clip_id=item["clip_id"], subject=item["subject"], action_id=item["action_id"],
            action_label=item["action_label"], take=item["take"], view=item["view"],
            source_type=item["source_type"], video_path=item["video_path"],
            keypoint_csv=item["keypoint_csv"], **qa,
            human_decision=DECISION_MAP.get(item["review_status"].strip().upper(), "UNREVIEWED"),
            human_note=notes.get(item["clip_id"], ""),
        ))
        print(f"{item['clip_id']:26s} {qa['qa_status']:9s} pose={qa['pose_ratio']:.3f} "
              f"single={qa['single_person_ratio']:.3f} conf={qa['mean_keypoint_conf']:.3f} "
              f"low={qa['low_critical_ratio']:.3f}  {qa['qa_reason']}")

    out = pd.DataFrame(rows)

    out.drop(columns=["human_decision", "human_note"]).to_csv(d["qa"], index=False, encoding="utf-8-sig")
    out.to_csv(d["review"], index=False, encoding="utf-8-sig")
    print()
    print(out["qa_status"].value_counts().to_string())
    print(f"\nQA     : {d['qa']}")
    print(f"Review : {d['review']}  (human_decision은 Manifest의 review_status에서 가져옴)")


if __name__ == "__main__":
    main()
