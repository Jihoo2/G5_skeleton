"""
[팀 05] Clip별 품질 요약 → 검수 상태 제안 (교과 4일차 STEP 5)

Clip마다 계산
  frames, valid_ratio(정규화 가능 Frame 비율), mean_conf,
  key_low_ratio(무릎·고관절 각도에 쓰이는 관절이 낮은 Frame 비율),
  max_gap_frames(연속 누락 최대 길이), reps(찾은 1회 구간 수), people_max

제안 상태 (suggested_status)
  PASS    : valid ≥ 90%, key_low ≤ 20%, 1회 구간 ≥ 1
  REVIEW  : 일부 기준 미달 → 사람이 영상 확인
  RESHOOT : valid < 50% 또는 1회 구간을 못 찾음
→ 최종 review_status는 사람이 Manifest에 직접 기입합니다 (자동으로 덮어쓰지 않음).

실행
  python scripts/team_05_sequence_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import (  # noqa: E402
    load_team_config, rep_metrics, resolved_series, team_paths,
)


def max_gap(valid: np.ndarray) -> int:
    best = cur = 0
    for v in valid:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    kp_conf, fps = cfg["keypoint_confidence"], cfg["fps"]
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    rows = []
    for _, r in man.iterrows():
        sp = p["sequences"] / f"{r['clip_id']}_sequence.csv"
        if not sp.exists():
            rows.append(dict(clip_id=r["clip_id"], suggested_status="RESHOOT", reason="sequence 없음"))
            continue
        seq = pd.read_csv(sp)
        comb = cfg["actions"].get(r["action_id"], {}).get("leg_combine", "mean")
        ser = resolved_series(seq, r["view"], kp_conf, comb)
        key_low = float(np.mean(~np.isfinite(ser["knee"]) | ~np.isfinite(ser["hip"])))
        reps, _ = rep_metrics(seq, r["view"], fps, kp_conf, comb)
        kp = pd.read_csv(p["keypoints"] / f"{r['clip_id']}_keypoints.csv")
        valid = seq["valid_pose"].to_numpy().astype(bool)
        info = dict(
            clip_id=r["clip_id"], subject_id=r["subject_id"], action_label=r["action_label"],
            view=r["view"], source_type=r["source_type"], frames=len(seq),
            valid_ratio=round(valid.mean(), 3),
            mean_conf=round(float(seq["mean_keypoint_conf"].mean()), 3),
            key_low_ratio=round(key_low, 3), max_gap_frames=max_gap(valid),
            reps=len(reps), people_max=int(kp["people"].max()) if "people" in kp else 1,
            side_used=ser["side"],
        )
        reasons = []
        if info["valid_ratio"] < 0.5 or info["reps"] == 0:
            status = "RESHOOT"
            reasons.append("정규화 가능 Frame 부족" if info["valid_ratio"] < 0.5 else "1회 구간 없음")
        else:
            status = "PASS"
            if info["valid_ratio"] < 0.9:
                status, _ = "REVIEW", reasons.append("valid < 90%")
            if info["key_low_ratio"] > 0.2:
                status, _ = "REVIEW", reasons.append("무릎·고관절 Confidence 낮은 구간 많음")
            if info["people_max"] > 1:
                status, _ = "REVIEW", reasons.append("여러 사람 검출")
            if r["source_type"] == "RGB" and info["reps"] > 1:
                status, _ = "REVIEW", reasons.append(f"1 Take인데 {info['reps']}회 검출")
        info["suggested_status"] = status
        info["reason"] = "; ".join(reasons)
        rows.append(info)

    out = pd.DataFrame(rows)
    path = p["tables"] / "sequence_qa.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    show = ["clip_id", "frames", "valid_ratio", "mean_conf", "reps", "people_max", "suggested_status", "reason"]
    print(out[[c for c in show if c in out]].to_string(index=False))
    print(f"\n저장: {path}")
    print(out["suggested_status"].value_counts().to_string())
    print("\n→ 확인 후 Manifest의 review_status 를 PASS / REVIEW / RESHOOT / EXCLUDE 로 직접 기입하세요.")


if __name__ == "__main__":
    main()
