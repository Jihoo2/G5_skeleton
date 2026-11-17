"""
[팀 07] VICON GLB → 촬영 각도별 기준(Reference) 만들기

교과 규칙 반영
  - VICON은 RGB와 분리된 별도 실험 자료 → 여기서는 '정석 동작의 모양(곡선)'과 비교 기준값으로만 사용
  - VICON에는 YOLO Confidence가 없으므로 conf = NaN (1.0으로 채우지 않음)
  - VICON 수행자·각도별 투영본은 새 Subject가 아님 → Split·학습 데이터에 섞지 않음

처리
  data/team_projects/<team>/vicon/<glb>  (configs/team_project.json 의 actions.*.glb)
  → 촬영 각도(config의 views_deg + Manifest에 등장한 degNN 각도)마다 2D 투영
  → 웹캠과 같은 build_frame_features → 1회 구간 분할 → 측정값 + 48점 평균 곡선

출력 (reports/team_projects/<team>/reference/)
  <action>_<view>_reps.csv       GLB 1회 구간별 측정값
  <action>_<view>_template.json  GLB 평균 곡선(knee/hip/leg_ratio 48점) + 측정값 평균

실행
  python scripts/team_07_build_reference.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import (  # noqa: E402
    REP_METRICS, build_frame_features, load_team_config, metrics_for, project_glb, rep_metrics,
    team_paths,
)


def view_to_deg(view: str, cfg: dict) -> float:
    if view in cfg["views_deg"]:
        return float(cfg["views_deg"][view])
    m = re.fullmatch(r"deg(\d{1,3})", view)
    if not m:
        raise ValueError(f"알 수 없는 view: {view}")
    return float(m.group(1))


def all_views(cfg: dict, manifest_path: Path) -> list[str]:
    views = list(cfg["views_deg"])
    if manifest_path.exists():
        man = pd.read_csv(manifest_path, dtype=str).fillna("")
        views += [v for v in man["view"].unique() if v and v not in views]
    return views


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    fps, kp_conf = cfg["fps"], cfg["keypoint_confidence"]
    summary = []
    for aid, a in cfg["actions"].items():
        if not a.get("glb"):
            print(f"[{aid}] GLB 지정 안 됨 → 팀 정석 규칙(rules)과 팀 영상으로만 판정")
            continue
        glb = p["vicon"] / a["glb"]
        if not glb.exists():
            print(f"[{aid}] GLB 없음 → {glb} 에 넣으세요")
            continue
        for view in all_views(cfg, p["manifest"]):
            deg = view_to_deg(view, cfg)
            fid, t, xy, conf = project_glb(glb, deg, fps)
            seq = build_frame_features(fid, t, xy, conf, view, kp_conf)
            reps, curves = rep_metrics(seq, view, fps, kp_conf, a.get("leg_combine", "mean"))
            stem = f"{aid}_{view}"
            reps.insert(0, "action_id", aid)
            reps.insert(1, "view", view)
            reps.to_csv(p["reference"] / f"{stem}_reps.csv", index=False, encoding="utf-8-sig")
            if len(reps) == 0:
                print(f"[{aid} {view}] 1회 구간을 찾지 못함 (다리 높이 변화가 작은 동작) → 템플릿 없음")
                summary.append(dict(action_id=aid, view=view, view_deg=deg, glb_reps=0))
                continue
            template = {
                "action_id": aid, "view": view, "view_deg": deg, "glb": a["glb"],
                "n_reps": len(reps), "scale_mode": seq["norm_scale_mode"].iloc[0],
                "curve": {k: np.nanmean([c[k] for c in curves], axis=0).round(4).tolist()
                          for k in ("knee", "hip", "leg_ratio")},
                "metrics": {m: (None if not np.isfinite(reps[m]).any()
                                else round(float(np.nanmean(reps[m])), 4))
                            for m in REP_METRICS + ["duration_s"]},
            }
            (p["reference"] / f"{stem}_template.json").write_text(
                json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.append(dict(action_id=aid, view=view, view_deg=deg, glb_reps=len(reps),
                                **{m: template["metrics"][m] for m in metrics_for(a, view, cfg)
                                   if m in template["metrics"]}))
            print(f"[{aid} {view:>6}] GLB {len(reps)}회 → {stem}_template.json")
    pd.DataFrame(summary).to_csv(p["tables"] / "reference_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {p['reference']}")


if __name__ == "__main__":
    main()
