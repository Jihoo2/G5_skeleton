"""
[팀 15] 교과 5일차 30번 — Approved Manifest를 Subject 기준으로 Split CSV로 나누기

입력  reports/.../day05/manifests/approved_clip_manifest.csv  (team_14)
      configs/team_project.json 의 split  (사람 배정은 이 한 곳에서만 관리)
출력  configs/day05_subject_split.json        교재 형식 내보내기 (자동 생성 — 직접 고치지 않기)
      reports/.../day05/splits/{train,val,val_ext,test,test_ext}.csv
        1행 = 1 Clip (clip_id, subject, action_id, action_label, view, keypoint_csv …)
        실제 영상·Keypoint는 옮기지 않고 '사용 목록'만 만듭니다.
      reports/.../day05/tables/split_coverage.csv  split × (동작, 각도) 판정 가능 Clip 수

검사 (교재 30번 + 팀 규칙)
  - 한 사람이 두 split에 배정 → 중지
  - 승인 데이터에 있는데 split 미배정 → 중지 / 설정에만 있는 사람 → 경고
  - (동작, 판정하는 각도) 포함 여부는 표로 출력 (우리 데이터는 각도별로 비어 있는 칸이 있어 경고만)
    --strict 를 주면 train·val 에서 판정 각도 하나라도 비면 중지

실행
  python scripts/team_15_build_subject_split.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import SPLIT_NAMES, day05_paths  # noqa: E402
from src.team_feature_utils import load_team_config, metrics_for, view_group  # noqa: E402


def judged_cells(cfg: dict) -> list[tuple[str, str]]:
    """판정하는 (동작, 각도) 목록 — 범위 제외 동작(A006)은 뺌"""
    out = []
    for aid, a in cfg["actions"].items():
        if a.get("excluded") or a["label"] == "plank_kick_through":
            continue
        for v in ("front", "diag", "side"):
            if metrics_for(a, v, cfg):
                out.append((aid, v))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    cfg = load_team_config()
    d = day05_paths(cfg)
    split_cfg = {k: list(cfg["split"].get(k, [])) for k in SPLIT_NAMES}

    # 교재 형식 설정 파일 내보내기
    export = {f"{k}_subjects": v for k, v in split_cfg.items()}
    export["note"] = "team_15가 configs/team_project.json 의 split 에서 자동 생성. 사람 배정은 team_project.json 에서 고치세요."
    d["split_config"].write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    seen: dict[str, str] = {}
    for name, subjects in split_cfg.items():
        for s in subjects:
            if s in seen:
                raise SystemExit(f"[중지] {s} 가 {seen[s]} 와 {name} 에 동시에 배정됨 (Subject Leakage)")
            seen[s] = name

    if not d["approved"].exists():
        raise SystemExit("[중지] approved_clip_manifest.csv 없음 → team_14 먼저 실행")
    df = pd.read_csv(d["approved"], dtype=str).fillna("")
    data_subjects = set(df["subject"])
    unassigned = sorted(data_subjects - set(seen))
    if unassigned:
        raise SystemExit(f"[중지] split 미배정 Subject: {unassigned} → team_project.json 의 split 에 추가")
    missing = sorted(set(seen) - data_subjects)
    if missing:
        print(f"[경고] 설정에는 있지만 승인 데이터에 없는 Subject: {missing}")

    df["split"] = df["subject"].map(seen)
    df["view_group"] = df["view"].map(lambda v: view_group(v, cfg))
    for name in SPLIT_NAMES:
        part = df[df["split"] == name]
        path = d["splits"] / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        subj = sorted(part["subject"].unique())
        print(f"[{name.upper():8s}] Subjects {len(subj):2d}명, Clips {len(part):2d}개 → {path.name}"
              + ("  (비어 있음)" if part.empty else ""))

    cells = judged_cells(cfg)
    label = {aid: cfg["actions"][aid]["label"] for aid, _ in cells}
    cov = pd.DataFrame(
        [{"action": f"{aid} {label[aid]}", "view": v,
          **{s: int(((df["split"] == s) & (df["action_id"] == aid) & (df["view_group"] == v)).sum())
             for s in SPLIT_NAMES}} for aid, v in cells])
    cov.to_csv(d["tables"] / "split_coverage.csv", index=False, encoding="utf-8-sig")
    print("\n[판정 각도별 Clip 수]")
    print(cov.to_string(index=False))

    gaps = cov[(cov["train"] == 0) | ((cov["val"] == 0) & (cov["val_ext"] == 0))]
    if len(gaps):
        print("\n[경고] train 이 비었거나 val·val_ext 모두 빈 판정 각도:")
        print(gaps[["action", "view", "train", "val", "val_ext"]].to_string(index=False))
        if args.strict:
            raise SystemExit("[중지] --strict: 판정 각도 Coverage 부족")
    print("\nSubject-wise split: COMPLETE")


if __name__ == "__main__":
    main()
