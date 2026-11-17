"""
[팀 06] Subject-wise Split (교과 5일차, 4일차 3장)

- configs/team_project.json 의 split 에 적은 사람 단위로 train / val / test 를 나눕니다.
- 같은 사람의 영상(정면·측면 동시 촬영본 포함)은 반드시 같은 쪽에 들어갑니다.
- 외부 영상(EXT_)은 기본으로 test_ext 에 넣습니다 (처음 보는 사람·환경 검증용).
- 결과를 Manifest의 split 열에 쓰고 tables/dataset_split.csv 로도 저장합니다.

실행
  python scripts/team_06_make_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_team_config, team_paths  # noqa: E402


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    lookup = {s: split for split, subjects in cfg["split"].items() for s in subjects}

    def assign(row):
        if row["source_type"] == "EXT":
            return "test_ext"
        return lookup.get(row["subject_id"], "unassigned")

    man["split"] = man.apply(assign, axis=1)
    man.to_csv(p["manifest"], index=False, encoding="utf-8-sig")

    table = man.groupby(["split", "subject_id", "action_label"]).size().rename("clips").reset_index()
    table.to_csv(p["tables"] / "dataset_split.csv", index=False, encoding="utf-8-sig")
    print(man.groupby(["split", "action_label"]).size().unstack(fill_value=0).to_string())

    # 누수 검사: 한 사람이 두 split에 걸쳐 있으면 안 됨
    leak = man.groupby("subject_id")["split"].nunique()
    if (leak > 1).any():
        print("\n[경고] 여러 split에 걸친 사람:", list(leak[leak > 1].index))
    if (man["split"] == "unassigned").any():
        print("\n[경고] split 미지정 사람:", sorted(man.loc[man.split == 'unassigned', 'subject_id'].unique()))
        print("→ configs/team_project.json 의 split 에 추가하세요.")


if __name__ == "__main__":
    main()
