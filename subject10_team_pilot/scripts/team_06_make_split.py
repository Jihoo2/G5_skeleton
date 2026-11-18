"""
[팀 06] Subject-wise Split (교과 5일차 PART 6)

- configs/team_project.json 의 split 에 적은 사람 단위로 나눕니다.
    train    : 팀 RGB — 정석 기준(team_08)을 만드는 데 사용
    val      : 팀 RGB — 규칙 숫자 조정에 사용
    val_ext  : 외부 영상 — 규칙 숫자 조정에 사용 (지금까지 본 외부 영상 전부)
    test     : 팀 RGB — 규칙 동결 후 한 번만 평가
    test_ext : 외부 영상 — 규칙 동결 후 한 번만 평가 (아직 한 번도 보지 않은 사람만)
- 같은 사람의 영상(정면·측면 동시 촬영본 포함)은 반드시 같은 split 에 들어갑니다.
- 외부 영상도 자동으로 배정하지 않습니다. 새 외부 영상은 split 에 사람 ID를 직접 적어야 합니다.
  (예전에는 EXT_ 를 모두 test_ext 로 넣었으나, 규칙 조정에 쓴 영상이 test 로 들어가는
   누수가 생겨 2026-11-18 에 val_ext / test_ext 로 나눔)
- 결과를 Manifest의 split 열에 쓰고 tables/dataset_split.csv 로도 저장합니다.
- 한 사람이 두 split 에 적혀 있으면 오류로 멈춥니다. split 미지정 사람은 경고합니다.

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

SPLITS = ["train", "val", "val_ext", "test", "test_ext"]
EXT_ONLY = {"val_ext", "test_ext"}   # 외부 영상만 들어가는 split
TEAM_ONLY = {"train", "val", "test"}  # 팀 RGB만 들어가는 split


def build_lookup(split_cfg: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    dup: dict[str, list[str]] = {}
    for name in SPLITS:
        for s in split_cfg.get(name, []):
            if s in lookup:
                dup.setdefault(s, [lookup[s]]).append(name)
            lookup[s] = name
    unknown = [k for k in split_cfg if k not in SPLITS and k != "note"]
    if unknown:
        raise SystemExit(f"[오류] 알 수 없는 split 이름: {unknown} (사용 가능: {SPLITS})")
    if dup:
        raise SystemExit(f"[오류] 한 사람이 여러 split 에 적혀 있습니다(Subject Leakage): {dup}")
    return lookup


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    lookup = build_lookup(cfg["split"])
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    man["split"] = man["subject_id"].map(lookup).fillna("unassigned")

    # 외부 영상이 팀 split 에, 팀 영상이 외부 split 에 들어가면 설정 실수
    wrong = man[((man["source_type"] == "EXT") & man["split"].isin(TEAM_ONLY))
                | ((man["source_type"] == "RGB") & man["split"].isin(EXT_ONLY))]
    if len(wrong):
        print(wrong[["clip_id", "subject_id", "source_type", "split"]].to_string(index=False))
        raise SystemExit("[오류] 팀 RGB 는 train/val/test, 외부 영상은 val_ext/test_ext 에만 넣으세요.")

    man.to_csv(p["manifest"], index=False, encoding="utf-8")

    table = man.groupby(["split", "subject_id", "action_label"]).size().rename("clips").reset_index()
    table.to_csv(p["tables"] / "dataset_split.csv", index=False, encoding="utf-8-sig")
    print(man.groupby(["split", "action_label"]).size().unstack(fill_value=0).to_string())

    if (man["split"] == "unassigned").any():
        print("\n[경고] split 미지정 사람:", sorted(man.loc[man.split == "unassigned", "subject_id"].unique()))
        print("→ configs/team_project.json 의 split 에 추가하세요. "
              "새 외부 영상은 test_ext 에 넣고, 규칙 동결 전까지 판정하지 않습니다.")


if __name__ == "__main__":
    main()
