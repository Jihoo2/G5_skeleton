"""교과 5일차 형식(team_12 ~ team_16)에서 같이 쓰는 경로·Clip 이름 규칙.

팀 Clip ID  : T05_S01_A008_T01_front / EXT_S04_A008_T01_diag
교재 형식   : subject=S01 / EXT_S04, action_id=A008, take=T01  (+ 팀 추가: view)
결과 위치   : reports/team_projects/<team>/day05/   (교재의 data/day05/ 와 같은 하위 구조)
"""
from __future__ import annotations

import re
from pathlib import Path

from src.team_feature_utils import team_paths

CLIP_PATTERN = re.compile(r"^(?:T\d{2}_)?((?:EXT_)?S\d{2})_(A\d{3})_(T\d{2})_(\w+)$")


def parse_team_clip(clip_id: str) -> dict:
    m = CLIP_PATTERN.match(clip_id)
    if m is None:
        raise ValueError(f"Clip 이름 규칙 오류: {clip_id}")
    subject, action_id, take, view = m.groups()
    return dict(subject=subject, action_id=action_id, take=take, view=view)


def day05_paths(cfg: dict) -> dict:
    p = team_paths(cfg)
    root = p["reports"] / "day05"
    d = {
        "root": root,
        "keypoints": root / "keypoints",
        "manifests": root / "manifests",
        "qa": root / "qa",
        "splits": root / "splits",
        "tables": root / "tables",
    }
    for k, v in d.items():
        v.mkdir(parents=True, exist_ok=True)
    d.update(
        keypoint_manifest=d["manifests"] / "keypoint_dataset_manifest.csv",
        qa=d["qa"] / "keypoint_clip_qa.csv",
        review=d["qa"] / "keypoint_clip_review.csv",
        approved=d["manifests"] / "approved_clip_manifest.csv",
        split_config=Path(p["manifest"]).parent / "day05_subject_split.json",
    )
    return d


SPLIT_NAMES = ["train", "val", "val_ext", "test", "test_ext"]
