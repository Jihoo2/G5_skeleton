"""
[팀 01] 영상 폴더를 읽어 Clip Manifest 만들기 (교과 4일차 STEP 1)

파일명 규칙
  팀 촬영 : T05_S01_A008_T01_side.mp4   (팀_사람_동작_Take_방향)
  외부 영상: EXT_S01_A008_T01_front.mp4  (외부 영상은 사람마다 S번호를 따로)
  방향     : front / diag / side / deg30 처럼 각도 직접 표기도 가능

이미 Manifest가 있으면 사람이 직접 고친 열(form_label, review_status, note)은 유지합니다.

실행
  python scripts/team_01_make_manifest.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_team_config, team_paths  # noqa: E402

PATTERN = re.compile(
    r"^(?P<team>T\d{2}|EXT)_(?P<subject>S\d{2})_(?P<action>A\d{2,3})_"
    r"(?P<take>T\d{2})_(?P<view>front|diag|side|deg\d{1,3})$"
)
COLUMNS = ["clip_id", "subject_id", "action_id", "action_label", "take_id", "view", "view_deg",
           "form_label", "source_type", "video_file", "split", "review_status", "note"]
KEEP_MANUAL = ["form_label", "review_status", "note"]


def view_to_deg(view: str, views: dict) -> int:
    return int(view[3:]) if view.startswith("deg") else int(views[view])


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    rows, bad = [], []
    for folder, source in ((p["rgb_videos"], "RGB"), (p["ext_videos"], "EXT")):
        for video in sorted(folder.glob("*.mp4")):
            m = PATTERN.match(video.stem)
            if not m:
                bad.append(video.name)
                continue
            g = m.groupdict()
            if g["action"] not in cfg["actions"]:
                bad.append(f"{video.name} (설정에 없는 동작 {g['action']})")
                continue
            subject = f"EXT_{g['subject']}" if g["team"] == "EXT" else g["subject"]
            rows.append(dict(
                clip_id=video.stem, subject_id=subject, action_id=g["action"],
                action_label=cfg["actions"][g["action"]]["label"], take_id=g["take"],
                view=g["view"], view_deg=view_to_deg(g["view"], cfg["views_deg"]),
                form_label="OK", source_type=source,
                video_file=str(video.relative_to(ROOT)), split="", review_status="PENDING", note="",
            ))

    new = pd.DataFrame(rows, columns=COLUMNS)
    if p["manifest"].exists() and len(new):
        old = pd.read_csv(p["manifest"], dtype=str).fillna("")
        new = new.merge(old[["clip_id"] + KEEP_MANUAL], on="clip_id", how="left", suffixes=("", "_old"))
        for c in KEEP_MANUAL:
            new[c] = new[f"{c}_old"].where(new[f"{c}_old"].fillna("") != "", new[c])
            new = new.drop(columns=f"{c}_old")
    new.to_csv(p["manifest"], index=False, encoding="utf-8-sig")

    print(f"Manifest: {p['manifest']}  ({len(new)}개 영상)")
    if len(new):
        print(new.groupby(["source_type", "action_id", "view"]).size().rename("clips").to_string())
        dup = new["clip_id"].duplicated().sum()
        if dup:
            print(f"[경고] 중복 clip_id {dup}개")
    if bad:
        print("\n[파일명 규칙에 맞지 않아 제외]")
        for b in bad:
            print("  -", b)


if __name__ == "__main__":
    main()
