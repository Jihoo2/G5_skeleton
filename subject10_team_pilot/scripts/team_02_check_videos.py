"""
[팀 02] Manifest의 모든 MP4를 검사 (교과 4일차 STEP 2)

파일 존재 · 재생(끝까지 디코딩) · FPS · Frame 수 · 해상도 · 영상 방향을 확인하고
reports/team_projects/<team>/tables/video_inventory.csv 에 저장합니다.

실행
  python scripts/team_02_check_videos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_team_config, team_paths  # noqa: E402


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    rows = []
    for _, r in man.iterrows():
        path = ROOT / r["video_file"]
        info = dict(clip_id=r["clip_id"], exists=path.exists())
        if path.exists():
            cap = cv2.VideoCapture(str(path))
            info.update(opened=cap.isOpened(), fps=round(cap.get(cv2.CAP_PROP_FPS), 2),
                        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            read = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                read += 1
            cap.release()
            info["read_frames"] = read
            info["duration_s"] = round(read / info["fps"], 2) if info["fps"] else None
            info["orientation"] = "portrait" if info["height"] > info["width"] else "landscape"
        problems = []
        if not info["exists"]:
            problems.append("파일 없음")
        else:
            if not info.get("opened") or info.get("read_frames", 0) == 0:
                problems.append("재생 불가")
            if abs(info.get("fps", 0) - cfg["fps"]) > 1:
                problems.append(f"FPS {info.get('fps')}")
            if info.get("frame_count", 0) and info.get("read_frames", 0) < info["frame_count"] * 0.95:
                problems.append("일부 Frame 디코딩 실패")
        info["problems"] = "; ".join(problems)
        rows.append(info)

    out = pd.DataFrame(rows)
    path = p["tables"] / "video_inventory.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(out[["clip_id", "fps", "read_frames", "width", "height", "problems"]].to_string(index=False))
    print(f"\n저장: {path}")
    n_bad = (out["problems"] != "").sum()
    print(f"문제 있는 영상: {n_bad} / {len(out)}")


if __name__ == "__main__":
    main()
