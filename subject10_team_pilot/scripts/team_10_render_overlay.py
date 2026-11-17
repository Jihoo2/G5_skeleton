"""
[팀 10] 판정 결과 확인 영상 만들기 (뼈대 + 1회 구간 판정 표시)

숫자만 보고는 NG가 '진짜 틀린 자세'인지 '측정 문제'인지 알 수 없으므로,
영상 위에 YOLO 관절·뼈대와 1회 구간별 판정(OK/NG/CHECK)·벗어난 항목을 그려서 눈으로 확인합니다.

출력: reports/team_projects/<team>/overlays/<clip_id>_overlay.mp4
  - 관절 색: Confidence ≥ 0.50 초록 / 0.30~0.50 노랑 / 0.30 미만 빨강 (교과 2일차 기준)
  - 위쪽 막대: 1회 구간 위치 (OK 초록 / NG 빨강 / CHECK 회색), 현재 위치 흰 선
  - 1회 구간 안에서는 판정과 벗어난 항목(예: depth>0.55)을 표시
  ※ OpenCV는 한글을 그리지 못해 영문 항목명으로 표시 (메시지는 rep_judgments.csv 참고)

실행
  python scripts/team_10_render_overlay.py --clip EXT_S04_A008_T01_diag
  python scripts/team_10_render_overlay.py --all        # 판정한 모든 Clip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_keypoints_csv, load_team_config, team_paths  # noqa: E402

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLOR = {"OK": (60, 200, 60), "NG": (40, 40, 230), "CHECK": (160, 160, 160)}


def conf_color(c: float) -> tuple:
    if not np.isfinite(c) or c < 0.30:
        return (40, 40, 230)
    return (60, 200, 60) if c >= 0.50 else (0, 220, 255)


def render(clip_id: str, man: pd.DataFrame, p: dict, reps: pd.DataFrame) -> Path | None:
    row = man[man["clip_id"] == clip_id]
    if row.empty:
        print(f"Manifest에 없음: {clip_id}")
        return None
    video = ROOT / row.iloc[0]["video_file"]
    kp_path = p["keypoints"] / f"{clip_id}_keypoints.csv"
    if not video.exists() or not kp_path.exists():
        print(f"영상 또는 keypoints 없음: {clip_id}")
        return None
    fid, _, xy, conf = load_keypoints_csv(kp_path)
    by_frame = {int(f): i for i, f in enumerate(fid)}
    r = reps[reps["clip_id"] == clip_id].sort_values("start_frame")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = p["reports"] / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{clip_id}_overlay.mp4"
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    s = max(1.0, min(w, h) / 500)  # 글자·선 굵기 배율
    bar_h = int(14 * s)

    frame_no = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frame_no += 1
        i = by_frame.get(frame_no)
        if i is not None:
            for a, b in EDGES:
                pa, pb = xy[i, a], xy[i, b]
                if np.isfinite(pa).all() and np.isfinite(pb).all() and min(conf[i, a], conf[i, b]) >= 0.30:
                    cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (255, 255, 255), int(2 * s))
            for k in range(17):
                if np.isfinite(xy[i, k]).all():
                    cv2.circle(img, tuple(int(v) for v in xy[i, k]), int(4 * s), conf_color(conf[i, k]), -1)

        # 위쪽 1회 구간 막대
        cv2.rectangle(img, (0, 0), (w, bar_h), (30, 30, 30), -1)
        for _, rr in r.iterrows():
            x0, x1 = int((rr.start_frame - 1) / n * w), int(rr.end_frame / n * w)
            cv2.rectangle(img, (x0, 0), (max(x1, x0 + 2), bar_h), COLOR.get(rr.verdict, (200, 200, 200)), -1)
        cx = int(frame_no / max(n, 1) * w)
        cv2.line(img, (cx, 0), (cx, bar_h), (255, 255, 255), max(1, int(s)))

        cur = r[(r.start_frame <= frame_no) & (r.end_frame >= frame_no)]
        text = [f"{clip_id}  f{frame_no}"]
        if len(cur):
            c0 = cur.iloc[0]
            text.append(f"rep {int(c0.rep)}: {c0.verdict}")
            if isinstance(c0.get("out_of_range"), str) and c0.out_of_range:
                text += c0.out_of_range.split("; ")
            for m in ("depth", "knee_min", "hip_min", "knee_spread_bottom"):
                if m in c0 and pd.notna(c0[m]):
                    text.append(f"{m}={c0[m]:.2f}")
        for k, t in enumerate(text):
            y = bar_h + int((22 + 22 * k) * s)
            col = COLOR.get(cur.iloc[0].verdict, (255, 255, 255)) if (len(cur) and k == 1) else (255, 255, 255)
            cv2.putText(img, t, (int(8 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), int(4 * s))
            cv2.putText(img, t, (int(8 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, col, max(1, int(1.5 * s)))
        out.write(img)
    cap.release()
    out.release()
    print(f"{clip_id} → {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_team_config()
    p = team_paths(cfg)
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    rj = p["judgments"] / "rep_judgments.csv"
    reps = pd.read_csv(rj) if rj.exists() else pd.DataFrame(columns=["clip_id", "start_frame"])
    clips = sorted(reps["clip_id"].unique()) if args.all else (args.clip or [])
    if not clips:
        print("--clip <clip_id> 또는 --all 을 지정하세요")
        return
    for c in clips:
        render(c, man, p, reps)


if __name__ == "__main__":
    main()
