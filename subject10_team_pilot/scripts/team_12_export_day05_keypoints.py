"""
[팀 12] 교과 5일차 26번 형식으로 Keypoint Dataset 내보내기 (YOLO 다시 실행 안 함)

입력
  reports/team_projects/<team>/keypoints/<clip_id>_keypoints.csv   (team_03, 1행 = 관절 1개)
  reports/team_projects/<team>/tables/video_inventory.csv          (team_02, 해상도·Frame 수)
  configs/<team>_clip_manifest.csv
출력 (교재 26번과 같은 열 구성)
  reports/.../day05/keypoints/<subject>/<action_label>/<clip_id>.csv
      1행 = 1 Frame, 기본 10열 + 관절 17 × (x_px, y_px, x_norm, y_norm, conf) 85열 = 95열
      Pose를 못 찾은 Frame도 pose_detected=0, 관절값 빈칸으로 보존 (frame_id 0부터)
  reports/.../day05/manifests/keypoint_dataset_manifest.csv   1행 = 1 Clip
      교재 열 + 팀 열(view, view_deg, source_type, form_label, split, review_status)

교재와 다른 점
  - selected_box_conf: team_03이 BBox Confidence를 저장하지 않아 빈칸
    (필요하면 team_03 --overwrite 로 다시 추출할 때 함께 저장하도록 바꿀 수 있음)
  - EXCLUDE Clip은 team_03이 추출하지 않았으므로 Manifest에서 빠짐 (건너뛴 목록 출력)

실행
  python scripts/team_12_export_day05_keypoints.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import day05_paths, parse_team_clip  # noqa: E402
from src.pose_utils import COCO_KEYPOINT_NAMES  # noqa: E402
from src.team_feature_utils import load_team_config, team_paths  # noqa: E402

BASE_COLUMNS = ["clip_id", "frame_id", "time_sec", "subject", "action_id", "action_label",
                "take", "person_count", "selected_box_conf", "pose_detected"]
KP_COLUMNS = [f"{n}_{k}" for n in COCO_KEYPOINT_NAMES
              for k in ("x_px", "y_px", "x_norm", "y_norm", "conf")]


def export_clip(row: pd.Series, long_df: pd.DataFrame, info: dict, out_path: Path) -> dict:
    meta = parse_team_clip(row["clip_id"])
    width, height, fps = int(info["width"]), int(info["height"]), float(info["fps"]) or 30.0
    n_frames = int(info.get("read_frames") or 0)
    if n_frames <= 0:
        n_frames = int(long_df["frame_id"].max()) if len(long_df) else 0

    # team_03: frame_id 1부터, Pose가 있는 Frame만 기록 → 교재: 0부터, 모든 Frame 기록
    by_frame = {int(f): g for f, g in long_df.groupby("frame_id")}
    rows, detected, single, frame_conf = [], 0, 0, []
    for f in range(1, n_frames + 1):
        r = dict(clip_id=row["clip_id"], frame_id=f - 1, time_sec=round((f - 1) / fps, 4),
                 subject=meta["subject"], action_id=meta["action_id"],
                 action_label=row["action_label"], take=meta["take"],
                 person_count=0, selected_box_conf="", pose_detected=0,
                 **{c: "" for c in KP_COLUMNS})
        g = by_frame.get(f)
        if g is not None and len(g) == len(COCO_KEYPOINT_NAMES):
            people = int(g["people"].iloc[0])
            r.update(person_count=people, pose_detected=1)
            detected += 1
            single += int(people == 1)
            for _, k in g.iterrows():
                n = k["keypoint_name"]
                x, y, c = float(k["pixel_x"]), float(k["pixel_y"]), float(k["confidence"])
                r[f"{n}_x_px"], r[f"{n}_y_px"] = round(x, 3), round(y, 3)
                r[f"{n}_x_norm"], r[f"{n}_y_norm"] = round(x / width, 6), round(y / height, 6)
                r[f"{n}_conf"] = round(c, 4)
            frame_conf.append(float(g["confidence"].mean()))
        rows.append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=BASE_COLUMNS + KP_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    total = max(len(rows), 1)
    return dict(
        clip_id=row["clip_id"], video_path=row["video_file"],
        keypoint_csv=str(out_path.relative_to(ROOT)),
        subject=meta["subject"], action_id=meta["action_id"], action_label=row["action_label"],
        take=meta["take"], width=width, height=height, fps=round(fps, 3),
        metadata_frames=int(info.get("frame_count") or n_frames), read_frames=len(rows),
        pose_detected_frames=detected, pose_ratio=round(detected / total, 4),
        single_person_ratio=round(single / total, 4),
        mean_keypoint_conf=round(float(np.mean(frame_conf)), 4) if frame_conf else 0.0,
        view=row["view"], view_deg=row["view_deg"], source_type=row["source_type"],
        form_label=row["form_label"], split=row["split"], review_status=row["review_status"],
    )


def main() -> None:
    cfg = load_team_config()
    p = team_paths(cfg)
    d = day05_paths(cfg)
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    inv_path = p["tables"] / "video_inventory.csv"
    if not inv_path.exists():
        raise SystemExit(f"[중지] {inv_path} 없음 → python scripts/team_02_check_videos.py 먼저 실행")
    inv = pd.read_csv(inv_path).set_index("clip_id").to_dict("index")

    out_rows, skipped = [], []
    for _, r in man.iterrows():
        kp = p["keypoints"] / f"{r['clip_id']}_keypoints.csv"
        if not kp.exists() or r["clip_id"] not in inv:
            skipped.append((r["clip_id"], r["review_status"],
                            "keypoints 없음" if not kp.exists() else "video_inventory에 없음"))
            continue
        meta = parse_team_clip(r["clip_id"])
        out = d["keypoints"] / meta["subject"] / r["action_label"] / f"{r['clip_id']}.csv"
        item = export_clip(r, pd.read_csv(kp), inv[r["clip_id"]], out)
        out_rows.append(item)
        print(f"{r['clip_id']:26s} frames={item['read_frames']:4d} pose={item['pose_ratio']:.3f} "
              f"single={item['single_person_ratio']:.3f} conf={item['mean_keypoint_conf']:.3f}")

    pd.DataFrame(out_rows).to_csv(d["keypoint_manifest"], index=False, encoding="utf-8-sig")
    if skipped:
        print("\n[건너뜀]")
        for c, s, why in skipped:
            print(f"  {c} ({s or '상태 없음'}): {why}")
    print(f"\nKeypoint CSV: {len(out_rows)}개 (95열) → {d['keypoints']}")
    print(f"Manifest   : {d['keypoint_manifest']}")
    print("Keypoint export: COMPLETE")


if __name__ == "__main__":
    main()
