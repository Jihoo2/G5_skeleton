"""
사전촬영 영상(웹캠) vs VICON 모캡(GLB) 비교

입력
  --keypoints : 10_pose_video_quality.py가 만든 *_keypoints.csv (영상 전체: --max-frames 0)
  --glb       : 같은 동작의 VICON GLB
  --view      : front | side | up   (up = 높은 위치의 정면 카메라 → front로 투영)

출력 (reports/pilot/)
  {이름}_compare.png   : 관절 Confidence + 관절각도(모캡 vs 웹캠) 그래프
  pilot_summary.csv    : 동작·방향·관절별 Confidence, 최소/최대 각도, DTW 차이 (실행할 때마다 누적)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glb_utils import evaluate  # noqa: E402

# ① COCO 17 번호 ↔ VICON Bone (어깨는 LeftShoulder(쇄골)가 아니라 LeftArm)
COCO_TO_VICON = {
    5: "LeftArm", 6: "RightArm",
    7: "LeftForeArm", 8: "RightForeArm",
    9: "LeftHand", 10: "RightHand",
    11: "LeftUpLeg", 12: "RightUpLeg",
    13: "LeftLeg", 14: "RightLeg",
    15: "LeftFoot", 16: "RightFoot",
}

# ② 비교할 관절각도 (a-b-c에서 b가 꼭짓점)
ANGLES = {
    "L_elbow": (5, 7, 9), "R_elbow": (6, 8, 10),
    "L_knee": (11, 13, 15), "R_knee": (12, 14, 16),
    "L_hip": (5, 11, 13), "R_hip": (6, 12, 14),
}

# ③ 모캡 3D → 2D 투영 축 (이번 VICON 촬영본 기준: x축=옆모습, z축=앞모습, y=위)
VIEW_AXES = {"side": [0, 1], "front": [2, 1], "up": [2, 1]}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints", required=True)
    p.add_argument("--glb", required=True)
    p.add_argument("--view", required=True, choices=VIEW_AXES)
    p.add_argument("--kp-conf", type=float, default=0.50)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def smooth(x, k=5):
    """NaN을 건너뛰는 이동평균 (프레임 간 떨림 완화)"""
    s = pd.Series(x)
    return s.rolling(k, center=True, min_periods=1).mean().to_numpy()


def dtw_mean_diff(a, b):
    """DTW로 속도 차이를 맞춘 뒤 평균 각도 차이(도)"""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        cost = np.abs(a[i - 1] - b)
        for j in range(1, m + 1):
            D[i, j] = cost[j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # 경로 길이로 나누어 '평균 몇 도 차이'로 해석
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else ((i - 1, j) if k == 1 else (i, j - 1))
    return D[n, m] / max(steps, 1)


def load_webcam(path, kp_conf):
    df = pd.read_csv(path)
    n = int(df["frame_id"].max())
    xy = np.full((n, 17, 2), np.nan)
    conf = np.zeros((n, 17))
    f = df["frame_id"].to_numpy() - 1
    k = df["keypoint_id"].to_numpy()
    xy[f, k] = df[["pixel_x", "pixel_y"]].to_numpy()
    conf[f, k] = df["confidence"].to_numpy()
    masked = xy.copy()
    masked[conf < kp_conf] = np.nan  # Confidence 낮은 관절은 각도 계산에서 제외
    return masked, conf


def load_mocap(glb, view):
    names, P, _ = evaluate(glb)
    out = np.full((len(P), 17, 2), np.nan)
    for cid, bone in COCO_TO_VICON.items():
        out[:, cid] = P[:, names.index(bone)][:, VIEW_AXES[view]]
    return out


def main():
    a = parse_args()
    name = f"{Path(a.keypoints).stem.replace('_keypoints', '')}_vs_{Path(a.glb).stem}"
    out_dir = ROOT / "reports" / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    web, conf = load_webcam(a.keypoints, a.kp_conf)
    moc = load_mocap(a.glb, a.view)
    tw = np.arange(len(web)) / a.fps
    tm = np.arange(len(moc)) / a.fps

    fig, axs = plt.subplots(4, 2, figsize=(14, 13))
    # ④ 관절 Confidence (웹캠)
    ax = axs[0, 0]
    for cid, lab in [(11, "L_hip"), (13, "L_knee"), (15, "L_ankle"), (7, "L_elbow"), (9, "L_wrist")]:
        ax.plot(tw, conf[:, cid], label=lab, lw=1)
    ax.axhline(a.kp_conf, color="r", ls="--", lw=1)
    ax.set_title("Webcam keypoint confidence")
    ax.legend(fontsize=7)
    axs[0, 1].axis("off")

    rows = []
    for ax, (an, (i, j, k)) in zip(axs[1:].ravel(), ANGLES.items()):
        w = smooth(angle(web[:, i], web[:, j], web[:, k]))
        m = smooth(angle(moc[:, i], moc[:, j], moc[:, k]))  # 웹캠과 같은 스무딩
        # DTW는 10fps로 줄여 계산 (속도·길이 차이 허용)
        d = dtw_mean_diff(m[::3], w[::3])
        valid = np.mean(np.min(conf[:, [i, j, k]], axis=1) >= a.kp_conf)
        ax.plot(tm / tm[-1], m, "k", lw=1.5, label="mocap")
        ax.plot(tw / tw[-1], w, "tab:orange", lw=1.2, label="webcam")
        ax.set_title(f"{an} | DTW diff {d:.1f} deg | valid {valid*100:.0f}%")
        ax.set_ylim(0, 185)
        ax.legend(fontsize=7)
        rows.append(dict(
            keypoints=Path(a.keypoints).name, glb=Path(a.glb).name, view=a.view, angle=an,
            valid_ratio=round(valid, 3),
            mean_conf=round(float(np.mean(np.min(conf[:, [i, j, k]], axis=1))), 3),
            mocap_min=round(np.nanmin(m), 1), mocap_max=round(np.nanmax(m), 1),
            webcam_min=round(np.nanmin(w), 1) if np.isfinite(w).any() else "",
            webcam_max=round(np.nanmax(w), 1) if np.isfinite(w).any() else "",
            dtw_mean_diff=round(d, 1) if np.isfinite(d) else "",
        ))
    for ax in axs[-1]:
        ax.set_xlabel("normalized time (0=start, 1=end)")
    fig.suptitle(name)
    plt.tight_layout()
    png = out_dir / f"{name}_compare.png"
    plt.savefig(png, dpi=90)

    summary = out_dir / "pilot_summary.csv"
    new = not summary.exists()
    with summary.open("a", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new:
            wr.writeheader()
        wr.writerows(rows)

    print(pd.DataFrame(rows)[["angle", "valid_ratio", "mocap_min", "webcam_min", "dtw_mean_diff"]].to_string(index=False))
    print(f"\nSaved: {png}\nSummary: {summary}")


if __name__ == "__main__":
    main()
