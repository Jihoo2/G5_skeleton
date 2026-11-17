"""
1회(rep) 단위로 웹캠 vs VICON 모캡 비교

반복 구간 찾기 신호 (카메라 방향과 무관)
  leg_ratio = (발목 높이 - 골반 높이) / 몸통 길이
    서 있음 → 약 1.5 이상, 스쿼트 바닥 → 약 0.5, 플랭크 → 0 근처
  leg_ratio가 '서 있을 때 값 × 0.6' 아래로 내려간 구간 = 1회

1회마다 계산
  depth      : leg_ratio 최솟값 (얼마나 깊이 내려갔는가)
  knee_min   : 무릎 최소 각도
  hip_min    : 고관절 최소 각도
  elbow_min  : 팔꿈치 최소 각도
  duration   : 1회 걸린 시간(초)

측면(side) 영상은 카메라에 가까운 쪽 관절만 사용 (반대쪽은 가려짐)

실행 예
  python scripts/12_rep_compare_mocap.py \
    --keypoints reports/pilot/tables/Front_A008_keypoints.csv \
    --glb data/pilot/mocap/261110_B008_T002_SHG.glb --view front
"""
from __future__ import annotations

import argparse
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

COCO_TO_VICON = {
    5: "LeftArm", 6: "RightArm", 7: "LeftForeArm", 8: "RightForeArm",
    9: "LeftHand", 10: "RightHand", 11: "LeftUpLeg", 12: "RightUpLeg",
    13: "LeftLeg", 14: "RightLeg", 15: "LeftFoot", 16: "RightFoot",
}
SIDE_IDS = {  # 어깨, 팔꿈치, 손목, 골반, 무릎, 발목
    "L": (5, 7, 9, 11, 13, 15),
    "R": (6, 8, 10, 12, 14, 16),
}
VIEW_AXES = {"side": [0, 1], "front": [2, 1], "up": [2, 1]}  # 모캡 x=옆모습, z=앞모습, y=위


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keypoints", required=True)
    p.add_argument("--glb", required=True)
    p.add_argument("--view", required=True, choices=VIEW_AXES)
    p.add_argument("--kp-conf", type=float, default=0.50)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--dip", type=float, default=0.6, help="서 있을 때 대비 몇 배 아래를 1회로 볼지")
    return p.parse_args()


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def smooth(x, k=7):
    return pd.Series(x).rolling(k, center=True, min_periods=1).median().rolling(k, center=True, min_periods=1).mean().to_numpy()


# ---------- 입력 ----------
def load_webcam(path, kp_conf):
    """반환: xy (프레임, 17, 2) — 위쪽이 + 가 되도록 y 반전 / conf (프레임, 17)"""
    df = pd.read_csv(path)
    n = int(df["frame_id"].max())
    xy = np.full((n, 17, 2), np.nan)
    conf = np.zeros((n, 17))
    f, k = df["frame_id"].to_numpy() - 1, df["keypoint_id"].to_numpy()
    xy[f, k, 0] = df["pixel_x"].to_numpy()
    xy[f, k, 1] = -df["pixel_y"].to_numpy()  # 이미지 y는 아래가 + → 위가 + 로 맞춤
    conf[f, k] = df["confidence"].to_numpy()
    xy[conf < kp_conf] = np.nan
    return xy, conf


def load_mocap(glb, view):
    names, P, _ = evaluate(glb)
    xy = np.full((len(P), 17, 2), np.nan)
    for cid, bone in COCO_TO_VICON.items():
        xy[:, cid] = P[:, names.index(bone)][:, VIEW_AXES[view]]
    return xy


# ---------- 특징 ----------
def features(xy, sides):
    """프레임별 leg_ratio, 무릎·고관절·팔꿈치 각도 (sides에 있는 쪽 평균)"""
    sh = np.nanmean(xy[:, [5, 6]], axis=1)
    hp = np.nanmean(xy[:, [11, 12]], axis=1)
    # 몸통 길이는 '서 있을 때 값' 하나로 고정
    # (정면 카메라에서 플랭크를 하면 몸통이 카메라를 향해 길이가 0에 가까워지므로 프레임별로 나누면 값이 폭주함)
    torso = np.nanpercentile(np.linalg.norm(sh - hp, axis=1), 90)
    out = {"leg_ratio": [], "knee": [], "hip": [], "elbow": []}
    for s in sides:
        S, E, W, H, K, A = SIDE_IDS[s]
        out["leg_ratio"].append((xy[:, H, 1] - xy[:, A, 1]) / torso)
        out["knee"].append(angle(xy[:, H], xy[:, K], xy[:, A]))
        out["hip"].append(angle(xy[:, S], xy[:, H], xy[:, K]))
        out["elbow"].append(angle(xy[:, S], xy[:, E], xy[:, W]))
    return {k: smooth(np.nanmean(np.vstack(v), axis=0)) for k, v in out.items()}


def find_reps(ratio, fps, dip):
    stand = np.nanpercentile(ratio, 90)
    low = ratio < stand * dip
    reps, i, n = [], 0, len(ratio)
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            reps.append([i, j])
            i = j
        else:
            i += 1
    merged = []  # 0.4초 이내로 끊긴 구간은 하나로 합침
    for r in reps:
        if merged and r[0] - merged[-1][1] < 0.4 * fps:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= 0.3 * fps], stand


def rep_table(feat, reps, fps, source):
    rows = []
    for n, (s, e) in enumerate(reps, 1):
        rows.append(dict(
            source=source, rep=n, start_s=round(s / fps, 2), duration_s=round((e - s) / fps, 2),
            depth=round(np.nanmin(feat["leg_ratio"][s:e]), 2),
            knee_min=round(np.nanmin(feat["knee"][s:e]), 1),
            hip_min=round(np.nanmin(feat["hip"][s:e]), 1),
            elbow_min=round(np.nanmin(feat["elbow"][s:e]), 1),
        ))
    return pd.DataFrame(rows)


def main():
    a = parse_args()
    name = Path(a.keypoints).stem.replace("_keypoints", "")
    out_dir = ROOT / "reports" / "pilot" / "reps"
    out_dir.mkdir(parents=True, exist_ok=True)

    web, conf = load_webcam(a.keypoints, a.kp_conf)
    if a.view == "side":  # 카메라에 가까운 쪽 = 팔·다리 Confidence 평균이 높은 쪽
        near = "L" if conf[:, list(SIDE_IDS["L"])].mean() >= conf[:, list(SIDE_IDS["R"])].mean() else "R"
        sides = (near,)
    else:
        sides = ("L", "R")
    moc = load_mocap(a.glb, a.view)

    fw, fm = features(web, sides), features(moc, ("L", "R") if a.view != "side" else sides)
    rw, stand_w = find_reps(fw["leg_ratio"], a.fps, a.dip)
    rm, stand_m = find_reps(fm["leg_ratio"], a.fps, a.dip)
    tw, tm = rep_table(fw, rw, a.fps, "webcam"), rep_table(fm, rm, a.fps, "mocap")
    table = pd.concat([tm, tw], ignore_index=True)
    table.insert(0, "video", name)
    table.insert(1, "view", a.view)
    table.insert(2, "sides", "".join(sides))
    table.to_csv(out_dir / f"{name}_reps.csv", index=False, encoding="utf-8-sig")

    # 그래프: leg_ratio와 찾은 반복 구간
    fig, axs = plt.subplots(2, 1, figsize=(13, 6))
    for ax, f, reps, stand, lab in [(axs[0], fm, rm, stand_m, "mocap"), (axs[1], fw, rw, stand_w, f"webcam ({name})")]:
        t = np.arange(len(f["leg_ratio"])) / a.fps
        ax.plot(t, f["leg_ratio"], "k", lw=1.2, label="leg_ratio")
        ax.axhline(stand * a.dip, color="r", ls="--", lw=1, label="rep threshold")
        for s, e in reps:
            ax.axvspan(s / a.fps, e / a.fps, color="tab:orange", alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(t, f["knee"], color="tab:blue", lw=0.8, alpha=0.7)
        ax2.set_ylabel("knee angle (deg)", color="tab:blue")
        ax.set_title(f"{lab}: {len(reps)} reps")
        ax.legend(fontsize=7, loc="lower left")
    axs[-1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_reps.png", dpi=85)
    plt.close(fig)

    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main()
