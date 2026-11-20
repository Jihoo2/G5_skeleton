"""
[팀 23] 1회 감지 신호 그래프 — 회차를 놓치거나 잘못 셀 때 원인 확인용 (교과 10일차 실패 분석 자리)

한 영상의 leg_ratio(골반–발목 세로 거리 / 몸통 길이)가 시간에 따라 어떻게 움직였는지,
녹화 판정(team_09)과 실시간(team_21 replay)이 각각 어디를 1회로 잡았는지 한 장에 그립니다.

  위 그래프  : leg_ratio(평활화) + '서 있을 때' 기준선 + 1회 시작선(×0.85) + 1회 인정선(×0.80)
               녹화 판정 회차(초록 OK / 빨강 NG / 회색 CHECK), 실시간에만 있는 회차(보라 점선 테두리)
  아래 그래프: 몸통 길이(px)를 영상 중앙값으로 나눈 값
               → 1에서 크게 벗어나면 사람이 카메라에 가까워지거나 멀어진 것
                 (leg_ratio 는 영상 전체의 몸통 길이 하나로 나누므로, 크기가 변하면 깊이가 덜/더 나와 보임)

실행
  python scripts/team_23_plot_rep_signal.py EXT_S08_A004_T01_diag EXT_S04_A008_T01_diag
  python scripts/team_23_plot_rep_signal.py --action A004        # 그 동작의 val·val_ext 영상 전부
출력
  reports/.../figures/rep_signal_<clip_id>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_utils import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER  # noqa: E402
from src.team_feature_utils import (  # noqa: E402
    find_reps, find_shallow_attempts, load_keypoints_csv, load_team_config, rep_min_len, resolved_series, seq_fps,
    smooth, team_paths,
)

ENTER, NEED = 0.85, 0.80  # find_reps 기본값과 같음
COLOR = {"OK": "#2e9e44", "NG": "#d33", "CHECK": "#999"}


def set_font(plt) -> bool:
    from matplotlib import font_manager
    for fp in ["/mnt/c/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def torso_scale(p: dict, clip_id: str, n: int) -> np.ndarray:
    """Frame별 몸통 길이(px) / 영상 중앙값 — 원본 Keypoint 기준"""
    kp_path = p["keypoints"] / f"{clip_id}_keypoints.csv"
    if not kp_path.exists():
        return np.full(n, np.nan)
    _, _, xy, _ = load_keypoints_csv(kp_path)
    sh = np.nanmean(xy[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], axis=1)
    hp = np.nanmean(xy[:, [LEFT_HIP, RIGHT_HIP]], axis=1)
    d = np.linalg.norm(sh - hp, axis=1)
    d = smooth(d)
    out = np.full(n, np.nan)
    m = min(n, len(d))
    out[:m] = d[:m] / np.nanmedian(d) if np.isfinite(d).any() else np.nan
    return out


def plot_clip(clip_id: str, cfg: dict, p: dict, man: pd.DataFrame, offline: pd.DataFrame | None,
              online: pd.DataFrame | None, plt, korean: bool) -> Path | None:
    row = man[man["clip_id"] == clip_id]
    sp = p["sequences"] / f"{clip_id}_sequence.csv"
    if row.empty or not sp.exists():
        print(f"[건너뜀] {clip_id}: manifest 또는 sequence 없음")
        return None
    r = row.iloc[0]
    action = cfg["actions"][r["action_id"]]
    seq = pd.read_csv(sp)
    fps = seq_fps(seq, cfg["fps"])
    ser = resolved_series(seq, r["view"], cfg["keypoint_confidence"], action.get("leg_combine", "mean"))
    lr = smooth(ser["leg_ratio"])
    stand = np.nanpercentile(lr, 90)
    min_rep = rep_min_len(action)
    reps, _ = find_reps(ser["leg_ratio"], fps, min_len_s=min_rep)
    shallow = find_shallow_attempts(ser["leg_ratio"], fps)
    x = np.arange(1, len(lr) + 1)  # frame 번호 (rep start_frame = s+1 과 같은 기준)
    scale = torso_scale(p, clip_id, len(lr))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, lr, color="#222", lw=1.6, label="leg_ratio (평활화)" if korean else "leg_ratio (smoothed)")
    for k, ls, txt in [(1.0, "-", "서 있을 때(90%)"), (ENTER, "--", "1회 시작 ×0.85"), (NEED, ":", "1회 인정 ×0.80")]:
        ax.axhline(stand * k, color="#555", ls=ls, lw=1, label=(txt if korean else f"stand x{k}"))
    ymax = np.nanmax(lr) * 1.05 if np.isfinite(lr).any() else 1
    # 지금 설정으로 다시 찾은 구간 (얇은 띠, 위쪽)
    for s, e in reps:
        ax.axvspan(s + 1, e, ymin=0.93, ymax=1.0, color="#3a7bd5", alpha=0.6)
    for s, e in shallow:
        ax.axvspan(s + 1, e, ymin=0.93, ymax=1.0, color="#f0a020", alpha=0.6)
    # 녹화 판정 결과 (rep_judgments.csv)
    if offline is not None:
        for _, o in offline[offline["clip_id"] == clip_id].iterrows():
            c = COLOR.get(o["verdict"], "#999")
            ax.axvspan(o["start_frame"], o["end_frame"], color=c, alpha=0.18)
            ax.text((o["start_frame"] + o["end_frame"]) / 2, ymax * 0.97, str(o["rep"]), ha="center",
                    va="top", fontsize=9, color=c)
    # 실시간에만 있는 회차 (realtime_replay_detail.csv)
    if online is not None:
        for _, o in online[(online["clip_id"] == clip_id) & (online["kind"] == "online_only")].iterrows():
            ax.axvspan(o["start_online"], o["end_online"], fill=False, ec="#8e44ad", ls="--", lw=1.5)
            ax.text((o["start_online"] + o["end_online"]) / 2, ymax * 0.88, f"실시간 {o['rep_online']}"
                    if korean else f"RT {o['rep_online']}", ha="center", va="top", fontsize=9, color="#8e44ad")
    ax.set_ylabel("leg_ratio")
    title = (f"{clip_id}  ({action.get('label', '')}, {r['view']}, {fps}fps, 최소 1회 {min_rep}초)  "
             f"— 파랑 띠: 1회, 주황 띠: 얕은 시도, 배경: 녹화 판정") if korean else f"{clip_id}"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower left", fontsize=8, ncol=4)
    ax2.plot(x, scale, color="#3a7bd5", lw=1.4)
    ax2.axhline(1.0, color="#555", lw=0.8)
    ax2.set_ylabel("몸통 크기 비" if korean else "torso / median")
    ax2.set_xlabel("Frame")
    fig.tight_layout()
    out_dir = p["reports"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rep_signal_{clip_id}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    rng = np.nanmax(scale) / np.nanmin(scale) if np.isfinite(scale).any() else np.nan
    print(f"저장: {out}  (1회 {len(reps)}개, 얕은 시도 {len(shallow)}개, 몸통 크기 변화 최대/최소 {rng:.2f}배)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--action", help="그 동작의 val·val_ext 영상 전부 (예: A004)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    korean = set_font(plt)

    cfg = load_team_config()
    p = team_paths(cfg)
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    clips = list(args.clips)
    if args.action:
        clips += man[(man["action_id"] == args.action) & man["split"].isin(["val", "val_ext"])]["clip_id"].tolist()
    if not clips:
        raise SystemExit("clip_id 를 적거나 --action A004 처럼 동작을 지정하세요")
    rj = p["judgments"] / "rep_judgments.csv"
    offline = pd.read_csv(rj) if rj.exists() else None
    if offline is not None:
        offline = offline[pd.to_numeric(offline["start_frame"], errors="coerce").notna()]
    dp = p["tables"] / "realtime_replay_detail.csv"
    online = pd.read_csv(dp) if dp.exists() else None
    for c in dict.fromkeys(clips):
        plot_clip(c, cfg, p, man, offline, online, plt, korean)


if __name__ == "__main__":
    main()
