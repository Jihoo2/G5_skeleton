"""
[팀 11] NG 검출 민감도 실험 — 정석 영상의 관절 좌표를 일부러 변형해서 틀린 자세를 잡는지 확인

왜 필요한가
  팀 방침상 일부러 틀린 자세는 촬영하지 않는다(범위가 너무 넓음).
  대신 정석(OK) 영상의 Keypoint를 정해진 방향으로 조금씩 변형해서
  "얼마나 틀려야 NG로 잡히는가"를 강도별로 측정한다.

변형 (pixel 좌표에서 적용 → 실제 판정과 같은 경로로 계산)
  shallow  덜 앉기     : 발목 중점 기준 자세를 '서 있는 자세' 쪽으로 a 비율만큼 당김 (움직임 폭 축소, 점프 이동은 유지)
  knee_in  무릎 모으기 : 두 무릎을 발목 중점 쪽으로 b 비율만큼 이동
  hip_sag  허리 처짐   : 플랭크 Frame에서 골반을 몸통 길이 × g 만큼 아래로
  hip_pike 엉덩이 들림 : 플랭크 Frame에서 골반을 몸통 길이 × g 만큼 위로

판정 경로 (team_09와 동일)
  keypoints CSV → 변형 → build_frame_features → rep_metrics → curve_distance → team_09.judge_rep
  기준: reports/.../standards/<action>_<view>.json  (team_08 결과)

출력
  tables/ng_sensitivity.csv          Clip × 변형 × 강도별 결과
  tables/ng_sensitivity_summary.csv  변형 × 강도별 검출률
  tables/ng_sensitivity.png          검출률 그래프

실행
  python scripts/team_11_ng_sensitivity.py
  python scripts/team_11_ng_sensitivity.py --splits val val_ext      # 사용할 split 지정 (test 계열은 사용 불가)
  python scripts/team_11_ng_sensitivity.py --only shallow knee_in    # 일부 변형만
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feature_utils import (  # noqa: E402
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
)
from src.team_feature_utils import (  # noqa: E402
    build_frame_features, curve_distance, load_keypoints_csv, load_team_config,
    rep_metrics, team_paths, torso_reference,
)

# 판정 함수는 team_09에서 그대로 가져온다 (실험과 실제 판정이 어긋나지 않도록)
_spec = importlib.util.spec_from_file_location("team09", ROOT / "scripts" / "team_09_judge.py")
team09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team09)

ANK, HIP, KNEE, SH = [LEFT_ANKLE, RIGHT_ANKLE], [LEFT_HIP, RIGHT_HIP], [LEFT_KNEE, RIGHT_KNEE], [LEFT_SHOULDER, RIGHT_SHOULDER]


def _mid(xy, ids):
    return np.nanmean(xy[:, ids], axis=1)  # (n, 2)


# ------------------------------------------------------------------
# 변형 함수 (xy: (n, 17, 2) pixel, 이미지 좌표계 y 아래로 +)
# ------------------------------------------------------------------
def shallow(xy, a):
    """덜 앉기: 발목 중점 기준 상대 자세를 서 있는 자세 쪽으로 a 비율만큼 당김"""
    if a == 0:
        return xy.copy()
    anchor = _mid(xy, ANK)[:, None, :]
    rel = xy - anchor
    height = anchor[:, 0, 1] - _mid(xy, HIP)[:, 1]  # 골반이 발목보다 위에 있는 정도
    stand = height >= np.nanpercentile(height, 90)
    stand_rel = np.nanmedian(rel[stand], axis=0)  # (17, 2) 서 있는 자세
    return anchor + stand_rel + (1 - a) * (rel - stand_rel)


def knee_in(xy, b):
    """무릎 모으기: 두 무릎 x를 발목 중점 쪽으로 b 비율만큼 이동 (knee_spread가 (1-b)배가 됨)"""
    out = xy.copy()
    cx = _mid(xy, ANK)[:, 0]
    for k in KNEE:
        out[:, k, 0] = cx + (1 - b) * (xy[:, k, 0] - cx)
    return out


def hip_shift(xy, g):
    """플랭크 Frame에서 골반을 몸통 길이 × g 만큼 이동 (g > 0 아래 = 허리 처짐, g < 0 위 = 엉덩이 들림)"""
    out = xy.copy()
    if g == 0:
        return out
    torso = torso_reference(xy)
    d = _mid(xy, SH) - _mid(xy, HIP)
    leg = _mid(xy, ANK) - _mid(xy, HIP)
    with np.errstate(invalid="ignore"):
        lying = ((np.abs(d[:, 1]) < np.abs(d[:, 0]) * np.tan(np.radians(35)))
                 & (np.abs(leg[:, 1]) < np.abs(leg[:, 0]) * np.tan(np.radians(25))))
    for h in HIP:
        out[lying, h, 1] = xy[lying, h, 1] + g * torso
    return out


EXPERIMENTS = {
    "shallow": dict(action="A008", fn=shallow, target="depth",
                    strengths=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], plot="shallow squat (A008)"),
    "knee_in": dict(action="A008", fn=knee_in, target="knee_spread_bottom",
                    strengths=[0, 0.1, 0.2, 0.3, 0.4, 0.5], plot="knees in (A008 front)"),
    "hip_sag": dict(action="A001", fn=hip_shift, target="body_line_min",
                    strengths=[0, 0.05, 0.1, 0.15, 0.2, 0.3], plot="hip sag (A001)"),
    "hip_pike": dict(action="A001", fn=lambda xy, s: hip_shift(xy, -s), target="body_line_max",
                     strengths=[0, 0.1, 0.2, 0.3, 0.4, 0.6], plot="hip pike (A001)"),
}


def judge_seq(seq, r, cfg, metrics, tpl, kp_conf, check_ratio):
    """team_09 main()의 1회 단위 판정과 같은 순서"""
    combine = cfg["actions"][r["action_id"]].get("leg_combine", "mean")
    reps, curves = rep_metrics(seq, r["view"], cfg["fps"], kp_conf, combine)
    out = []
    for i, row in reps.iterrows():
        d = row.to_dict()
        d["curve_dist"] = curve_distance(curves[i], tpl) if tpl else np.nan
        if d["low_conf_ratio"] > check_ratio:
            out.append(("CHECK", []))
        else:
            v, bad, _, _ = team09.judge_rep(d, metrics)
            out.append((v, bad))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "val_ext"],
                    help="정석 규칙 판정이라 train 영상도 공정하게 쓸 수 있음 (form_label=OK만 사용)")
    ap.add_argument("--only", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    args = ap.parse_args()
    if set(args.splits) & {"test", "test_ext"}:
        raise SystemExit("[중지] test / test_ext 는 규칙 조정 실험에 쓰지 않습니다 (Test 누수).")

    cfg = load_team_config()
    p = team_paths(cfg)
    kp_conf = cfg["keypoint_confidence"]
    check_ratio = cfg["standard"]["low_conf_check_ratio"]
    glb_only = cfg["standard"].get("judge_glb_only", False)

    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    man = man[man["split"].isin(args.splits) & (man["form_label"] == "OK")
              & (~man["review_status"].isin(["RESHOOT", "EXCLUDE"]))]

    rows = []
    for name in args.only:
        ex = EXPERIMENTS[name]
        for _, r in man[man["action_id"] == ex["action"]].iterrows():
            std_path = p["standards"] / f"{r['action_id']}_{r['view']}.json"
            kp_path = p["keypoints"] / f"{r['clip_id']}_keypoints.csv"
            if not std_path.exists() or not kp_path.exists():
                continue
            metrics = team09.usable_metrics(json.loads(std_path.read_text(encoding="utf-8")), glb_only)
            if ex["target"] not in metrics:
                continue  # 이 각도에서는 해당 항목을 판정하지 않음 (예: 무릎 모으기는 정면만)
            tpl_path = p["reference"] / f"{r['action_id']}_{r['view']}_template.json"
            tpl = json.loads(tpl_path.read_text(encoding="utf-8"))["curve"] if tpl_path.exists() else None
            fid, t, xy, conf = load_keypoints_csv(kp_path)

            for s in ex["strengths"]:
                seq = build_frame_features(fid, t, ex["fn"](xy, s), conf, r["view"], kp_conf, "auto")
                res = judge_seq(seq, r, cfg, metrics, tpl, kp_conf, check_ratio)
                v = [x[0] for x in res]
                rows.append(dict(
                    experiment=name, strength=s, clip_id=r["clip_id"], view=r["view"], split=r["split"],
                    n_reps=len(v), n_ok=v.count("OK"), n_ng=v.count("NG"), n_check=v.count("CHECK"),
                    n_ng_target=sum(1 for vv, bad in res if vv == "NG" and any(b.startswith(ex["target"]) for b in bad)),
                    rules_hit="; ".join(f"{k}x{c}" for k, c in sorted(pd.Series(
                        [b for vv, bad in res if vv == "NG" for b in bad]).value_counts().items())),
                ))
            print(f"[{name:>8}] {r['clip_id']} 완료")

    if not rows:
        print("실험할 Clip이 없습니다. split·standards·keypoints를 확인하세요.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(p["tables"] / "ng_sensitivity.csv", index=False, encoding="utf-8-sig")

    base = df[df["strength"] == 0].groupby("experiment")["n_reps"].sum()
    g = df.groupby(["experiment", "strength"], as_index=False)[["n_reps", "n_ok", "n_ng", "n_check", "n_ng_target"]].sum()
    g["clips"] = df.groupby(["experiment", "strength"])["clip_id"].nunique().to_numpy()
    judged = (g["n_ok"] + g["n_ng"]).replace(0, np.nan)
    g["ng_rate"] = (g["n_ng"] / judged).round(3)                  # 판정된 회차 중 NG 비율
    g["target_rate"] = (g["n_ng_target"] / judged).round(3)       # 그중 '변형한 항목'으로 잡힌 비율
    g["check_rate"] = (g["n_check"] / g["n_reps"].replace(0, np.nan)).round(3)
    g["rep_found"] = (g["n_reps"] / g["experiment"].map(base)).round(3)  # 원본 대비 1회 구간 검출 비율
    g = g[["experiment", "strength", "clips", "n_reps", "n_ok", "n_ng", "n_check",
           "ng_rate", "target_rate", "check_rate", "rep_found"]]
    g.to_csv(p["tables"] / "ng_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[NG 검출 민감도] strength 0 = 원본(정석). 강도가 커질수록 ng_rate가 1에 가까워져야 정상")
    print(g.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 한글 폰트: 윈도우 맑은 고딕(WSL에서 바로 읽음) → 나눔고딕 → 없으면 영어 라벨
        ko = False
        for fp in ["/mnt/c/Windows/Fonts/malgun.ttf",
                   "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
            if Path(fp).exists():
                font_manager.fontManager.addfont(fp)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
                plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지
                ko = True
                break
        names_ko = {"shallow": "덜 앉기 (스쿼트 점프, 움직임 폭 축소)",
                    "knee_in": "무릎 모으기 (스쿼트 점프 정면)",
                    "hip_sag": "허리 처짐 (버피 측면, 몸통 길이 대비)",
                    "hip_pike": "엉덩이 들림 (버피 측면, 몸통 길이 대비)"}

        fig, ax = plt.subplots(figsize=(8, 4.8))
        for name in g["experiment"].unique():
            sub = g[g["experiment"] == name]
            ax.plot(sub["strength"] * 100, sub["ng_rate"] * 100, marker="o",
                    label=names_ko[name] if ko else EXPERIMENTS[name]["plot"])
        ax.set_xlabel("변형 강도 (%)" if ko else "perturbation strength (%)")
        ax.set_ylabel("NG로 판정된 회차 비율 (%)" if ko else "reps judged NG (%)")
        ax.set_title("정석 영상을 변형했을 때 NG 검출률 (팀05)" if ko
                     else "NG detection vs. perturbation strength (team05)")
        ax.set_ylim(-5, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(p["tables"] / "ng_sensitivity.png", dpi=150)
        print(f"\n그래프: {p['tables'] / 'ng_sensitivity.png'} ({'한글' if ko else '영어'} 라벨)")
    except Exception as e:  # 그래프는 선택 사항
        print(f"(그래프 생략: {e})")
    print(f"저장: {p['tables'] / 'ng_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
