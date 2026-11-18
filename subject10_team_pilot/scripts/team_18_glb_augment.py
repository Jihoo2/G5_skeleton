"""
[팀 18] GLB 정석 데이터 대량 생성 (포트폴리오 1단계: GLB 메인)

동작마다 configs/team_project.json 의 actions.*.glb 를 불러와
카메라 방향·높이·거리, 체형, 속도, YOLO 흔들림을 무작위로 바꾼 '가상 촬영본'을 N개 만들고,
웹캠 영상과 같은 경로(build_frame_features → rep_metrics)로 1회 단위 측정값을 계산합니다.
(src/glb_augment.py 참고. 범위 제외 동작 A006은 건너뜀)

출력 (reports/team_projects/<team>/glb_aug/)
  <action>_reps.csv        1행 = 가상 촬영본의 1회 동작: 촬영 조건 + 측정값(knee_min, depth …)
  <action>_reps.npz        같은 순서의 배열 — 모델 학습용
                             curves (n, 3, 48)  : knee, hip, leg_ratio 1회 곡선
                             feat38 (n, 48, 38) : 1회 구간을 48점으로 맞춘 교과 38-D Feature
  glb_aug_summary.csv      동작 × 각도 × 측정값 분포(p5·중앙값·p95) + 현재 규칙 위반 비율
화면: 각도별 분포와 "정석 GLB인데 현재 규칙으로 NG가 되는 비율"
      → 이 비율이 큰 규칙은 정석을 부정하므로 경계 재검토 대상

이 데이터는 VICON 수행자 1명의 정석 동작에서 나온 것이므로 Split 상 train 쪽(기준·학습)에만 씁니다.
val·val_ext·test 평가에 섞지 않습니다.

실행
  python scripts/team_18_glb_augment.py                 # 동작당 300개
  python scripts/team_18_glb_augment.py --n 50 --actions A004
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.glb_augment import glb_coco3d, make_variant_xy, sample_variant  # noqa: E402
from src.glb_utils import evaluate  # noqa: E402
from src.team_feature_utils import (  # noqa: E402
    TEMPLATE_LEN, build_frame_features, feature_columns, load_team_config, metrics_for,
    rep_metrics, team_paths, view_group,
)

SKIP_LABELS = {"plank_kick_through"}


def resample_block(a: np.ndarray, n: int = TEMPLATE_LEN) -> np.ndarray:
    """(T, F) → (n, F) 선형 보간, NaN은 열마다 보간 후 남으면 NaN 유지"""
    T = len(a)
    if T < 2:
        return np.full((n, a.shape[1]), np.nan)
    src = np.linspace(0, T - 1, n)
    out = np.empty((n, a.shape[1]))
    idx = np.arange(T)
    for j in range(a.shape[1]):
        col = a[:, j]
        ok = np.isfinite(col)
        out[:, j] = np.interp(src, idx[ok], col[ok]) if ok.sum() >= 2 else np.nan
    return out


def rule_violation(df: pd.DataFrame, rules: dict, view: str) -> dict:
    out = {}
    for m, r in rules.items():
        if not r or m not in df:
            continue
        if r.get("views") and view not in r["views"]:
            continue
        v = df[m].dropna()
        if v.empty:
            continue
        bad = np.zeros(len(v), bool)
        if r.get("lo") is not None:
            bad |= v.to_numpy() < r["lo"]
        if r.get("hi") is not None:
            bad |= v.to_numpy() > r["hi"]
        out[m] = float(bad.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="동작당 가상 촬영본 수")
    ap.add_argument("--actions", nargs="+", help="예: A004 A008 (기본: GLB가 있는 모든 동작)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_team_config()
    p = team_paths(cfg)
    out_dir = p["reports"] / "glb_aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps, kp_conf = float(cfg["fps"]), cfg["keypoint_confidence"]
    cols38 = feature_columns()
    rng = np.random.default_rng(args.seed)
    summary = []

    for aid, a in cfg["actions"].items():
        if args.actions and aid not in args.actions:
            continue
        if a["label"] in SKIP_LABELS or not a.get("glb"):
            continue
        glb = p["vicon"] / a["glb"]
        if not glb.exists():
            print(f"[{aid}] GLB 없음 → {glb}")
            continue
        t0 = time.time()
        names, P, _ = evaluate(glb, fps=fps)
        X3 = glb_coco3d(names, P)
        rows, curves, feats = [], [], []
        for i in range(args.n):
            v = sample_variant(rng, i)
            view = view_group(f"deg{int(round(v.yaw_deg))}", cfg)
            xy = make_variant_xy(X3, v, rng)
            fid = np.arange(1, len(xy) + 1)
            seq = build_frame_features(fid, (fid - 1) / fps, xy, np.full((len(xy), 17), np.nan), view, kp_conf)
            reps, crv = rep_metrics(seq, view, fps, kp_conf, a.get("leg_combine", "mean"))
            F = seq[cols38].to_numpy(float)
            for (_, rr), c in zip(reps.iterrows(), crv):
                rows.append(dict(action_id=aid, action_label=a["label"], view=view,
                                 **v.as_dict(), **rr.to_dict()))
                curves.append(np.stack([c["knee"], c["hip"], c["leg_ratio"]]))
                feats.append(resample_block(F[int(rr["start_frame"]) - 1:int(rr["end_frame"])]))
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{aid}_reps.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(out_dir / f"{aid}_reps.npz", curves=np.array(curves, dtype=np.float32),
                            feat38=np.array(feats, dtype=np.float32), feature_names=np.array(cols38))
        print(f"\n[{aid} {a['label']}] 가상 촬영본 {args.n}개 → 1회 동작 {len(df)}개 ({time.time() - t0:.0f}초)")

        for view in ("front", "diag", "side"):
            sub = df[df["view"] == view]
            ms = [m for m in metrics_for(a, view, cfg) if m in sub]
            if sub.empty or not ms:
                continue
            viol = rule_violation(sub, a.get("rules", {}), view)
            for m in ms:
                q = sub[m].dropna()
                if q.empty:
                    continue
                row = dict(action_id=aid, view=view, metric=m, n=len(q),
                           p5=round(q.quantile(.05), 3), median=round(q.median(), 3),
                           p95=round(q.quantile(.95), 3), rule_ng_ratio=viol.get(m))
                summary.append(row)
                ng = "" if row["rule_ng_ratio"] is None else f"   현재 규칙 NG {row['rule_ng_ratio']:.0%}"
                print(f"  {view:5s} {m:18s} n={row['n']:4d}  p5={row['p5']:8.3f}  "
                      f"중앙={row['median']:8.3f}  p95={row['p95']:8.3f}{ng}")

    pd.DataFrame(summary).to_csv(out_dir / "glb_aug_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
