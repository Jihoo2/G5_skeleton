"""
[팀 19] GLB 정석 분포를 실제 촬영 영상으로 보정 → GLB 메인 판정 기준 (포트폴리오 2단계)

생각
  GLB 증강(team_18)은 '정석 동작이 여러 카메라·체형에서 어떻게 보이는가'의 분포를 줍니다.
  하지만 실제 웹캠 + YOLO 값은 GLB 투영과 일정하게 어긋납니다(관절 위치 정의, 원근, 검출 습관).
  그래서 (동작, 각도, 측정값)마다
      shift = 실제 정석 1회 중앙값 − GLB 증강 중앙값
      scale = 실제 IQR / GLB IQR      (실제 표본이 많을 때만, 0.5~2배로 제한)
  를 학습해서 GLB 분포를 실제 카메라 쪽으로 옮깁니다. → "GLB가 기준, 영상은 보정"
  실제 표본이 없거나 적은 칸은 shift 없이 GLB 분포 그대로 씁니다.

허용 범위
  보정된 GLB 분포의 [q, 1−q] 분위(기본 q=0.01) ± 여유(min_halfwidth × margin_ratio)
  측정값마다 NG 방향이 정해져 있음 (깊이는 덜 앉았을 때만, 허리 처짐은 아래쪽만 …)
  팀이 정한 정석 규칙(actions.*.rules)이 있으면 그 경계가 우선 (--no-rules 로 끌 수 있음)
  GLB만으로 만든 칸(실제 보정 없음)은 기본적으로 provisional = 참고용 (config: standard.glb_only_active)

학습에 쓰는 실제 영상: 기본 train split (팀 RGB, form_label=OK) — val·val_ext는 평가용으로 남김
  최종 배포 때만 --fit-splits train val val_ext

출력
  reports/.../standards_glb/<action>_<view>.json   team_09 --standards glb 로 판정
  reports/.../tables/glb_calibration.csv            칸별 GLB·실제 중앙값, shift, scale, 최종 범위, 기존 팀 기준

실행
  python scripts/team_18_glb_augment.py      # 먼저
  python scripts/team_19_calibrate_glb.py
  python scripts/team_09_judge.py --standards glb
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

from src.team_feature_utils import (  # noqa: E402
    curve_distance, load_team_config, metrics_for, team_paths, view_group,
)

# NG 방향: hi = 값이 클 때만 NG, lo = 작을 때만 NG, both = 양쪽
DIRECTION = {
    "depth": "hi", "body_line_min": "lo", "body_line_max": "hi", "knee_spread_bottom": "lo",
    "curve_dist": "hi", "knee_min": "both", "hip_min": "both", "elbow_min": "both", "duration_s": "both",
}
VIEWS = ("front", "diag", "side")


def load_team08():
    spec = importlib.util.spec_from_file_location("team08", ROOT / "scripts" / "team_08_fit_standard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def glb_curve_dist(aid: str, view: str, df: pd.DataFrame, npz: dict, p: dict) -> np.ndarray:
    tpl_path = p["reference"] / f"{aid}_{view}_template.json"
    if not tpl_path.exists():
        return np.full(len(df), np.nan)
    tpl = json.loads(tpl_path.read_text(encoding="utf-8"))["curve"]
    curves = npz["curves"]
    out = np.full(len(df), np.nan)
    for k, i in enumerate(df.index):
        c = curves[i]
        out[k] = curve_distance({"knee": c[0], "hip": c[1], "leg_ratio": c[2]}, tpl)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-splits", nargs="+", default=["train"])
    ap.add_argument("--no-rules", action="store_true", help="팀 정석 규칙을 쓰지 않고 GLB 보정 범위만 사용")
    args = ap.parse_args()
    if set(args.fit_splits) & {"test", "test_ext"}:
        raise SystemExit("[중지] test 계열로는 보정하지 않습니다 (Test 누수).")

    cfg = load_team_config()
    p = team_paths(cfg)
    st = cfg["standard"]
    q = float(st.get("glb_quantile", 0.01))
    margin_ratio = float(st.get("glb_margin_ratio", 0.5))
    min_real = int(st.get("glb_min_real_reps", 5))
    glb_only_active = bool(st.get("glb_only_active", False))
    mh, mx = st["min_halfwidth"], st.get("max_halfwidth", {})

    aug_dir = p["reports"] / "glb_aug"
    out_dir = p["reports"] / "standards_glb"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 실제 정석 1회 측정값 (team_08과 같은 방식으로 수집)
    t08 = load_team08()
    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    pool = man[man["split"].isin(args.fit_splits) & (man["form_label"] == "OK")
               & (~man["review_status"].isin(["RESHOOT", "EXCLUDE"]))].copy()
    pool["view"] = pool["view"].map(lambda v: view_group(v, cfg))
    real = t08.collect_reps(pool, p, cfg)
    if len(real):
        real = real[real["low_conf_ratio"] <= st["low_conf_check_ratio"]]
    old_std = {}
    for f in p["standards"].glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for m, s in d["metrics"].items():
            old_std[(d["action_id"], d["view"], m)] = (s.get("lo"), s.get("hi"), s.get("source"))

    table = []
    for aid, a in cfg["actions"].items():
        csv_path, npz_path = aug_dir / f"{aid}_reps.csv", aug_dir / f"{aid}_reps.npz"
        if not csv_path.exists():
            continue
        aug = pd.read_csv(csv_path)
        npz = dict(np.load(npz_path, allow_pickle=True))
        for view in VIEWS:
            ms = metrics_for(a, view, cfg)
            g = aug[aug["view"] == view]
            if not ms or g.empty:
                continue
            if "curve_dist" in ms:
                g = g.copy()
                g["curve_dist"] = glb_curve_dist(aid, view, g, npz, p)
            r = real[(real["action_id"] == aid) & (real["view"] == view)] if len(real) else pd.DataFrame()
            metrics = {}
            for m in ms:
                gv = g[m].dropna().to_numpy(float) if m in g else np.array([])
                if len(gv) < 20:
                    continue
                rv = r[m].dropna().to_numpy(float) if m in r else np.array([])
                n_real = len(rv)
                n_subj = int(r.loc[r[m].notna(), "subject_id"].nunique()) if n_real else 0
                g_med = float(np.median(gv))
                shift = float(np.median(rv) - g_med) if n_real >= min_real else 0.0
                scale = 1.0
                if n_real >= 2 * min_real:
                    g_iqr = np.subtract(*np.percentile(gv, [75, 25]))
                    r_iqr = np.subtract(*np.percentile(rv, [75, 25]))
                    if g_iqr > 1e-9 and r_iqr > 0:
                        scale = float(np.clip(r_iqr / g_iqr, 0.5, 2.0))
                cal = (gv - g_med) * scale + g_med + shift
                margin = mh.get(m, 0.0) * margin_ratio
                d = DIRECTION.get(m, "both")
                lo = float(np.quantile(cal, q) - margin) if d in ("lo", "both") else None
                hi = float(np.quantile(cal, 1 - q) + margin) if d in ("hi", "both") else None
                calibrated = n_real >= min_real
                width = ((hi if hi is not None else np.quantile(cal, 1 - q)) -
                         (lo if lo is not None else np.quantile(cal, q))) / 2
                s = dict(lo=None if lo is None else round(lo, 4), hi=None if hi is None else round(hi, 4),
                         center=round(g_med + shift, 4), n=len(gv), n_real=n_real, n_real_subjects=n_subj,
                         shift=round(shift, 4), scale=round(scale, 3),
                         provisional=not (calibrated or glb_only_active),
                         unreliable=bool(m in mx and width > mx[m]),
                         source="glb_calibrated" if calibrated else "glb")
                rule = a.get("rules", {}).get(m)
                if rule and rule.get("views") and view not in rule["views"]:
                    rule = None
                if rule and not args.no_rules:
                    s.update(lo=rule.get("lo"), hi=rule.get("hi"), provisional=False, unreliable=False,
                             source="rule", glb_lo=s["lo"], glb_hi=s["hi"])
                metrics[m] = s
                old = old_std.get((aid, view, m), (None, None, None))
                table.append(dict(action_id=aid, view=view, metric=m, glb_median=round(g_med, 3),
                                  real_median=round(float(np.median(rv)), 3) if n_real else None,
                                  n_real=n_real, n_real_subjects=n_subj, shift=s["shift"], scale=s["scale"],
                                  lo=s["lo"], hi=s["hi"], source=s["source"], provisional=s["provisional"],
                                  unreliable=s["unreliable"], glb_lo=s.get("glb_lo", s["lo"]),
                                  glb_hi=s.get("glb_hi", s["hi"]),
                                  team_lo=old[0], team_hi=old[1], team_source=old[2]))
            if metrics:
                doc = {"action_id": aid, "action_label": a["label"], "view": view,
                       "source": "glb_augment+calibration", "fit_splits": args.fit_splits,
                       "metrics": metrics}
                (out_dir / f"{aid}_{view}.json").write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    tab = pd.DataFrame(table)
    tab.to_csv(p["tables"] / "glb_calibration.csv", index=False, encoding="utf-8-sig")
    show = ["action_id", "view", "metric", "glb_median", "real_median", "n_real", "shift",
            "glb_lo", "glb_hi", "lo", "hi", "source", "provisional", "team_lo", "team_hi"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tab[show].to_string(index=False))
    print(f"\n저장: {out_dir}")
    print("→ python scripts/team_09_judge.py --standards glb   (기존 팀 기준과 결과 비교)")


if __name__ == "__main__":
    main()
