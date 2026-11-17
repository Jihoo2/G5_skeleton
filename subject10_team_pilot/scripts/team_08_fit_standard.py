"""
[팀 08] 정석 자세 기준(허용 범위) 만들기 — NG 데이터 없이

팀 방침: "정석 자세를 벗어나면 무조건 NG"
  → NG 영상을 모으지 않고, 정석(OK) 영상들의 분포로 '허용 범위'를 만듭니다.

동작 × 촬영 각도마다 (측정값 목록은 configs/team_project.json 의 actions.*.metrics_by_view)
  1) train split 의 RGB, form_label=OK, review_status ∉ {RESHOOT, EXCLUDE} Clip을 모음
  2) 각 Clip의 1회 구간 측정값(rep_metrics) + GLB 곡선과의 차이(curve_dist)
  3) 측정값별 허용 범위
       center = 팀 OK 평균
       half   = max(k_std × 표준편차, min_halfwidth)     (config: standard)
       범위   = [center − half, center + half]
     curve_dist 는 한쪽(클수록 나쁨)만: 상한 = max(평균 + k·std, min_halfwidth)
  4) 1회 구간이 min_samples 미만이거나 사람이 min_subjects 미만이면 provisional=true 로 표시하고
     (09 판정에서는 임시 기준을 NG가 아니라 '참고'로만 사용)
     half 를 min_halfwidth × 1.5 로 넓게 잡음 (팀 표본이 0이면 GLB 값을 중심으로 사용)
  5) GLB 값과 팀 평균의 차이(bias)를 같이 기록 → 사람마다 다른 신체 비율·실행 차이를 확인
  6) config actions.*.rules 에 팀이 정한 경계(lo/hi)가 있으면 그 값이 우선 (source=rule)
     → 사람마다 실행이 달라 데이터만으로 정석을 정할 수 없을 때, 정석의 '정의'를 숫자로 고정
  7) half 가 max_halfwidth 를 넘으면 unreliable=true
     → 그 각도에서는 정석 영상끼리도 값이 크게 흔들리는 항목(예: 정면에서 본 무릎 각도)
     → 09 판정에서 제외하고 표에 남김 (그 항목은 다른 각도에서 판정)

출력: reports/team_projects/<team>/standards/<action>_<view>.json, tables/standards_summary.csv

실행
  python scripts/team_08_fit_standard.py
  python scripts/team_08_fit_standard.py --splits train val   # val까지 기준에 포함(최종 배포용)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import (  # noqa: E402
    curve_distance, load_team_config, metrics_for, rep_metrics, team_paths, view_group,
)


def collect_reps(man: pd.DataFrame, p: dict, cfg: dict) -> pd.DataFrame:
    """Manifest 행들 → 1회 구간 측정값 + curve_dist 표"""
    fps, kp_conf = cfg["fps"], cfg["keypoint_confidence"]
    out = []
    for _, r in man.iterrows():
        sp = p["sequences"] / f"{r['clip_id']}_sequence.csv"
        if not sp.exists():
            continue
        seq = pd.read_csv(sp)
        reps, curves = rep_metrics(seq, r["view"], fps, kp_conf,
                                    cfg["actions"].get(r["action_id"], {}).get("leg_combine", "mean"))
        tpl_path = p["reference"] / f"{r['action_id']}_{r['view']}_template.json"
        tpl = json.loads(tpl_path.read_text(encoding="utf-8"))["curve"] if tpl_path.exists() else None
        for i, row in reps.iterrows():
            d = row.to_dict()
            d["curve_dist"] = curve_distance(curves[i], tpl) if tpl else np.nan
            d.update(clip_id=r["clip_id"], subject_id=r["subject_id"],
                     action_id=r["action_id"], view=r["view"])
            out.append(d)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train"])
    args = ap.parse_args()

    cfg = load_team_config()
    p = team_paths(cfg)
    st = cfg["standard"]
    k, nmin, mh = st["k_std"], st["min_samples"], st["min_halfwidth"]
    smin = st.get("min_subjects", 3)
    mx = st.get("max_halfwidth", {})

    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    pool = man[(man["source_type"] == "RGB") & (man["split"].isin(args.splits))
               & (man["form_label"] == "OK")
               & (~man["review_status"].isin(["RESHOOT", "EXCLUDE"]))]
    reps = collect_reps(pool, p, cfg)
    if len(reps):
        # 확인 신호가 많은 구간은 기준 계산에서 제외
        reps = reps[reps["low_conf_ratio"] <= st["low_conf_check_ratio"]]
        reps.to_csv(p["tables"] / "standard_source_reps.csv", index=False, encoding="utf-8-sig")

    views = sorted(set(man["view"]) | set(cfg["views_deg"]))
    summary = []
    for aid, a in cfg["actions"].items():
        for view in views:
            tpl_path = p["reference"] / f"{aid}_{view}_template.json"
            glb_m = json.loads(tpl_path.read_text(encoding="utf-8"))["metrics"] if tpl_path.exists() else {}
            sub = reps[(reps["action_id"] == aid) & (reps["view"] == view)] if len(reps) else pd.DataFrame()
            # 팀 영상·GLB가 없어도 이 각도에 적용되는 정석 규칙이 있으면 규칙만으로 기준 생성
            has_rule = any(
                r and (not r.get("views") or view_group(view, cfg) in r["views"])
                for r in a.get("rules", {}).values()
            )
            if sub.empty and not glb_m and not has_rule:
                continue
            metrics = {}
            for m in metrics_for(a, view, cfg):
                vals = sub[m].dropna().to_numpy(float) if m in sub else np.array([])
                glb_v = glb_m.get(m)
                n = len(vals)
                rule = a.get("rules", {}).get(m)
                if rule and rule.get("views") and view_group(view, cfg) not in rule["views"]:
                    rule = None  # 절대 각도 규칙은 정해 둔 각도(예: 측면)에서만 사용 — 사선·정면은 원근 때문에 각도가 커 보임
                if n == 0 and rule is None and (glb_v is None or m == "curve_dist"):
                    continue
                if n == 0 and glb_v is None:  # 규칙만 있는 항목
                    metrics[m] = dict(lo=rule.get("lo"), hi=rule.get("hi"), center=None, std=None, n=0,
                                      provisional=False, unreliable=False, source="rule",
                                      glb=None, bias_team_minus_glb=None)
                    summary.append(dict(action_id=aid, view=view, metric=m, **metrics[m]))
                    continue
                n_subj = int(sub.loc[sub[m].notna(), "subject_id"].nunique()) if n else 0
                provisional = n < nmin or n_subj < smin
                center = float(vals.mean()) if n else float(glb_v)
                std = float(vals.std(ddof=1)) if n >= 2 else np.nan
                half = mh[m] * 1.5 if provisional else max(k * std, mh[m])
                if m == "curve_dist":  # 한쪽 기준
                    lo, hi = None, (max(center + k * std, mh[m]) if not provisional
                                    else max(center, 0) + mh[m] * 1.5)
                else:
                    lo, hi = center - half, center + half
                width = (hi - center) if lo is None else half
                unreliable = bool(m in mx and width > mx[m])
                metrics[m] = dict(
                    lo=None if lo is None else round(lo, 4), hi=round(hi, 4),
                    center=round(center, 4), std=None if np.isnan(std) else round(std, 4),
                    n=n, provisional=provisional, unreliable=unreliable,
                    glb=None if glb_v is None else round(glb_v, 4),
                    bias_team_minus_glb=None if (glb_v is None or n == 0) else round(center - glb_v, 4),
                    source="team" if n else "glb",
                )
                if rule:  # 팀이 정한 정석 규칙이 있으면 그 경계가 우선 (데이터 분포보다 정의가 먼저)
                    metrics[m].update(lo=rule.get("lo"), hi=rule.get("hi"),
                                      provisional=False, unreliable=False, source="rule")
                summary.append(dict(action_id=aid, view=view, metric=m, **metrics[m]))
            if not metrics:
                continue
            doc = {"action_id": aid, "action_label": a["label"], "view": view,
                   "source_splits": args.splits, "k_std": k,
                   "n_clips": int(sub["clip_id"].nunique()) if len(sub) else 0,
                   "n_subjects": int(sub["subject_id"].nunique()) if len(sub) else 0,
                   "metrics": metrics}
            (p["standards"] / f"{aid}_{view}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            prov = sum(v["provisional"] for v in metrics.values())
            unrel = [m for m, v in metrics.items() if v["unreliable"]]
            print(f"[{aid} {view:>6}] 사람 {doc['n_subjects']}명 / Clip {doc['n_clips']}개, "
                  f"기준 {len(metrics)}개 (임시 {prov}개)"
                  + (f", 흔들림 커서 판정 제외: {', '.join(unrel)}" if unrel else ""))

    pd.DataFrame(summary).to_csv(p["tables"] / "standards_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {p['standards']}")


if __name__ == "__main__":
    main()
