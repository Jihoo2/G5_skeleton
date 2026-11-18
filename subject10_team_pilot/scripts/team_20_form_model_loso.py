"""
[팀 20] 정석 이상 탐지 모델 — 실제 정석 영상 중심 + GLB 증강 보충, 사람 단위 교차 검증(LOSO)

데이터 (동작마다 하나의 모델, 촬영 각도는 섞어서 학습 — GLB 증강이 0~90° 각도를 채워 줌)
  실제 정석 1회 : split ∈ {train, val_ext}, form_label=OK, review_status≠EXCLUDE·RESHOOT
                  configs/<team>_rep_labels.csv 에 적힌 회차(덜 앉은 회차 등)는 정석에서 뺌
  GLB 증강      : reports/.../glb_aug/<action>_reps.npz 의 feat_model (team_18 실행 필요)
                  실제 영상과 체계적으로 다르므로(모캡 수행자가 더 깊게 움직임) 각도별로 실제 정석 평균에
                  맞춘 뒤 섞음(카메라·체형 다양성만 보탬). 가중치도 낮게:
                  GLB 전체 가중치 = 실제 전체 가중치 × --glb-weight (기본 0.5)

LOSO (Leave-One-Subject-Out)
  사람 s 한 명을 빼고 나머지 실제 정석 + GLB로 학습 → s의 1회 동작 점수
    - 정석 오판정  : s의 정석 1회 중 score > τ 비율 (낮을수록 좋음)
    - 틀린 자세 검출: s의 Keypoint를 team_11과 같은 방식으로 변형(덜 앉기·무릎 모으기·허리 처짐·엉덩이 들림)
                     한 뒤 score > τ 비율 (높을수록 좋음)
  여러 τ(1.0 ~ 3.0)에 대해 표를 내서, val 단계에서 τ를 고름 (test는 보지 않음)
  val split(S03, 덜 앉은 스쿼트 점프)은 학습에 쓰지 않고 '실제 틀린 자세'로 따로 점수만 냄

최종 모델: 모든 실제 정석 + GLB로 다시 학습 → reports/.../models/form_model_<action>.npz

출력
  tables/form_model_loso_reps.csv     fold·사람·회차별 점수
  tables/form_model_loso_summary.csv  동작 × τ 별 정석 오판정률 / 변형 검출률
  models/form_model_<action>.npz

실행
  python scripts/team_18_glb_augment.py     # feat_model 포함해서 다시 생성 (한 번)
  python scripts/team_20_form_model_loso.py
  python scripts/team_20_form_model_loso.py --glb-weight 0   # GLB 없이 비교
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.form_model import FormModel, rep_block, series_frame  # noqa: E402
from src.team_feature_utils import (  # noqa: E402
    build_frame_features, load_keypoints_csv, load_team_config, rep_metrics, seq_fps, team_paths,
    view_group,
)

TAUS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
FIT_SPLITS = ["train", "val_ext"]


def load_team11():
    spec = importlib.util.spec_from_file_location("team11", ROOT / "scripts" / "team_11_ng_sensitivity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 동작별로 검증에 쓸 변형 (이름, 함수 이름, 강도, 적용할 각도)
PERTURB = {
    "A008": [("덜 앉기 40%", "shallow", 0.4, None), ("무릎 모으기 30%", "knee_in", 0.3, {"front"})],
    "A001": [("허리 처짐 15%", "hip_sag", 0.15, {"side"}), ("엉덩이 들림 40%", "hip_pike", 0.4, {"side"})],
    "A004": [("덜 내려가기 40%", "shallow", 0.4, None)],
}


def clip_reps(seq: pd.DataFrame, view: str, cfg: dict, aid: str):
    kp_conf = cfg["keypoint_confidence"]
    reps, _ = rep_metrics(seq, view_group(view, cfg), seq_fps(seq, cfg["fps"]), kp_conf,
                          cfg["actions"][aid].get("leg_combine", "mean"))
    ser = series_frame(seq, kp_conf)
    return reps, [rep_block(ser, r.start_frame, r.end_frame) for r in reps.itertuples()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb-weight", type=float, default=0.5)
    ap.add_argument("--actions", nargs="+")
    args = ap.parse_args()

    cfg = load_team_config()
    p = team_paths(cfg)
    kp_conf = cfg["keypoint_confidence"]
    model_dir = p["reports"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    t11 = load_team11()
    exp_fn = {"shallow": t11.shallow, "knee_in": t11.knee_in,
              "hip_sag": lambda xy, g: t11.hip_shift(xy, g), "hip_pike": lambda xy, g: t11.hip_shift(xy, -g)}

    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    man = man[~man["review_status"].isin(["EXCLUDE", "RESHOOT"])]
    lab_path = ROOT / "configs" / f"{cfg['team_id']}_rep_labels.csv"
    flagged = set()
    if lab_path.exists():
        lab = pd.read_csv(lab_path, dtype=str).fillna("")
        flagged = {(c, int(n)) for c, n, l in zip(lab["clip_id"], lab["rep"], lab["label"]) if l == "NG"}

    rep_rows, summary = [], []
    for aid, a in cfg["actions"].items():
        if (args.actions and aid not in args.actions) or aid not in PERTURB:
            continue
        # ---- 실제 1회 동작 수집 ----
        real_X, real_meta, perturbed = [], [], []
        for _, r in man[man["action_id"] == aid].iterrows():
            sp = p["sequences"] / f"{r['clip_id']}_sequence.csv"
            if not sp.exists():
                continue
            seq = pd.read_csv(sp)
            reps, blocks = clip_reps(seq, r["view"], cfg, aid)
            for rr, b in zip(reps.itertuples(), blocks):
                kind = ("val_ng" if r["split"] == "val" and r["form_label"] == "NG" else
                        "ok" if r["form_label"] == "OK" and (r["clip_id"], rr.rep) not in flagged else
                        "flagged" if r["form_label"] == "OK" else "other")
                real_X.append(b)
                real_meta.append(dict(clip_id=r["clip_id"], subject=r["subject_id"], view=r["view"],
                                      split=r["split"], rep=rr.rep, kind=kind))
            # 변형 (학습에 쓰는 split의 정석 영상만)
            kp = p["keypoints"] / f"{r['clip_id']}_keypoints.csv"
            if r["split"] in FIT_SPLITS and r["form_label"] == "OK" and kp.exists():
                fid, t, xy, conf = load_keypoints_csv(kp)
                for name, fn, s, views in PERTURB[aid]:
                    if views and view_group(r["view"], cfg) not in views:
                        continue
                    seq2 = build_frame_features(fid, t, exp_fn[fn](xy, s), conf, r["view"], kp_conf)
                    _, b2 = clip_reps(seq2, r["view"], cfg, aid)
                    perturbed += [dict(subject=r["subject_id"], clip_id=r["clip_id"], view=r["view"],
                                       experiment=name, X=b) for b in b2]
        if not real_X:
            print(f"[{aid}] 실제 Sequence 없음 → 건너뜀")
            continue
        RX, meta = np.array(real_X), pd.DataFrame(real_meta)
        fit_mask = (meta["kind"] == "ok") & meta["split"].isin(FIT_SPLITS)
        gpath = p["reports"] / "glb_aug" / f"{aid}_reps.npz"
        GX = (np.load(gpath)["feat_model"] if gpath.exists() and "feat_model" in np.load(gpath).files
              else np.empty((0,) + RX.shape[1:]))
        gcsv = p["reports"] / "glb_aug" / f"{aid}_reps.csv"
        GV = pd.read_csv(gcsv)["view"].to_numpy() if len(GX) and gcsv.exists() else np.array([], object)
        if args.glb_weight > 0 and len(GX) == 0:
            print(f"[{aid}] glb_aug feat_model 없음 → team_18을 다시 실행하세요 (지금은 실제 영상만 사용)")

        def align_glb(mask):
            """GLB를 각도별로 실제 정석 평균에 맞춤 → GLB는 깊이 차이가 아니라 카메라·체형 다양성만 보탬"""
            G = GX.astype(float).copy()
            real_views = meta.loc[mask, "view"].map(lambda v: view_group(v, cfg)).to_numpy()
            all_mean = np.nanmean(RX[mask], 0)
            for v in np.unique(GV):
                gm = GV == v
                rm = real_views == v
                target = np.nanmean(RX[mask][rm], 0) if rm.sum() >= 3 else all_mean
                G[gm] += np.nan_to_num(target - np.nanmean(G[gm], 0))
            return G

        def fit(mask):
            Xr = RX[mask]
            X = np.concatenate([Xr, align_glb(mask)]) if args.glb_weight > 0 and len(GX) else Xr
            w = np.ones(len(X))
            if len(X) > len(Xr):
                w[len(Xr):] = args.glb_weight * len(Xr) / len(GX)
            is_real = np.r_[np.ones(len(Xr), bool), np.zeros(len(X) - len(Xr), bool)]
            groups = np.r_[meta.loc[mask, "subject"].to_numpy(object), np.full(len(X) - len(Xr), None, object)]
            return FormModel().fit(X, w, is_real, groups)

        subjects = sorted(meta.loc[fit_mask, "subject"].unique())
        print(f"\n[{aid} {a['label']}] 실제 정석 1회 {int(fit_mask.sum())}개 / 사람 {len(subjects)}명, "
              f"GLB 증강 {len(GX)}개 (가중치 {args.glb_weight})")
        if len(subjects) < 3:
            print("  사람이 3명 미만이라 LOSO 평가를 하지 않습니다.")
            continue
        for s in subjects:
            m = fit(fit_mask & (meta["subject"] != s))
            held = meta["subject"] == s
            sc = m.score(RX[held])
            for (_, row), v in zip(meta[held].iterrows(), sc):
                rep_rows.append(dict(action_id=aid, fold=s, **row.to_dict(), experiment="", score=round(float(v), 3)))
            for d in [d for d in perturbed if d["subject"] == s]:
                rep_rows.append(dict(action_id=aid, fold=s, clip_id=d["clip_id"], subject=s, view=d["view"],
                                     split="perturbed", rep=None, kind="perturbed", experiment=d["experiment"],
                                     score=round(float(m.score(d["X"][None])[0]), 3)))
        # val (S03 등)은 학습에 안 쓰였으므로 전체 모델 점수로
        final = fit(fit_mask)
        val_mask = meta["split"] == "val"
        for (_, row), v in zip(meta[val_mask].iterrows(), final.score(RX[val_mask])):
            rep_rows.append(dict(action_id=aid, fold="final", **row.to_dict(), experiment="", score=round(float(v), 3)))
        final.to_npz(model_dir / f"form_model_{aid}.npz", glb_weight=args.glb_weight,
                     n_real=int(fit_mask.sum()), n_glb=len(GX))

        df = pd.DataFrame([r for r in rep_rows if r["action_id"] == aid])
        for tau in TAUS:
            row = dict(action_id=aid, tau=tau)
            ok = df[(df["kind"] == "ok") & (df["fold"] != "final")]
            row["정석_오판정률"] = round(float((ok["score"] > tau).mean()), 3) if len(ok) else None
            row["정석_n"] = len(ok)
            for name, *_ in PERTURB[aid]:
                pe = df[df["experiment"] == name]
                row[f"검출_{name}"] = round(float((pe["score"] > tau).mean()), 3) if len(pe) else None
            vn = df[df["kind"] == "val_ng"]
            if len(vn):
                row["val_NG영상_회차_NG비율"] = round(float((vn["score"] > tau).mean()), 3)
            fl = df[df["kind"] == "flagged"]
            if len(fl):
                row["덜앉은회차(라벨)_검출"] = f"{int((fl['score'] > tau).sum())}/{len(fl)}"
            summary.append(row)
        print(pd.DataFrame([r for r in summary if r["action_id"] == aid]).drop(columns="action_id").to_string(index=False))

    pd.DataFrame(rep_rows).to_csv(p["tables"] / "form_model_loso_reps.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(p["tables"] / "form_model_loso_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n저장: {p['tables'] / 'form_model_loso_summary.csv'}, 모델: {model_dir}")
    print("τ는 정석 오판정률이 낮으면서 검출률이 높은 값으로 고르고(검증 단계), test_ext에서는 고정해서 한 번만 평가합니다.")


if __name__ == "__main__":
    main()
