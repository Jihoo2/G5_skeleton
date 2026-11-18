"""
[팀 09] 1회 단위 OK / NG / CHECK 판정 + 교정 메시지

판정 규칙 (정석 기준)
  CHECK : 1회 구간을 못 찾음, 또는 무릎·고관절 Confidence 낮은 Frame 비율 > low_conf_check_ratio
          → 자세가 틀렸다고 말하지 않고 "카메라 위치·전신이 보이게" 안내 (교과 2일차 원칙)
  NG    : 핵심 측정값 하나라도 허용 범위 밖 → 벗어난 방향에 맞는 한국어 교정 메시지
  OK    : 모든 핵심 측정값이 허용 범위 안
  판정에서 빼는 항목: unreliable(그 각도에서 흔들림이 큰 항목), 팀 OK 영상 없이 GLB 값만 있는 항목
          (GLB만 있는 항목은 config standard.judge_glb_only=true 일 때만 사용 — 촬영 각도·높이 차이로
           GLB 값이 웹캠과 어긋나므로 기본은 사용하지 않음)

출력
  judgments/rep_judgments.csv   1회 단위 판정·벗어난 항목·메시지
  tables/judgment_summary.csv   Clip 단위 결과 (Clip 안에 NG 1회라도 있으면 NG)
  화면: split별 OK Clip 오판정률(false NG) / NG 라벨 Clip 검출률(있을 때만)

실행
  python scripts/team_09_judge.py                       # val, val_ext (규칙 조정용)
  python scripts/team_09_judge.py --splits train val val_ext
  python scripts/team_09_judge.py --final               # 규칙 동결 후 test, test_ext 한 번만
test / test_ext 는 --final 없이는 판정하지 않습니다 (결과를 보고 규칙을 고치면 Test 누수).
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
    MESSAGES, curve_distance, load_team_config, metrics_for, rep_metrics, seq_fps, team_paths, view_group,
)


def usable_metrics(standard: dict, glb_only: bool) -> dict:
    return {m: s for m, s in standard["metrics"].items()
            if not s.get("unreliable") and (s["n"] > 0 or s.get("source") == "rule" or glb_only)}


def judge_rep(row: dict, metrics: dict) -> tuple[str, list[str], list[str], list[str]]:
    """반환: 판정, 벗어난 항목(확정 기준), 메시지, 참고(임시 기준에서 벗어난 항목)"""
    bad, msgs, notes = [], [], []
    for m, s in metrics.items():
        v = row.get(m)
        if v is None or not np.isfinite(v):
            continue
        hit = None
        if s["lo"] is not None and v < s["lo"]:
            hit = (f"{m}<{s['lo']}", MESSAGES.get((m, "low"), f"{m} 값이 기준보다 작습니다"))
        elif s["hi"] is not None and v > s["hi"]:
            hit = (f"{m}>{s['hi']}", MESSAGES.get((m, "high"), f"{m} 값이 기준보다 큽니다"))
        if hit is None:
            continue
        if s.get("provisional") and s.get("source") != "rule":
            notes.append(hit[0])  # 임시 기준(사람·표본 부족)은 NG로 쓰지 않고 참고로만 기록
        else:
            bad.append(hit[0])
            msgs.append(hit[1])
    return ("NG" if bad else "OK"), bad, msgs, notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "val_ext"])
    ap.add_argument("--standards", choices=["team", "glb"], default="team",
                    help="team: team_08 기준(팀 영상 분포+규칙) / glb: team_19 기준(GLB 증강+영상 보정+규칙)")
    ap.add_argument("--final", action="store_true",
                    help="규칙 동결(rules-v1 태그) 후 test, test_ext 를 한 번만 평가")
    args = ap.parse_args()
    if args.final:
        args.splits = ["test", "test_ext"]
    elif set(args.splits) & {"test", "test_ext"}:
        raise SystemExit("[중지] test / test_ext 는 규칙을 동결한 뒤 --final 로만 판정합니다.")

    cfg = load_team_config()
    p = team_paths(cfg)
    fps, kp_conf = cfg["fps"], cfg["keypoint_confidence"]
    check_ratio = cfg["standard"]["low_conf_check_ratio"]
    glb_only = cfg["standard"].get("judge_glb_only", False)

    man = pd.read_csv(p["manifest"], dtype=str).fillna("")
    man = man[man["split"].isin(args.splits) & (man["review_status"] != "EXCLUDE")]

    rep_rows, clip_rows = [], []
    for _, r in man.iterrows():
        base = dict(clip_id=r["clip_id"], subject_id=r["subject_id"], action_id=r["action_id"],
                    view=r["view"], split=r["split"], source_type=r["source_type"],
                    form_label=r["form_label"])
        if not metrics_for(cfg["actions"][r["action_id"]], r["view"], cfg):
            clip_rows.append(dict(**base, n_reps=0, n_ok=0, n_ng=0, n_check=0, verdict="CHECK",
                                  note=f"이 동작은 {view_group(r['view'], cfg)} 각도에서 판정하지 않음 → 측면·사선으로 촬영"))
            continue
        std_dir = p["standards"] if args.standards == "team" else p["reports"] / "standards_glb"
        std_path = std_dir / f"{r['action_id']}_{r['view']}.json"
        sp = p["sequences"] / f"{r['clip_id']}_sequence.csv"
        if not std_path.exists() or not sp.exists():
            reason = "기준 없음(해당 각도 OK 영상·GLB 필요)" if not std_path.exists() else "sequence 없음"
            clip_rows.append(dict(**base, n_reps=0, n_ok=0, n_ng=0, n_check=0, verdict="CHECK", note=reason))
            continue
        standard = json.loads(std_path.read_text(encoding="utf-8"))
        metrics = usable_metrics(standard, glb_only)
        if not metrics:
            clip_rows.append(dict(**base, n_reps=0, n_ok=0, n_ng=0, n_check=0, verdict="CHECK",
                                  note="이 각도의 팀 정석 영상이 없어 판정 기준 없음"))
            continue
        tpl_path = p["reference"] / f"{r['action_id']}_{r['view']}_template.json"
        tpl = json.loads(tpl_path.read_text(encoding="utf-8"))["curve"] if tpl_path.exists() else None

        seq = pd.read_csv(sp)
        reps, curves = rep_metrics(seq, r["view"], seq_fps(seq, fps), kp_conf,
                                    cfg["actions"][r["action_id"]].get("leg_combine", "mean"))
        if reps.empty:
            clip_rows.append(dict(**base, n_reps=0, n_ok=0, n_ng=0, n_check=1, verdict="CHECK",
                                  note="1회 구간을 찾지 못함 — 전신이 화면에 들어오게 촬영"))
            continue
        verdicts = []
        for i, row in reps.iterrows():
            d = row.to_dict()
            d["curve_dist"] = curve_distance(curves[i], tpl) if tpl else np.nan
            if d["low_conf_ratio"] > check_ratio:
                v, bad, msgs, notes = "CHECK", [], ["관절이 잘 보이지 않습니다. 카메라 위치를 조정하세요"], []
            else:
                v, bad, msgs, notes = judge_rep(d, metrics)
            verdicts.append(v)
            rep_rows.append(dict(**base, **{k: d[k] for k in reps.columns}, curve_dist=d["curve_dist"],
                                 verdict=v, out_of_range="; ".join(bad),
                                 provisional_out="; ".join(notes),
                                 message=" / ".join(dict.fromkeys(msgs))))
        n_ng, n_ok, n_ck = verdicts.count("NG"), verdicts.count("OK"), verdicts.count("CHECK")
        clip_v = "NG" if n_ng else ("OK" if n_ok else "CHECK")
        clip_rows.append(dict(**base, n_reps=len(verdicts), n_ok=n_ok, n_ng=n_ng, n_check=n_ck,
                              verdict=clip_v, note=""))

    reps_df = pd.DataFrame(rep_rows)
    clips_df = pd.DataFrame(clip_rows)
    # 회차 단위 정답 라벨: 팀원이 확인(checked_by)한 행만 사용, 없으면 Clip 라벨(form_label)
    if len(reps_df):
        reps_df["rep_label"] = reps_df["form_label"]
        lab_path = ROOT / "configs" / f"{cfg['team_id']}_rep_labels.csv"
        if lab_path.exists():
            lab = pd.read_csv(lab_path, dtype=str).fillna("")
            lab = lab[lab["label"].isin(["OK", "NG"]) & (lab["checked_by"].str.strip() != "")]
            key = {(c, int(n)): l for c, n, l in zip(lab["clip_id"], lab["rep"], lab["label"])}
            reps_df["rep_label"] = [key.get((c, int(n)), f) for c, n, f
                                    in zip(reps_df["clip_id"], reps_df["rep"], reps_df["form_label"])]
            print(f"회차 라벨 적용: {len(key)}개 ({lab_path.name})")
    tag = "" if args.standards == "team" else "_glb"
    reps_df.to_csv(p["judgments"] / f"rep_judgments{tag}.csv", index=False, encoding="utf-8-sig")
    clips_df.to_csv(p["tables"] / f"judgment_summary{tag}.csv", index=False, encoding="utf-8-sig")
    print(f"[기준: {'team_08 팀 기준' if args.standards == 'team' else 'team_19 GLB 보정 기준'}]")

    if clips_df.empty:
        print("판정할 Clip이 없습니다. split 과 standards 를 확인하세요.")
        return
    print(clips_df[["clip_id", "split", "form_label", "n_reps", "n_ok", "n_ng", "n_check", "verdict", "note"]]
          .to_string(index=False))
    if len(reps_df):
        ng = reps_df[reps_df["verdict"] == "NG"]
        if len(ng):
            print("\n[NG 1회 구간 메시지]")
            print(ng[["clip_id", "rep", "out_of_range", "message"]].to_string(index=False))

    print("\n[성능 요약]")
    for split, g in clips_df.groupby("split"):
        ok = g[(g["form_label"] == "OK") & (g["verdict"] != "CHECK")]
        line = f"{split:>8}: OK 영상 {len(ok)}개 중 NG 오판정 {int((ok['verdict'] == 'NG').sum())}개"
        if len(ok):
            line += f" (false NG {np.mean(ok['verdict'] == 'NG'):.0%})"
        ngl = g[g["form_label"] == "NG"]
        if len(ngl):
            line += f" | NG 라벨 {len(ngl)}개 중 검출 {int((ngl['verdict'] == 'NG').sum())}개"
        line += f" | CHECK {int((g['verdict'] == 'CHECK').sum())}개"
        if len(reps_df):
            rr = reps_df[(reps_df["split"] == split) & (reps_df["verdict"] != "CHECK")]
            ok_r, ng_r = rr[rr["rep_label"] == "OK"], rr[rr["rep_label"] == "NG"]
            if len(ok_r):
                line += f" | 1회 단위: 정석 {len(ok_r)}회 중 오판정 {int((ok_r['verdict'] == 'NG').sum())}회"
            if len(ng_r):
                line += f", 틀린 자세 {len(ng_r)}회 중 검출 {int((ng_r['verdict'] == 'NG').sum())}회"
        print(line)


if __name__ == "__main__":
    main()
