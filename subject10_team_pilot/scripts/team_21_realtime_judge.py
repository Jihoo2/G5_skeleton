"""
[팀 21] 실시간 자세 판정 — 웹캠 / IP Webcam / 영상 파일 (포트폴리오 시연, 교과 7일차 자리)

화면: 뼈대, 현재 상태(대기·서 있음·내려감), 회차 수, 마지막 회차 판정(OK 초록 / NG 빨강 + 교정 메시지 /
      CHECK 회색), 모델 '주의'(노랑), 누적 OK·NG
판정 기준: team_09와 같은 standards·규칙·메시지 (src/realtime_judge.py)
  - 시작 후 1초 동안 준비 → 카메라 앞에 전신이 보이게 서 있다가 시작
  - 1회가 끝나고 약 0.6초 뒤에 판정이 뜹니다 (다음 동작과 합쳐지지 않는지 확인하는 시간)
  - 촬영 각도는 --view 로 지정하거나 auto(어깨 폭/몸통 길이로 추정)

실행
  # 노트북 웹캠
  python scripts/team_21_realtime_judge.py --action A008 --source 0
  # 휴대폰 IP Webcam (같은 와이파이)
  python scripts/team_21_realtime_judge.py --action A004 --view side --source http://192.168.0.10:8080/video
  # 영상 파일로 시연 영상 만들기 (화면 없이 저장만)
  python scripts/team_21_realtime_judge.py --action A008 --view front \\
      --source data/team_projects/team05/ext/videos/EXT_S05_A008_T01_front.mp4 --save demo.mp4 --no-show

검증 (YOLO·영상 없이, team_03 Keypoint를 한 Frame씩 넣어 team_09 녹화 판정과 비교)
  python scripts/team_21_realtime_judge.py --replay EXT_S05_A008_T01_front
  python scripts/team_21_realtime_judge.py --replay-all        # val·val_ext 전체 일치율
  → tables/realtime_replay_check.csv  (영상별 회차 수·일치 수)
    tables/realtime_replay_detail.csv (회차별 녹화 vs 실시간 판정·depth·경계 — 불일치 원인 확인용)

실시간 로그 (교과 7일차 Runtime Log 자리) — 실행할 때마다 reports/.../realtime_logs/ 에 회차별 CSV 저장
  회차 판정이 뜬 뒤 키보드로 사람이 본 정답을 기록:  o = OK,  n = NG,  q = 종료
  → 로그의 human_label 과 verdict 를 비교하면 '실시간 일치율'(Runtime Match)을 낼 수 있음
    ※ 이 값은 test_ext 최종 평가(team_09 --final)와 다른 지표입니다. 섞어서 쓰지 않습니다.
  proc_fps        : 그 회차가 끝날 때까지의 전체 처리 속도 (입력 읽기 + YOLO + 판정 + 그리기)
  verdict_delay_s : 카메라 입력일 때만 — 1회가 끝난 시점부터 판정이 나온 시점까지 걸린 시간
설정: configs/team_realtime.json (없으면 src/realtime_judge.py 의 기본값)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_judge import OnlineJudge, load_runtime_config  # noqa: E402
from src.team_feature_utils import load_keypoints_csv, load_team_config, team_paths  # noqa: E402

EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
BGR = {"OK": (60, 200, 60), "NG": (40, 40, 230), "CHECK": (160, 160, 160), "주의": (0, 210, 255)}
STATE_KO = {"WARMUP": "준비: 전신이 보이게 서 주세요", "STAND": "서 있음", "DOWN": "내려가는 중"}


# ----------------------------------------------------------------------
# 그리기 (한글은 PIL + 맑은 고딕, 없으면 영어)
# ----------------------------------------------------------------------
class Painter:
    def __init__(self):
        self.font = None
        try:
            from PIL import ImageFont
            for fp in ["/mnt/c/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
                if Path(fp).exists():
                    self.font = {s: ImageFont.truetype(fp, s) for s in (22, 30, 40)}
                    break
        except ImportError:
            pass

    def text(self, img, lines: list[tuple[str, tuple, int]], org=(16, 16)):
        """lines: (문자열, BGR, 크기) 를 위에서부터 차례로"""
        if self.font is None:
            y = org[1] + 24
            for s, c, size in lines:
                cv2.putText(img, s.encode("ascii", "ignore").decode() or "-", (org[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32, c, 2)
                y += size + 10
            return img
        from PIL import Image, ImageDraw
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dr = ImageDraw.Draw(pil)
        y = org[1]
        for s, c, size in lines:
            f = self.font[min(self.font, key=lambda k: abs(k - size))]
            dr.text((org[0] + 2, y + 2), s, font=f, fill=(0, 0, 0))
            dr.text((org[0], y), s, font=f, fill=(c[2], c[1], c[0]))
            y += size + 10
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_skeleton(img, xy, conf, kp_conf):
    for a, b in EDGES:
        if np.isfinite(xy[[a, b]]).all() and conf[a] >= kp_conf and conf[b] >= kp_conf:
            cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), (255, 255, 255), 3)
    for k in range(17):
        if np.isfinite(xy[k]).all():
            c = (60, 200, 60) if conf[k] >= kp_conf else (0, 220, 255) if conf[k] >= 0.3 else (40, 40, 230)
            cv2.circle(img, tuple(map(int, xy[k])), 5, c, -1)


def overlay(img, judge: OnlineJudge, last: dict | None, painter: Painter, label: str):
    lines = [(f"{label}  |  각도 {judge.view or '추정 중'}  |  {STATE_KO.get(judge.state, judge.state)}",
              (255, 255, 255), 22),
             (f"회차 {judge.rep_no}   OK {judge.counts['OK']}   NG {judge.counts['NG']}", (255, 255, 255), 30)]
    if last:
        c = BGR.get(last["verdict"], (255, 255, 255))
        lines.append((f"{last['rep']}회: {last['verdict']}", c, 40))
        if last.get("message"):
            lines.append((last["message"], c, 30))
        if last.get("boundary"):
            lines.append(("경계: 기준선 근처라 판정이 바뀔 수 있음 — 한 번 더 확인", BGR["주의"], 22))
        if last.get("human_label"):
            lines.append((f"사람 정답 기록: {last['human_label']}", (255, 255, 255), 22))
        if last.get("model_flag"):
            lines.append((f"주의: 정석과 다른 움직임 (점수 {last['model_score']:.1f})", BGR["주의"], 22))
    return painter.text(img, lines)


# ----------------------------------------------------------------------
# 실시간 로그
# ----------------------------------------------------------------------
LOG_DROP = ("checks", "curve", "template", "values")


def log_row(ev: dict) -> dict:
    """판정 이벤트 → 로그 1행 (곡선 등 큰 값은 빼고 측정값은 v_<이름> 컬럼으로 펼침)"""
    row = {k: v for k, v in ev.items() if k not in LOG_DROP}
    for m, v in (ev.get("values") or {}).items():
        row[f"v_{m}"] = v
    return row


def save_log(rows: list[dict], p: dict, name: str, meta: dict) -> Path | None:
    if not rows:
        return None
    out_dir = p["reports"] / "realtime_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def summarize_log(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "proc_fps" in df and df["proc_fps"].notna().any():
        print(f"처리 속도: 평균 {df['proc_fps'].mean():.1f} FPS")
    if "verdict_delay_s" in df and df["verdict_delay_s"].notna().any():
        print(f"판정 지연(1회 끝 → 판정 표시): 평균 {df['verdict_delay_s'].mean():.2f}초, "
              f"최대 {df['verdict_delay_s'].max():.2f}초")
    if "human_label" in df:
        lab = df[df["human_label"].isin(["OK", "NG"]) & df["verdict"].isin(["OK", "NG"])]
        if len(lab):
            same = int((lab["human_label"] == lab["verdict"]).sum())
            print(f"실시간 일치(사람 정답 기록 {len(lab)}회 중): {same}회 ({same / len(lab):.0%})"
                  "  ※ Runtime 확인용 — test_ext 최종 평가와 다른 지표")


# ----------------------------------------------------------------------
# 실시간 / 영상 파일
# ----------------------------------------------------------------------
def run_live(args, cfg, p):
    from ultralytics import YOLO
    from src.common import load_config, resolve_device
    pc = load_config()
    model, device = YOLO(pc["pose_model"]), resolve_device(pc["device"])
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"[중지] 입력을 열 수 없습니다: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 5 < fps < 121 else 30.0
    rt = load_runtime_config()
    judge = OnlineJudge(cfg, args.action, args.view, p, fps, args.standards, not args.no_model, rt=rt)
    painter, writer, last, n = Painter(), None, None, 0
    label = f"{args.action} {cfg['actions'][args.action]['label']}"
    is_file = not str(args.source).isdigit() and not str(args.source).startswith("http")
    rows: list[dict] = []
    t0 = time.time()

    def record(ev: dict) -> None:
        now = time.time() - t0
        ev["wall_s"] = round(now, 2)
        ev["proc_fps"] = round(n / now, 2) if now > 0 else None
        # 카메라 입력은 t가 실제 시간이라 지연을 잴 수 있음 (영상 파일은 영상 시간이라 의미 없음)
        ev["verdict_delay_s"] = None if is_file else round(now - float(ev["end_s"]), 2)
        ev["human_label"] = ""
        rows.append(log_row(ev))
        print(f"[{ev['start_s']:6.1f}s] {ev['rep']}회 {ev['verdict']:5s} {ev.get('message', '')}"
              + (f"  (경계 {ev['boundary']})" if ev.get("boundary") else "")
              + (f"  (주의 {ev['model_score']})" if ev.get("model_flag") else ""))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        r = model.predict(source=frame, imgsz=pc["image_size"], conf=pc["person_confidence"],
                          device=device, verbose=False)[0]
        xy, conf = np.full((17, 2), np.nan), np.zeros(17)
        if r.boxes is not None and len(r.boxes) and r.keypoints is not None and r.keypoints.conf is not None:
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            xy, conf = r.keypoints.xy.cpu().numpy()[i], r.keypoints.conf.cpu().numpy()[i]
        t = (n - 1) / fps if is_file else time.time() - t0
        for ev in judge.push(n, t, xy, conf):
            record(ev)
            last = ev
        draw_skeleton(frame, xy, conf, judge.kp_conf)
        frame = overlay(frame, judge, last, painter, label)
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("team05 form judge (o: OK, n: NG, q: 종료)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("o"), ord("n")) and rows:
                # 마지막 판정 회차에 사람이 본 정답 기록 (다시 누르면 덮어씀)
                rows[-1]["human_label"] = "OK" if key == ord("o") else "NG"
                if last is not None:
                    last["human_label"] = rows[-1]["human_label"]
                print(f"  → {rows[-1]['rep']}회 사람 정답: {rows[-1]['human_label']}")
    for ev in (judge.flush() if is_file else []):
        record(ev)
    cap.release()
    if writer:
        writer.release()
        print(f"저장: {args.save}")
    cv2.destroyAllWindows()
    print(f"\n회차 {judge.rep_no}, OK {judge.counts['OK']}, NG {judge.counts['NG']}, CHECK {judge.counts['CHECK']}")
    summarize_log(rows)
    src_name = "file" if is_file else ("ipwebcam" if str(args.source).startswith("http") else "webcam")
    path = save_log(rows, p, f"live_{args.action}_{src_name}",
                    dict(source=str(args.source), source_type=src_name, action_id=args.action,
                         view_arg=args.view, standards=args.standards, input_fps=fps,
                         **{f"rt_{k}": v for k, v in rt.items()}))
    if path:
        print(f"로그: {path}")


# ----------------------------------------------------------------------
# 검증: team_03 Keypoint를 한 Frame씩 → team_09 녹화 판정과 비교
# ----------------------------------------------------------------------
def _num(x):
    try:
        x = float(x)
        return round(x, 4) if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _txt(x) -> str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else str(x)


def replay(clip_id: str, cfg, p, man, offline: pd.DataFrame, standards: str, rt: dict,
           verbose=True) -> tuple[dict, list[dict]]:
    r = man[man["clip_id"] == clip_id].iloc[0]
    fid, t, xy, conf = load_keypoints_csv(p["keypoints"] / f"{clip_id}_keypoints.csv")
    ok_t = t[np.isfinite(t)]
    fps = round((len(ok_t) - 1) / (ok_t[-1] - ok_t[0]), 2) if len(ok_t) > 1 else 30.0
    judge = OnlineJudge(cfg, r["action_id"], r["view"], p, fps, standards, use_model=True, rt=rt)
    ev = []
    for i in range(len(fid)):
        ev += judge.push(int(fid[i]), float(i / fps), xy[i], conf[i])
    ev += judge.flush()
    on = pd.DataFrame(ev)
    off = offline[offline["clip_id"] == clip_id].copy()
    # 녹화 판정 회차마다 겹치는 실시간 회차를 찾아 판정 비교
    match, bnd_mis, rows, detail, used = 0, 0, [], [], set()
    for _, o in off.iterrows():
        cand = on[(on["start_frame"] <= o["end_frame"]) & (on["end_frame"] >= o["start_frame"])] if len(on) else on
        c = cand.iloc[0] if len(cand) else None
        if c is not None:
            used.add(cand.index[0])
        v_on = c["verdict"] if c is not None else "(없음)"
        same = v_on == o["verdict"]
        b_off = _txt(o.get("boundary"))
        b_on = _txt(c.get("boundary")) if c is not None else ""
        # 둘 중 한쪽이라도 기준 경계였던 불일치는 '경계 불일치'로 따로 셈
        is_bnd = (not same) and c is not None and bool(b_off or b_on)
        match += same
        bnd_mis += is_bnd
        note = "" if same else ("  ← 다름(기준 경계)" if is_bnd else "  ← 다름")
        rows.append((o["rep"], o["verdict"], v_on, note))
        detail.append(dict(clip_id=clip_id, kind="matched" if c is not None else "offline_only",
                           rep_offline=o["rep"], rep_online=c["rep"] if c is not None else "",
                           verdict_offline=o["verdict"], verdict_online=v_on, same=bool(same),
                           boundary_mismatch=bool(is_bnd),
                           start_offline=o.get("start_frame"), end_offline=o.get("end_frame"),
                           start_online=c["start_frame"] if c is not None else None,
                           end_online=c["end_frame"] if c is not None else None,
                           depth_offline=_num(o.get("depth")), depth_online=_num(c.get("depth")) if c is not None else None,
                           knee_min_offline=_num(o.get("knee_min")),
                           knee_min_online=_num(c.get("knee_min")) if c is not None else None,
                           boundary_offline=b_off, boundary_online=b_on,
                           out_of_range_offline=_txt(o.get("out_of_range")),
                           out_of_range_online=_txt(c.get("out_of_range")) if c is not None else ""))
    # 녹화 판정과 겹치지 않는 실시간 회차 (= 실시간에만 있는 회차)
    for k, c in (on.iterrows() if len(on) else []):
        if k in used:
            continue
        detail.append(dict(clip_id=clip_id, kind="online_only", rep_offline="", rep_online=c["rep"],
                           verdict_offline="(없음)", verdict_online=c["verdict"], same=False, boundary_mismatch=False,
                           start_offline=None, end_offline=None, start_online=c["start_frame"],
                           end_online=c["end_frame"], depth_offline=None, depth_online=_num(c.get("depth")),
                           knee_min_offline=None, knee_min_online=_num(c.get("knee_min")),
                           boundary_offline="", boundary_online=_txt(c.get("boundary")),
                           out_of_range_offline="", out_of_range_online=_txt(c.get("out_of_range"))))
    extra = sum(1 for d in detail if d["kind"] == "online_only")
    if verbose:
        print(f"\n[{clip_id}] fps={fps}  녹화 {len(off)}회 / 실시간 {len(on)}회, 판정 일치 {match}/{len(off)}"
              + (f", 실시간에만 있는 회차 {extra}" if extra > 0 else ""))
        for rep, a, b, note in rows:
            print(f"  {str(rep):>5}: 녹화 {a:5s}  실시간 {b:6s}{note}")
        for d in detail:
            if d["kind"] == "online_only":
                print(f"  실시간에만: {d['rep_online']}회 {d['verdict_online']}  "
                      f"Frame {d['start_online']}~{d['end_online']}  depth {d['depth_online']}")
    return (dict(clip_id=clip_id, offline=len(off), online=len(on), match=match, extra=extra,
                 boundary_mismatch=bnd_mis), detail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["A001", "A004", "A008"])
    ap.add_argument("--view", default="auto", choices=["auto", "front", "diag", "side"])
    ap.add_argument("--source", default="0")
    ap.add_argument("--standards", default="team", choices=["team", "glb"])
    ap.add_argument("--save")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--replay")
    ap.add_argument("--replay-all", action="store_true")
    args = ap.parse_args()

    cfg = load_team_config()
    p = team_paths(cfg)
    if args.replay or args.replay_all:
        man = pd.read_csv(p["manifest"], dtype=str).fillna("")
        rj = p["judgments"] / "rep_judgments.csv"
        if not rj.exists():
            raise SystemExit("[중지] rep_judgments.csv 없음 → python scripts/team_09_judge.py 먼저")
        offline = pd.read_csv(rj)
        if "boundary" not in offline:
            print("[참고] rep_judgments.csv 에 boundary 컬럼이 없습니다 → team_09_judge.py 를 다시 실행하면 "
                  "녹화 판정 쪽 '경계'도 비교합니다.")
        rt = load_runtime_config()
        print("실시간 설정: " + ", ".join(f"{k}={v}" for k, v in rt.items()))
        clips = [args.replay] if args.replay else sorted(offline["clip_id"].unique())
        out = [replay(c, cfg, p, man, offline, args.standards, rt, verbose=True) for c in clips]
        res = [o[0] for o in out]
        detail = [d for o in out for d in o[1]]
        df = pd.DataFrame(res)
        mis = int(df.offline.sum() - df.match.sum())
        hard = mis - int(df.boundary_mismatch.sum())
        print(f"\n[전체] 녹화 회차 {df.offline.sum()}개 중 실시간 판정 일치 {df.match.sum()}개 "
              f"({df.match.sum() / max(df.offline.sum(), 1):.0%}), 실시간에만 있는 회차 {df.extra.sum()}개")
        print(f"       불일치 {mis}개 = 기준 경계 회차 {int(df.boundary_mismatch.sum())}개 + 그 밖의 불일치 {hard}개")
        if args.replay_all:
            df.to_csv(p["tables"] / "realtime_replay_check.csv", index=False, encoding="utf-8-sig")
        name = "realtime_replay_detail.csv" if args.replay_all else f"realtime_replay_detail_{args.replay}.csv"
        pd.DataFrame(detail).to_csv(p["tables"] / name, index=False, encoding="utf-8-sig")
        print(f"회차별 비교: {p['tables'] / name}")
        return
    if not args.action:
        raise SystemExit("--action A001|A004|A008 을 지정하세요")
    run_live(args, cfg, p)


if __name__ == "__main__":
    main()
