"""
[팀 21] 실시간 자세 판정 — 웹캠 / IP Webcam / 영상 파일 (포트폴리오 시연, 교과 7일차 자리)

화면: 뼈대, 현재 상태(대기·서 있음·내려감), 회차 수, 마지막 회차 판정(OK 초록 / NG 빨강 + 교정 메시지 /
      CHECK 회색), 모델 '주의'(노랑), 누적 OK·NG
판정 기준: team_09와 같은 standards·규칙·메시지 (src/realtime_judge.py)
  - 시작 후 1.5초는 '서 있는 자세'로 기준을 잡습니다 → 카메라 앞에 전신이 보이게 서 있다가 시작
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
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_judge import OnlineJudge  # noqa: E402
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
        if last.get("model_flag"):
            lines.append((f"주의: 정석과 다른 움직임 (점수 {last['model_score']:.1f})", BGR["주의"], 22))
    return painter.text(img, lines)


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
    judge = OnlineJudge(cfg, args.action, args.view, p, fps, args.standards, not args.no_model)
    painter, writer, last, n = Painter(), None, None, 0
    label = f"{args.action} {cfg['actions'][args.action]['label']}"
    t0 = time.time()
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
        t = (n - 1) / fps if not str(args.source).isdigit() and not str(args.source).startswith("http") \
            else time.time() - t0
        for ev in judge.push(n, t, xy, conf):
            last = ev
            print(f"[{ev['start_s']:6.1f}s] {ev['rep']}회 {ev['verdict']:5s} {ev.get('message', '')}"
                  + (f"  (주의 {ev['model_score']})" if ev.get("model_flag") else ""))
        draw_skeleton(frame, xy, conf, judge.kp_conf)
        frame = overlay(frame, judge, last, painter, label)
        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("team05 form judge (q: 종료)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
        print(f"저장: {args.save}")
    cv2.destroyAllWindows()
    print(f"\n회차 {judge.rep_no}, OK {judge.counts['OK']}, NG {judge.counts['NG']}, CHECK {judge.counts['CHECK']}")


# ----------------------------------------------------------------------
# 검증: team_03 Keypoint를 한 Frame씩 → team_09 녹화 판정과 비교
# ----------------------------------------------------------------------
def replay(clip_id: str, cfg, p, man, offline: pd.DataFrame, standards: str, verbose=True) -> dict:
    r = man[man["clip_id"] == clip_id].iloc[0]
    fid, t, xy, conf = load_keypoints_csv(p["keypoints"] / f"{clip_id}_keypoints.csv")
    ok_t = t[np.isfinite(t)]
    fps = round((len(ok_t) - 1) / (ok_t[-1] - ok_t[0]), 2) if len(ok_t) > 1 else 30.0
    judge = OnlineJudge(cfg, r["action_id"], r["view"], p, fps, standards, use_model=True)
    ev = []
    for i in range(len(fid)):
        ev += judge.push(int(fid[i]), float(i / fps), xy[i], conf[i])
    on = pd.DataFrame(ev)
    off = offline[offline["clip_id"] == clip_id].copy()
    # 녹화 판정 회차마다 겹치는 실시간 회차를 찾아 판정 비교
    match, rows = 0, []
    for _, o in off.iterrows():
        cand = on[(on["start_frame"] <= o["end_frame"]) & (on["end_frame"] >= o["start_frame"])] if len(on) else on
        v_on = cand.iloc[0]["verdict"] if len(cand) else "(없음)"
        same = v_on == o["verdict"]
        match += same
        rows.append((o["rep"], o["verdict"], v_on, "" if same else "  ← 다름"))
    extra = len(on) - sum(1 for *_, v, _ in rows if v != "(없음)") if len(on) else 0
    if verbose:
        print(f"\n[{clip_id}] fps={fps}  녹화 {len(off)}회 / 실시간 {len(on)}회, 판정 일치 {match}/{len(off)}"
              + (f", 실시간에만 있는 회차 {extra}" if extra > 0 else ""))
        for rep, a, b, note in rows:
            print(f"  {str(rep):>5}: 녹화 {a:5s}  실시간 {b:6s}{note}")
    return dict(clip_id=clip_id, offline=len(off), online=len(on), match=match, extra=max(extra, 0))


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
        clips = [args.replay] if args.replay else sorted(offline["clip_id"].unique())
        res = [replay(c, cfg, p, man, offline, args.standards, verbose=bool(args.replay) or True) for c in clips]
        if args.replay_all:
            df = pd.DataFrame(res)
            print(f"\n[전체] 녹화 회차 {df.offline.sum()}개 중 실시간 판정 일치 {df.match.sum()}개 "
                  f"({df.match.sum() / max(df.offline.sum(), 1):.0%}), 실시간에만 있는 회차 {df.extra.sum()}개")
            df.to_csv(p["tables"] / "realtime_replay_check.csv", index=False, encoding="utf-8-sig")
        return
    if not args.action:
        raise SystemExit("--action A001|A004|A008 을 지정하세요")
    run_live(args, cfg, p)


if __name__ == "__main__":
    main()
