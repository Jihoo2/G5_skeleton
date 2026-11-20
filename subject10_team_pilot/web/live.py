"""웹 실시간 판정 — WebSocket 한 연결 = LiveSession 하나 (교과 7일차 실시간 판정의 웹 버전)

입력 두 가지
  browser : 이 PC 웹캠. 브라우저가 Frame을 JPEG로 보냄 (process)
  ipcam   : 휴대폰 IP Webcam 앱(http://<휴대폰IP>:8080/video). 서버가 직접 받아 판정하고
            Frame(JPEG)과 결과를 브라우저로 보냄 (IPCamReader + ip_step) — 교과 1·7일차 IP Webcam 방식
            rotate=90/270 이면 세로로 세운 휴대폰 영상을 판정 전에 돌림
녹화      : rec_root 를 주면 판정한 Frame과 결과를 임시 녹화본으로 남김 (web/records.py, 저장 안 하면 사라짐)

판정 엔진은 team_21(터미널 실시간)·업로드 분석과 같은 OnlineJudge 입니다. 기준·메시지·경계 표시가 모두 같습니다.

시간 맞추기
  브라우저는 처리 속도에 따라 1초에 몇 장을 보낼지 매번 달라집니다(서버 응답을 받은 뒤 다음 장을 보냄).
  그런데 판정 엔진은 '1초 = fps 장'을 전제로 1회 길이(0.4초 등)와 평활화 창을 계산합니다.
  → 브라우저가 붙여 보낸 촬영 시각 t(초)에 맞춰, 엔진 시간축(ENGINE_FPS, 녹화 영상과 같은 30fps)에
    마지막 자세를 반복해 채워 넣습니다. 예) 1초에 10장이 오면 한 장을 3번씩 넣음.
  → 브라우저 전송 속도와 상관없이 녹화 영상 판정과 같은 시간 기준을 씁니다.
  2초 넘게 끊기면(탭 전환 등) 그 사이는 채우지 않고 이어서 받습니다.

받는 데이터  : [8바이트 float64 촬영 시각(ms, little-endian)] + [JPEG]
보내는 데이터: 관절 좌표, 현재 상태, 회차·OK·NG 수, 이번에 확정된 회차 판정, 처리 시간
로그        : 연결이 끝나면 reports/.../realtime_logs/web_live_<입력>_<동작>_<시각>.csv (team_21 로그와 같은 형식)
"""
from __future__ import annotations

import base64
import datetime as dt
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.realtime_judge import OnlineJudge, load_runtime_config
from web.processing import _clean, predict_pose
from web.records import Recorder

# IP Webcam 화면 방향: 휴대폰을 세로로 세우면 영상이 옆으로 누워 들어오므로 판정 전에 돌림
ROTATE = {"0": None, "90": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "270": cv2.ROTATE_90_COUNTERCLOCKWISE}

STATE_KO = {"WARMUP": "준비: 전신이 보이게 서 주세요", "STAND": "서 있음", "DOWN": "내려가는 중"}
LOG_DROP = ("checks", "curve", "template", "values")
EVENT_KEYS = ("kind", "rep", "verdict", "message", "boundary", "model_flag", "model_score",
              "start_s", "end_s", "verdict_delay_s", "values")
MAX_GAP_S = 2.0
IPCAM_SCHEMES = ("http://", "https://", "rtsp://")
IPCAM_NO_FRAME_S = 3.0   # 이 시간 동안 새 Frame이 없으면 오류
SEND_LONG = 640          # 브라우저로 보내는 영상의 긴 변 크기 (가로 640x360 / 세로 360x640, 판정은 원본 크기로)


class IPCamReader:
    """IP Webcam 스트림을 별도 스레드에서 계속 읽고 '가장 최근 Frame'만 보관.
    판정이 Frame 속도를 못 따라가도 오래된 Frame이 쌓이지 않아 화면 지연이 커지지 않음
    (OpenCV는 스트림을 버퍼에 쌓아 두므로 순서대로 읽으면 점점 늦어짐)."""

    def __init__(self, url: str):
        if not url.startswith(IPCAM_SCHEMES):
            raise ValueError("IP Webcam 주소는 http:// 로 시작해야 합니다 (예: http://192.168.0.10:8080/video)")
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError("IP Webcam에 연결할 수 없습니다 — 주소, 휴대폰 앱 '서버 시작', 같은 와이파이인지 확인하세요")
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.error: str | None = None
        self.t0 = time.time()
        self.stop_flag = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        fails = 0
        while not self.stop_flag:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 100:  # 약 5초 동안 못 읽음
                    with self.cond:
                        self.error = "IP Webcam 영상이 끊겼습니다 — 휴대폰 화면 꺼짐·와이파이를 확인하세요"
                        self.cond.notify_all()
                    break
                time.sleep(0.05)
                continue
            fails = 0
            with self.cond:
                self.frame, self.t, self.seq = f, time.time() - self.t0, self.seq + 1
                self.cond.notify_all()

    def latest(self, after_seq: int, timeout: float = IPCAM_NO_FRAME_S):
        """after_seq 보다 새 Frame이 올 때까지 기다렸다가 (Frame, 촬영 시각, 번호). 시간 초과면 None"""
        with self.cond:
            ok = self.cond.wait_for(lambda: self.seq > after_seq or self.error is not None, timeout)
            if not ok or self.error is not None or self.frame is None:
                return None
            return self.frame, self.t, self.seq

    def close(self):
        self.stop_flag = True
        self.th.join(timeout=2)
        self.cap.release()


class LiveSession:
    def __init__(self, cfg: dict, p: dict, action: str, view: str = "auto", send_fps: float = 15.0,
                 standards: str = "team", source: str = "browser", rotate: str = "0",
                 rec_root: Path | None = None, rec_meta: dict | None = None):
        self.cfg, self.p, self.action, self.view_arg, self.source = cfg, p, action, view, source
        self.rotate = ROTATE.get(str(rotate))
        # 녹화: rec_root 가 있으면 판정한 Frame과 결과를 임시 녹화본으로 기록 (저장 안 하면 사라짐)
        self.rec = Recorder(rec_root, dict(rec_meta or {}, action=action, source=source, view_arg=view,
                                           rotate=str(rotate))) if rec_root is not None else None
        self.last_seq = 0        # IP Webcam: 마지막으로 판정한 Frame 번호
        self.engine_fps = float(cfg.get("fps", 30))
        self.send_fps, self.standards = float(send_fps), standards
        self.rt = load_runtime_config()
        self.judge = OnlineJudge(cfg, action, view, p, self.engine_fps, standards, use_model=True, rt=self.rt)
        self.k = 0               # 엔진에 넣은 Frame 수
        self.n_recv = 0          # 브라우저에서 받은 장 수
        self.t_last = 0.0        # 마지막으로 받은 촬영 시각(초)
        self.offset = 0.0        # 오래 끊겼을 때 건너뛴 시간
        self.proc_ms: list[float] = []
        self.rows: list[dict] = []
        self.t0 = time.time()
        self.saved: dict | None = None

    # ------------------------------------------------------------------
    def process(self, data: bytes) -> dict:
        """browser 입력: [8바이트 촬영 시각(ms)] + [JPEG]"""
        if len(data) < 9:
            return {"error": "데이터 형식 오류"}
        t = struct.unpack("<d", data[:8])[0] / 1000.0
        img = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "이미지를 읽지 못했습니다"}
        out = self.step_image(img, t)
        if self.rec is not None:
            self.rec.add(data[8:], t, out)  # 브라우저가 보낸 JPEG 그대로 기록
        return out

    def ip_step(self, reader: IPCamReader) -> dict:
        """ipcam 입력: 가장 최근 Frame 하나를 판정하고, 브라우저 표시용 JPEG(긴 변 640)를 함께 돌려줌"""
        got = reader.latest(self.last_seq)
        if got is None:
            return {"error": reader.error or f"IP Webcam 영상이 {IPCAM_NO_FRAME_S:.0f}초 넘게 들어오지 않습니다"}
        img, t, self.last_seq = got
        if self.rotate is not None:
            img = cv2.rotate(img, self.rotate)  # 세로 촬영이면 판정 전에 바로 세움
        out = self.step_image(img, t)
        h, w = img.shape[:2]
        s = min(1.0, SEND_LONG / max(w, h))
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
        ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        out["jpg"] = base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None
        if out.get("kp"):
            out["kp"] = [[None if x is None else x * s, None if y is None else y * s, c] for x, y, c in out["kp"]]
        out["width"], out["height"] = small.shape[1], small.shape[0]
        if self.rec is not None and ok:
            self.rec.add(jpg.tobytes(), t, out)  # 브라우저에 보낸 것과 같은 Frame·좌표로 기록
        return out

    def step_image(self, img, t: float) -> dict:
        """Frame 한 장(촬영 시각 t초) → Pose → 엔진 시간축 채우기 → 확정된 회차 판정"""
        t1 = time.time()
        xy, conf = predict_pose(img)
        self.proc_ms.append((time.time() - t1) * 1000)
        self.n_recv += 1

        # 촬영 시각에 맞춰 엔진 시간축을 채움 (마지막 자세 반복)
        if self.k == 0:
            self.offset = t  # 첫 Frame 시각을 0초로 (IP Webcam은 연결하는 동안 시간이 흘러 있음)
        elif t - self.t_last > MAX_GAP_S:
            self.offset += (t - self.t_last) - 1.0 / self.engine_fps  # 끊긴 시간은 건너뜀
        self.t_last = t
        target = int((t - self.offset) * self.engine_fps) + 1
        slots = max(0, target - self.k)
        events = []
        for _ in range(slots):
            self.k += 1
            events += self.judge.push(self.k, (self.k - 1) / self.engine_fps, xy, conf)

        now_engine = (self.k - 1) / self.engine_fps
        wall = max(time.time() - self.t0, 1e-6)
        out_events = []
        for ev in events:
            ev["proc_fps"] = round(self.n_recv / wall, 2)
            ev["verdict_delay_s"] = round(now_engine - float(ev["end_s"]), 2)  # 1회 끝 → 판정 확정까지
            ev["human_label"] = ""
            row = {k: v for k, v in ev.items() if k not in LOG_DROP}
            for m, v in (ev.get("values") or {}).items():
                row[f"v_{m}"] = v
            self.rows.append(row)
            out_events.append({k: ev.get(k) for k in EVENT_KEYS})
        j = self.judge
        kp = None if not np.isfinite(xy).any() else \
            [[float(x), float(y), float(c)] for (x, y), c in zip(xy, conf)]
        return _clean(dict(
            kp=kp, width=int(img.shape[1]), height=int(img.shape[0]),
            state=STATE_KO.get(j.state, j.state), view=j.view, rep_no=j.rep_no, counts=dict(j.counts),
            events=out_events, proc_ms=round(self.proc_ms[-1], 1),
            recv_fps=round(self.n_recv / wall, 1), engine_fps=self.engine_fps,
        ))

    def label(self, value: str) -> dict | None:
        """마지막 판정 회차에 사람이 본 정답(OK/NG) 기록 — 다시 누르면 덮어씀"""
        if value not in ("OK", "NG") or not self.rows:
            return None
        self.rows[-1]["human_label"] = value
        return {"rep": self.rows[-1].get("rep"), "human_label": value}

    # ------------------------------------------------------------------
    def finish(self) -> dict:
        """연결 종료: 로그 저장 + 요약 (여러 번 불러도 한 번만 저장)"""
        if self.saved is not None:
            return self.saved
        df = pd.DataFrame(self.rows)
        summary = dict(reps=self.judge.rep_no, **self.judge.counts, frames_received=self.n_recv,
                       seconds=round(time.time() - self.t0, 1),
                       recv_fps=round(self.n_recv / max(time.time() - self.t0, 1e-6), 1),
                       pose_ms_mean=round(float(np.mean(self.proc_ms)), 1) if self.proc_ms else None,
                       verdict_delay_mean=None, labeled=0, label_match=None, log=None, record=None)
        if self.rec is not None:
            summary["record"] = self.rec.close(summary)  # Frame이 없으면 None
        if len(df):
            summary["verdict_delay_mean"] = round(float(df["verdict_delay_s"].mean()), 2)
            lab = df[df["human_label"].isin(["OK", "NG"]) & df["verdict"].isin(["OK", "NG"])]
            summary["labeled"] = int(len(lab))
            if len(lab):
                summary["label_match"] = round(float((lab["human_label"] == lab["verdict"]).mean()), 3)
            out_dir = self.p["reports"] / "realtime_logs"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"web_live_{self.source}_{self.action}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
            meta = dict(source_type=self.source, action_id=self.action, view_arg=self.view_arg,
                        standards=self.standards, engine_fps=self.engine_fps, send_fps=self.send_fps,
                        **{f"rt_{k}": v for k, v in self.rt.items()})
            for k, v in meta.items():
                df[k] = v
            df.to_csv(path, index=False, encoding="utf-8-sig")
            try:
                summary["log"] = str(Path(path).relative_to(self.p["reports"].parents[2]))
            except ValueError:
                summary["log"] = str(path)
        self.saved = _clean(summary)
        return self.saved
