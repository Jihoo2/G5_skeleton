"""업로드 영상 분석 — YOLO Pose → OnlineJudge(실시간 엔진과 같은 판정) → result.json

결과 JSON (브라우저가 원본 영상 위에 캔버스로 뼈대를 그림)
  meta    : 동작·각도·fps·해상도·Frame 수
  frames  : Frame마다 17관절 [x, y, conf] (사람이 없으면 null)
  events  : 1회마다 판정, 메시지, 측정값 vs 기준, 무릎·고관절 곡선 vs GLB 정석 곡선, 모델 주의
  summary : 회차 수, OK·NG·CHECK 수
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_judge import OnlineJudge  # noqa: E402

_MODEL = None


def _yolo():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        from src.common import load_config, resolve_device
        pc = load_config()
        _MODEL = (YOLO(pc["pose_model"]), pc, resolve_device(pc["device"]))
    return _MODEL


def _clean(o):
    """numpy·NaN → JSON"""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def analyze_video(video: Path, action: str, view: str, cfg: dict, p: dict, out_dir: Path,
                  standards: str = "team", progress=lambda x: None) -> dict:
    model, pc, device = _yolo()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 5 < fps < 121 else 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    judge = OnlineJudge(cfg, action, view, p, fps, standards, use_model=True)
    frames, events, n, t0 = [], [], 0, time.time()
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
            frames.append([[round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                           for (x, y), c in zip(xy, conf)])
        else:
            frames.append(None)
        events += judge.push(n, (n - 1) / fps, xy, conf)
        if n % 15 == 0:
            progress(min(0.99, n / n_total))
    cap.release()
    events += judge.flush()
    result = _clean(dict(
        meta=dict(action=action, action_label=cfg["actions"][action]["label"], view=judge.view,
                  view_requested=view, fps=fps, width=w, height=h, n_frames=n,
                  standards=standards, seconds=round(time.time() - t0, 1)),
        frames=frames, events=events,
        summary=dict(reps=judge.rep_no, **judge.counts,
                     shallow=sum(1 for e in events if e.get("kind") == "shallow"),
                     attention=sum(1 for e in events if e.get("model_flag"))),
    ))
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    progress(1.0)
    return result
