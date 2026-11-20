"""
팀05 맨몸운동 자세 판정 웹 (포트폴리오 시연)

실행 (subject10_team_pilot 폴더에서)
  uv add fastapi "uvicorn[standard]" python-multipart     # 처음 한 번
  uvicorn web.app:app --host 0.0.0.0 --port 8000
  → 브라우저 http://localhost:8000

기능
  1단계 업로드 분석 ( / )
    - 영상 업로드 + 동작·각도 선택 → 백그라운드 분석(YOLO Pose + 실시간 엔진과 같은 판정)
    - 결과 페이지: 원본 영상 위 뼈대(캔버스), 회차 타임라인, 회차별 측정값 vs 기준, 무릎·고관절 곡선 vs GLB 정석
  3단계 웹캠 실시간 ( /live , WebSocket /ws/live ) — 교과 7일차 실시간 판정의 웹 버전
    - 브라우저 카메라 → JPEG를 WebSocket으로 전송 → 같은 OnlineJudge → 1회마다 OK/NG·교정 메시지
    - 판정 뒤 o/n 키(또는 버튼)로 사람이 본 정답 기록 → 끝나면 realtime_logs/web_live_*.csv
    - 입력 두 가지: 이 PC 웹캠(브라우저가 Frame 전송) / 휴대폰 IP Webcam(서버가 주소에서 직접 받음)
    - 이 PC 웹캠은 http://localhost:8000 처럼 '이 PC'에서 열어야 허용됨(브라우저 보안 규칙)
    - IP Webcam: 휴대폰 앱 'IP Webcam' → 서버 시작 → http://<휴대폰IP>:8080/video (PC와 같은 와이파이)
      휴대폰을 세로로 세우면 '화면 방향'을 세로로 → 서버가 판정 전에 영상을 돌림
  녹화 ( /records , /records/<id> )
    - 실시간 판정 중 Frame과 판정 결과를 임시 녹화 → 끝난 뒤 바로 재생(뼈대·회차·OK/NG·메시지를 크게 표시)
    - '저장'을 누른 녹화본만 남고, 저장하지 않은 것은 다음 실시간 판정 시작·서버 재시작 때 지워짐
    - WebSocket에는 uvicorn[standard](websockets 패키지)가 필요
업로드 영상은 reports/team_projects/<team>/web_jobs/<id>/ 에만 저장됩니다 (Git 제외). 외부 공개 배포 금지.
실시간 영상은 '녹화하기'를 켰을 때만 임시 녹화하고, '저장'을 누른 것만 live_records/ 에 남깁니다 (Git 제외).
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_team_config, team_paths  # noqa: E402
from web import records  # noqa: E402
from web.live import ROTATE, IPCamReader, LiveSession  # noqa: E402
from web.processing import analyze_video  # noqa: E402

CFG = load_team_config()
P = team_paths(CFG)
JOBS_DIR = P["reports"] / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
REC_ROOT = P["reports"] / "live_records"
REC_ROOT.mkdir(parents=True, exist_ok=True)
records.cleanup_temp(REC_ROOT)  # 서버를 다시 켜면 저장하지 않은 녹화본은 지움
STATIC = Path(__file__).parent / "static"
ACTIONS = {aid: a["label"] for aid, a in CFG["actions"].items() if aid in ("A001", "A004", "A008")}
ACTION_KO = {"A001": "버피", "A004": "런지", "A008": "스쿼트 점프"}
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="team05 form judge")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
pool = ThreadPoolExecutor(max_workers=1)  # YOLO는 한 번에 하나씩
lock = threading.Lock()
status: dict[str, dict] = {}


def _status_path(job: str) -> Path:
    return JOBS_DIR / job / "status.json"


def _set(job: str, **kw):
    with lock:
        st = status.setdefault(job, {})
        st.update(kw)
        _status_path(job).write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def _run(job: str, video: Path, action: str, view: str, standards: str):
    _set(job, state="running", progress=0.0)
    try:
        analyze_video(video, action, view, CFG, P, JOBS_DIR / job, standards,
                      progress=lambda x: _set(job, progress=round(x, 3)))
        _set(job, state="done", progress=1.0)
    except Exception as e:  # 결과 페이지에 오류 표시
        traceback.print_exc()
        _set(job, state="error", error=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/actions")
def actions():
    return [{"id": k, "label": v, "ko": ACTION_KO.get(k, v)} for k, v in ACTIONS.items()]


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), action: str = Form(...), view: str = Form("auto"),
                     standards: str = Form("team")):
    if action not in ACTIONS:
        raise HTTPException(400, "동작을 선택하세요")
    if view not in ("auto", "front", "diag", "side"):
        raise HTTPException(400, "각도 값 오류")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"지원하지 않는 형식: {ext}")
    job = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job
    d.mkdir(parents=True)
    video = d / f"input{ext}"
    with video.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    (d / "request.json").write_text(json.dumps(dict(action=action, view=view, standards=standards,
                                                    filename=file.filename), ensure_ascii=False),
                                    encoding="utf-8")
    _set(job, state="queued", progress=0.0, video=video.name)
    pool.submit(_run, job, video, action, view, standards)
    return {"job": job}


@app.get("/api/jobs/{job}")
def job_status(job: str):
    if job in status:
        return status[job]
    sp = _status_path(job)
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    raise HTTPException(404, "없는 작업")


@app.get("/api/jobs/{job}/result")
def job_result(job: str):
    rp = JOBS_DIR / job / "result.json"
    if not rp.exists():
        raise HTTPException(404, "아직 결과가 없습니다")
    return JSONResponse(json.loads(rp.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job}/video")
def job_video(job: str):
    d = JOBS_DIR / job
    vids = [v for v in d.glob("input.*")] if d.exists() else []
    if not vids:
        raise HTTPException(404, "영상 없음")
    return FileResponse(vids[0])


@app.get("/jobs/{job}", response_class=HTMLResponse)
def job_page(job: str):
    if not (JOBS_DIR / job).exists():
        raise HTTPException(404, "없는 작업")
    return (STATIC / "job.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# 3단계: 웹캠 실시간 판정
# ----------------------------------------------------------------------
@app.get("/live", response_class=HTMLResponse)
def live_page():
    return (STATIC / "live.html").read_text(encoding="utf-8")


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket, action: str = "", view: str = "auto", fps: float = 15.0,
                  standards: str = "team", source: str = "browser", url: str = "",
                  rotate: str = "0", record: int = 1):
    """source=browser: 브라우저가 Frame을 보내면 판정 결과를 돌려줌
    source=ipcam  : 서버가 IP Webcam(url)에서 Frame을 받아 판정하고 Frame(JPEG)+결과를 계속 보냄
    rotate        : IP Webcam 화면 방향 (0 / 90 / 180 / 270)
    record=1      : 임시 녹화 (끝난 뒤 summary.record 의 ID로 재생·저장)"""
    await ws.accept()
    if action not in ACTIONS or view not in ("auto", "front", "diag", "side") or not 5 <= fps <= 30 \
            or standards not in ("team", "glb") or source not in ("browser", "ipcam") or rotate not in ROTATE:
        await ws.send_json({"error": "동작·각도·입력·전송 속도·화면 방향 값을 확인하세요"})
        await ws.close()
        return
    if record:
        await run_in_threadpool(records.cleanup_temp, REC_ROOT)  # 새로 시작하면 지난번 저장 안 한 녹화본은 지움
    sess, reader, producer = None, None, None
    send_lock = asyncio.Lock()  # 영상 전송 작업과 명령 응답이 동시에 보내지 않도록

    async def send(obj):
        async with send_lock:
            await ws.send_json(obj)

    async def ipcam_loop():
        while True:
            out = await run_in_threadpool(sess.ip_step, reader)
            await send(out)
            if out.get("error"):
                return

    try:
        sess = await run_in_threadpool(
            LiveSession, CFG, P, action, view, fps, standards, source, rotate,
            REC_ROOT if record else None,
            dict(action_ko=ACTION_KO.get(action, action), standards=standards))
        if source == "ipcam":
            reader = await run_in_threadpool(IPCamReader, url.strip())
        await send({"ready": True, "source": source, "engine_fps": sess.engine_fps})
        if reader is not None:
            producer = asyncio.create_task(ipcam_loop())
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes") is not None and source == "browser":
                await send(await run_in_threadpool(sess.process, msg["bytes"]))
            elif msg.get("text"):
                cmd = json.loads(msg["text"])
                if cmd.get("label"):
                    await send({"labeled": sess.label(cmd["label"])})
                elif cmd.get("stop"):
                    if producer is not None:
                        producer.cancel()
                    await send({"summary": await run_in_threadpool(sess.finish)})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # 화면에 오류 표시 (IP Webcam 연결 실패 등)
        traceback.print_exc()
        try:
            await send({"error": str(e)})
        except Exception:
            pass
    finally:
        if producer is not None:
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):
                pass
        if reader is not None:
            await run_in_threadpool(reader.close)
        if sess is not None:
            await run_in_threadpool(sess.finish)  # 브라우저를 그냥 닫아도 로그 저장·녹화 마무리


# ----------------------------------------------------------------------
# 녹화본: 목록·재생·저장·삭제
# ----------------------------------------------------------------------
@app.get("/records", response_class=HTMLResponse)
def records_page():
    return (STATIC / "records.html").read_text(encoding="utf-8")


@app.get("/records/{rec_id}", response_class=HTMLResponse)
def record_page(rec_id: str):
    return (STATIC / "record.html").read_text(encoding="utf-8")


@app.get("/api/records")
def api_records():
    return records.list_records(REC_ROOT, "saved")


@app.get("/api/records/{rec_id}")
def api_record(rec_id: str):
    try:
        return JSONResponse(records.timeline(REC_ROOT, rec_id))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.get("/api/records/{rec_id}/frame/{n}")
def api_record_frame(rec_id: str, n: int):
    try:
        return FileResponse(records.frame_path(REC_ROOT, rec_id, n), media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=3600"})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.post("/api/records/{rec_id}/save")
def api_record_save(rec_id: str, title: str = Form("")):
    try:
        return records.save(REC_ROOT, rec_id, title)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.delete("/api/records/{rec_id}")
def api_record_delete(rec_id: str):
    try:
        return {"deleted": records.delete(REC_ROOT, rec_id)}
    except ValueError as e:
        raise HTTPException(400, str(e))
