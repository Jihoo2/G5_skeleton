"""
팀05 맨몸운동 자세 판정 웹 (포트폴리오 시연)

실행 (subject10_team_pilot 폴더에서)
  uv add fastapi "uvicorn[standard]" python-multipart     # 처음 한 번
  uvicorn web.app:app --host 0.0.0.0 --port 8000
  → 브라우저 http://localhost:8000

기능 (1단계)
  - 영상 업로드 + 동작·각도 선택 → 백그라운드 분석(YOLO Pose + 실시간 엔진과 같은 판정)
  - 결과 페이지: 원본 영상 위 뼈대(캔버스), 회차 타임라인, 회차별 측정값 vs 기준, 무릎·고관절 곡선 vs GLB 정석
업로드 영상은 reports/team_projects/<team>/web_jobs/<id>/ 에만 저장됩니다 (Git 제외). 외부 공개 배포 금지.
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.team_feature_utils import load_team_config, team_paths  # noqa: E402
from web.processing import analyze_video  # noqa: E402

CFG = load_team_config()
P = team_paths(CFG)
JOBS_DIR = P["reports"] / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
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
