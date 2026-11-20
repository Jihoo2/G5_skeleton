"""실시간 판정 녹화본 — 저장하지 않으면 한 번만 볼 수 있고 사라짐

저장 위치: reports/team_projects/<team>/live_records/<녹화ID>/   (Git 제외 폴더)
  frames/000001.jpg ...   판정에 쓴 Frame (가로 640 이하 JPEG)
  timeline.json           Frame마다 촬영 시각·관절·상태·회차 수 + 회차 판정 이벤트 (재생할 때 화면 위에 그림)
  meta.json               제목·동작·입력·길이·회차·OK/NG 수, status = temp(임시) / saved(저장)

임시(temp) 녹화본이 지워지는 때
  - 다음 실시간 판정을 시작할 때
  - 서버를 다시 켤 때
  - 녹화본 화면에서 '삭제'를 누를 때
'저장'을 누르면 saved 가 되어 녹화 목록(/records)에서 언제든 다시 볼 수 있습니다.
영상에는 사람 얼굴이 담기므로 외부 공개·Git 업로드를 하지 않습니다.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import uuid
from pathlib import Path

REC_MAX_S = 600.0  # 한 번에 최대 10분까지만 녹화 (디스크 보호)
_ID_RE = re.compile(r"^[0-9A-Za-z_-]{6,64}$")
ACTIVE: set[str] = set()  # 지금 녹화 중인 ID (다른 탭에서 새로 시작해도 지우지 않음)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def rec_dir(root: Path, rec_id: str) -> Path:
    """녹화ID 검사 후 폴더 경로 (../ 같은 경로 조작 방지)"""
    if not _ID_RE.match(rec_id or ""):
        raise ValueError("잘못된 녹화 ID")
    return root / rec_id


class Recorder:
    """실시간 판정 중 Frame과 판정 정보를 임시 녹화본으로 기록"""

    def __init__(self, root: Path, meta: dict):
        self.id = f"{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        self.dir = root / self.id
        (self.dir / "frames").mkdir(parents=True)
        self.meta = dict(meta, id=self.id, status="temp", created=dt.datetime.now().isoformat(timespec="seconds"),
                         title="", truncated=False)
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self.t0: float | None = None
        self.size: tuple[int, int] | None = None
        self.stopped = False
        ACTIVE.add(self.id)
        _write_json(self.dir / "meta.json", self.meta)

    @property
    def n(self) -> int:
        return len(self.frames)

    def add(self, jpg: bytes, t: float, info: dict) -> None:
        """jpg: 화면에 보이는 그대로의 Frame / t: 촬영 시각(초) / info: 판정 응답(kp·상태·회차·events)"""
        if self.stopped or not jpg:
            return
        if self.t0 is None:
            self.t0 = t
        tt = round(t - self.t0, 3)
        if tt > REC_MAX_S:
            self.stopped, self.meta["truncated"] = True, True
            return
        i = self.n + 1
        (self.dir / "frames" / f"{i:06d}.jpg").write_bytes(jpg)
        if self.size is None:
            self.size = (int(info.get("width") or 0), int(info.get("height") or 0))
        self.frames.append(dict(i=i, t=tt, kp=info.get("kp"), state=info.get("state"), view=info.get("view"),
                                rep_no=info.get("rep_no"), counts=info.get("counts")))
        for ev in info.get("events") or []:
            self.events.append(dict(ev, t=tt))

    def close(self, summary: dict) -> str | None:
        """녹화 종료: timeline.json 저장. Frame이 하나도 없으면 폴더를 지우고 None"""
        ACTIVE.discard(self.id)
        if not self.frames:
            shutil.rmtree(self.dir, ignore_errors=True)
            return None
        w, h = self.size or (0, 0)
        c = self.frames[-1].get("counts") or {}
        self.meta.update(width=w, height=h, n_frames=self.n, duration=self.frames[-1]["t"],
                         reps=summary.get("reps"), OK=c.get("OK", 0), NG=c.get("NG", 0), CHECK=c.get("CHECK", 0))
        _write_json(self.dir / "timeline.json", dict(meta=self.meta, frames=self.frames, events=self.events))
        _write_json(self.dir / "meta.json", self.meta)
        return self.id


# ----------------------------------------------------------------------
# 목록·저장·삭제
# ----------------------------------------------------------------------
def list_records(root: Path, status: str = "saved") -> list[dict]:
    out = []
    for m in root.glob("*/meta.json") if root.exists() else []:
        try:
            meta = _read_json(m)
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("status") == status and (m.parent / "timeline.json").exists():
            out.append(meta)
    return sorted(out, key=lambda x: x.get("created", ""), reverse=True)


def timeline(root: Path, rec_id: str) -> dict:
    p = rec_dir(root, rec_id) / "timeline.json"
    if not p.exists():
        raise FileNotFoundError("녹화본이 없습니다 (저장하지 않아 지워졌을 수 있습니다)")
    return _read_json(p)


def frame_path(root: Path, rec_id: str, n: int) -> Path:
    p = rec_dir(root, rec_id) / "frames" / f"{int(n):06d}.jpg"
    if not p.exists():
        raise FileNotFoundError("Frame 없음")
    return p


def save(root: Path, rec_id: str, title: str = "") -> dict:
    d = rec_dir(root, rec_id)
    if not (d / "timeline.json").exists():
        raise FileNotFoundError("녹화본이 없습니다 (이미 지워졌을 수 있습니다)")
    meta = _read_json(d / "meta.json")
    meta.update(status="saved", title=(title or "").strip()[:80] or meta.get("title", ""),
                saved=dt.datetime.now().isoformat(timespec="seconds"))
    _write_json(d / "meta.json", meta)
    tl = _read_json(d / "timeline.json")
    tl["meta"] = meta
    _write_json(d / "timeline.json", tl)
    return meta


def delete(root: Path, rec_id: str) -> bool:
    d = rec_dir(root, rec_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def cleanup_temp(root: Path) -> int:
    """저장하지 않은(temp) 녹화본을 모두 지움 — 새 실시간 판정 시작·서버 시작 때"""
    n = 0
    for d in root.iterdir() if root.exists() else []:
        m = d / "meta.json"
        try:
            temp = (not m.exists()) or _read_json(m).get("status") != "saved"
        except (OSError, json.JSONDecodeError):
            temp = True
        if d.is_dir() and temp and d.name not in ACTIVE:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
