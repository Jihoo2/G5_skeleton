"""실시간(한 Frame씩) 1회 판정 엔진 — team_21

녹화 영상 판정(team_09)은 영상 전체를 본 뒤 '서 있을 때 높이'와 1회 구간을 계산합니다.
실시간에서는 미래 Frame이 없으므로 두 단계로 나눕니다.
  1) 감지(매 Frame): 최근 버퍼의 다리 높이에 녹화 판정과 같은 find_reps를 다시 적용하고,
     끝난 지 0.4초+평활화 여유가 지나 더 이상 합쳐지지 않을 회차만 '완료'로 확정
     (2026-11-18: 처음엔 별도 상태 머신을 썼으나 녹화 판정과 일치율 67% → 같은 함수로 통일)
     영상 파일 끝에서는 flush()로 남은 회차를 녹화 판정과 같은 규칙으로 마무리
  2) 측정·판정(1회가 끝날 때 한 번): 그 1회 앞뒤를 포함한 최근 구간을 team_09와 같은 함수
       (build_frame_features → rep_metrics → judge_rep)로 계산 → 녹화 판정과 같은 기준·메시지
     1회로 인정되는 깊이에 못 미쳤으면 '얕은 시도' NG (shallow_attempt_ng 동작만)
모델(team_20)이 있으면 같은 1회에 '주의' 점수를 붙입니다 (판정은 바꾸지 않음).
측정값이 기준값 ±boundary_margin 안이면 '경계'를 붙입니다 (판정은 바꾸지 않음, team_09 와 같은 함수).
실행 방식 값(대기·버퍼·확정 대기·측정 앞뒤 구간)은 configs/team_realtime.json 에서 읽습니다.
"""
from __future__ import annotations

import importlib.util
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_utils import LEFT_ANKLE, LEFT_HIP, LEFT_SHOULDER, RIGHT_ANKLE, RIGHT_HIP, RIGHT_SHOULDER
from src.form_model import FormModel, rep_block, series_frame
from src.team_feature_utils import (
    MESSAGES, build_frame_features, curve_distance, find_reps, find_shallow_attempts, metrics_for,
    rep_metrics, rep_min_len, shallow_attempts, view_group,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("team09", ROOT / "scripts" / "team_09_judge.py")
team09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team09)

# 아래 4개는 녹화 판정의 find_reps와 반드시 같아야 하므로 설정 파일로 빼지 않습니다.
ENTER, NEED = 0.85, 0.80           # find_reps와 같은 값
EXIT_HOLD_S, MIN_LEN_S = 0.4, 0.3  # merge_gap_s, min_len_s
SMOOTH_LAG = 7  # smooth()의 가운데 정렬 창(7) 때문에 끝쪽 값이 확정되기까지 필요한 Frame

# 실행 방식 설정(교과 7일차 day07_realtime.json 자리) — configs/team_realtime.json, 없으면 아래 기본값
RUNTIME_DEFAULTS = {
    "warmup_s": 1.0,    # 시작 후 판정 대기 (전신이 보이게 서 있는 시간)
    "buffer_s": 20.0,   # 최근 몇 초의 Frame을 기억할지 (회차 감지·기준 높이 계산 범위)
    "settle_s": 0.4,    # 1회가 끝나고 이만큼 + 평활화 여유가 지나야 확정 (find_reps merge_gap_s와 같게 시작)
    "pre_s": 1.2,       # 측정할 때 1회 앞에 붙이는 구간 (서 있는 기준 높이를 잡는 데 쓰임)
    "post_s": 0.5,      # 측정할 때 1회 뒤에 붙이는 구간
}
RUNTIME_PATH = ROOT / "configs" / "team_realtime.json"


def load_runtime_config(path: Path | None = None) -> dict:
    """configs/team_realtime.json 을 읽어 기본값 위에 덮어씀 ('_'로 시작하는 키는 설명이라 무시)"""
    rt = dict(RUNTIME_DEFAULTS)
    path = path or RUNTIME_PATH
    if path.exists():
        user = json.loads(path.read_text(encoding="utf-8"))
        unknown = [k for k in user if not k.startswith("_") and k not in RUNTIME_DEFAULTS]
        if unknown:
            raise ValueError(f"{path.name}: 알 수 없는 설정 {unknown} (가능: {sorted(RUNTIME_DEFAULTS)})")
        rt.update({k: float(v) for k, v in user.items() if not k.startswith("_")})
    return rt


def estimate_view(xy_buf: np.ndarray) -> str:
    """어깨 폭 / 몸통 길이로 촬영 각도 추정 (서 있는 Frame 기준)"""
    sw = np.linalg.norm(xy_buf[:, LEFT_SHOULDER] - xy_buf[:, RIGHT_SHOULDER], axis=1)
    sh = np.nanmean(xy_buf[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], 1)
    hp = np.nanmean(xy_buf[:, [LEFT_HIP, RIGHT_HIP]], 1)
    tl = np.linalg.norm(sh - hp, axis=1)
    r = np.nanmedian(sw / np.where(tl > 1e-6, tl, np.nan))
    if not np.isfinite(r):
        return "front"
    return "front" if r >= 0.55 else ("diag" if r >= 0.25 else "side")


class OnlineJudge:
    def __init__(self, cfg: dict, action_id: str, view: str, p: dict, fps: float = 30.0,
                 standards: str = "team", use_model: bool = True, rt: dict | None = None):
        self.cfg, self.aid, self.fps = cfg, action_id, float(fps)
        self.rt = rt if rt is not None else load_runtime_config()
        self.action = cfg["actions"][action_id]
        self.kp_conf = cfg["keypoint_confidence"]
        self.check_ratio = cfg["standard"]["low_conf_check_ratio"]
        self.margin = float(cfg["standard"].get("boundary_margin", 0.02))
        self.combine = self.action.get("leg_combine", "mean")
        self.min_rep = rep_min_len(self.action)  # 녹화 판정과 같은 동작별 1회 최소 길이
        self.p, self.std_kind = p, standards
        self.n_buf = int(self.rt["buffer_s"] * fps)
        self.fid, self.t = deque(maxlen=self.n_buf), deque(maxlen=self.n_buf)
        self.xy, self.conf = deque(maxlen=self.n_buf), deque(maxlen=self.n_buf)
        self.lr = deque(maxlen=self.n_buf)        # 감지용 (발목 y − 골반 y, 몸통 길이)
        self.view_fixed = view if view != "auto" else None
        self.view = view if view != "auto" else None
        self.state, self.rep_no, self.shallow_no = "WARMUP", 0, 0
        self.total = 0  # 지금까지 받은 Frame 수 (버퍼가 차서 앞쪽이 지워져도 위치를 절대값으로 관리)
        self.last_end = -1  # 마지막으로 판정한 구간의 끝 (절대 Frame 위치)
        self.counts = {"OK": 0, "NG": 0, "CHECK": 0}
        self.events: list[dict] = []
        self.model = None
        mpath = p["reports"] / "models" / f"form_model_{action_id}.npz"
        if use_model and mpath.exists():
            self.model = FormModel.load(mpath)
            self.model_tau = float(cfg["standard"].get("model_tau", 2.0))
        self._load_standard()

    # ------------------------------------------------------------------
    def _load_standard(self):
        self.metrics, self.tpl = None, None
        if self.view is None:
            return
        std_dir = self.p["standards"] if self.std_kind == "team" else self.p["reports"] / "standards_glb"
        sp = std_dir / f"{self.aid}_{self.view}.json"
        if metrics_for(self.action, self.view, self.cfg) and sp.exists():
            self.metrics = team09.usable_metrics(json.loads(sp.read_text(encoding="utf-8")),
                                                 self.cfg["standard"].get("judge_glb_only", False))
        tp = self.p["reference"] / f"{self.aid}_{self.view}_template.json"
        self.tpl = json.loads(tp.read_text(encoding="utf-8"))["curve"] if tp.exists() else None

    @staticmethod
    def _raw_leg(xy: np.ndarray) -> tuple[float, float]:
        """(발목 y − 골반 y, 몸통 길이) — 나눗셈은 버퍼 전체의 몸통 기준으로 나중에 (녹화 판정과 같은 방식)"""
        sh = np.nanmean(xy[[LEFT_SHOULDER, RIGHT_SHOULDER]], 0)
        hp = np.nanmean(xy[[LEFT_HIP, RIGHT_HIP]], 0)
        an = np.nanmean(xy[[LEFT_ANKLE, RIGHT_ANKLE]], 0)
        return float(an[1] - hp[1]), float(np.linalg.norm(sh - hp))

    def _lr_buffer(self) -> np.ndarray:
        raw = np.array(self.lr, float)          # (n, 2)
        ref = np.nanpercentile(raw[:, 1], 90) if np.isfinite(raw[:, 1]).any() else np.nan
        return raw[:, 0] / ref if np.isfinite(ref) and ref > 1e-6 else np.full(len(raw), np.nan)

    # ------------------------------------------------------------------
    def push(self, frame_id: int, t: float, xy: np.ndarray, conf: np.ndarray) -> list[dict]:
        """한 Frame 입력 → 이번 Frame에서 확정된 1회 판정 이벤트 목록"""
        self.fid.append(frame_id)
        self.t.append(t)
        self.xy.append(np.asarray(xy, float))
        self.conf.append(np.asarray(conf, float))
        self.lr.append(self._raw_leg(np.asarray(xy, float)))
        self.total += 1
        if len(self.fid) < self.rt["warmup_s"] * self.fps:
            self.state = "WARMUP"
            return []
        if self.view is None:
            self.view = estimate_view(np.array(self.xy))
            self._load_standard()
        settle = len(self.fid) - (int(round(self.rt["settle_s"] * self.fps)) + SMOOTH_LAG)
        out = self._emit(settle)
        lr = self._lr_buffer()
        stand = np.nanpercentile(lr, 90) if np.isfinite(lr).any() else np.nan
        cur = np.nanmean(lr[-5:]) if np.isfinite(lr[-5:]).any() else np.nan
        self.state = "DOWN" if (np.isfinite(cur) and np.isfinite(stand) and cur < stand * ENTER) else "STAND"
        self.events += out
        return out

    def _emit(self, settle: int, final: bool = False) -> list[dict]:
        lr = self._lr_buffer()
        first = self.total - len(self.fid)
        out = []
        reps, _ = find_reps(lr, self.fps, min_len_s=self.min_rep)
        segs = [(s, e, "rep") for s, e in reps]
        if self.action.get("shallow_attempt_ng"):
            segs += [(s, e, "shallow") for s, e in find_shallow_attempts(lr, self.fps)]
        for s, e, kind in sorted(segs):
            s_abs, e_abs = s + first, e + first
            if s_abs < self.last_end or (e > settle and not final):
                continue
            if kind == "shallow" and final and e >= len(lr) - max(1, int(round(0.1 * self.fps))):
                continue  # 영상 끝에 걸쳐 잘린 구간 (녹화 판정과 같은 규칙)
            self.last_end = e_abs
            out += self._finish(s_abs, e_abs, kind)
        return out

    def flush(self) -> list[dict]:
        """영상 파일 끝: 아직 확정 안 된 회차를 녹화 판정과 같은 규칙으로 마무리"""
        if len(self.fid) < 2:
            return []
        if self.view is None:
            self.view = estimate_view(np.array(self.xy))
            self._load_standard()
        out = self._emit(len(self.fid), final=True)
        self.events += out
        return out

    # ------------------------------------------------------------------
    def _finish(self, s_abs: int, e_abs: int, kind: str = "rep") -> list[dict]:
        first = self.total - len(self.fid)            # 버퍼 첫 칸의 절대 위치
        s, e = max(0, s_abs - first), e_abs - first   # 버퍼 안 위치
        a = max(0, s - int(self.rt["pre_s"] * self.fps))
        b = min(len(self.fid), e + int(self.rt["post_s"] * self.fps))
        fid = np.array(list(self.fid)[a:b])
        t = np.array(list(self.t)[a:b])
        xy = np.array(list(self.xy)[a:b])
        conf = np.array(list(self.conf)[a:b])
        view = self.view or "front"
        seq = build_frame_features(fid, t, xy, conf, view, self.kp_conf)
        reps, curves = rep_metrics(seq, view_group(view, self.cfg), self.fps, self.kp_conf, self.combine,
                                   min_len_s=self.min_rep)
        lo, hi = s - a, e - a
        best, best_ov = None, 0
        for k, rr in reps.iterrows():
            ov = min(hi, rr["end_frame"]) - max(lo, rr["start_frame"] - 1)
            if ov > best_ov:
                best, best_ov = k, ov
        base = dict(start_frame=int(fid[lo]), end_frame=int(fid[min(hi, len(fid) - 1)]),
                    start_s=round(float(t[lo]), 2), end_s=round(float(t[min(hi, len(t) - 1)]), 2))
        if kind == "shallow" or best is None:
            if kind == "shallow" or (self.action.get("shallow_attempt_ng") and shallow_attempts(
                    seq, view, self.fps, self.kp_conf, self.combine)):
                self.shallow_no += 1
                self.counts["NG"] += 1
                return [dict(kind="shallow", rep=f"얕은{self.shallow_no}", verdict="NG",
                             out_of_range="depth>shallow_attempt",
                             message="더 깊이 내려가세요 (1회 동작으로 인정되는 깊이까지 내려가지 않음)", **base)]
            return []
        self.rep_no += 1
        d = reps.loc[best].to_dict()
        d["curve_dist"] = curve_distance(curves[best], self.tpl) if self.tpl else np.nan
        if not self.metrics:
            v, bad, msgs = "CHECK", [], [f"이 동작은 {view_group(view, self.cfg)} 각도에서 판정하지 않음 → 측면·사선으로"]
        elif d["low_conf_ratio"] > self.check_ratio:
            v, bad, msgs = "CHECK", [], ["관절이 잘 보이지 않습니다. 카메라 위치를 조정하세요"]
        else:
            v, bad, msgs, _ = team09.judge_rep(d, self.metrics)
        bnd = team09.near_boundary(d, self.metrics, self.margin) if v != "CHECK" and self.metrics else []
        self.counts[v] += 1
        ev = dict(kind="rep", rep=self.rep_no, verdict=v, out_of_range="; ".join(bad),
                  message=" / ".join(dict.fromkeys(msgs)) if msgs else ("좋습니다" if v == "OK" else ""),
                  boundary="; ".join(bnd), view=view,
                  depth=d.get("depth"), knee_min=d.get("knee_min"), **base)
        # 로그용: 판정에 쓴 측정값 (team_21 이 v_<측정값> 컬럼으로 저장)
        ev["values"] = {m: (None if not np.isfinite(d.get(m, np.nan)) else round(float(d[m]), 4))
                        for m in (self.metrics or {})}
        # 웹 결과 화면용: 측정값 vs 기준, 1회 곡선 vs GLB 정석 곡선
        ev["checks"] = [dict(metric=m, value=None if not np.isfinite(d.get(m, np.nan)) else round(float(d[m]), 3),
                             lo=sd.get("lo"), hi=sd.get("hi"), provisional=bool(sd.get("provisional")),
                             ok=not any(b.startswith(m) for b in bad))
                        for m, sd in (self.metrics or {}).items()]
        c = curves[best]
        ev["curve"] = {k: [None if not np.isfinite(x) else round(float(x), 3) for x in c[k]]
                       for k in ("knee", "hip", "leg_ratio")}
        if self.tpl:
            ev["template"] = {k: [None if x is None or not np.isfinite(x) else round(float(x), 3) for x in self.tpl[k]]
                              for k in ("knee", "hip", "leg_ratio") if k in self.tpl}
        if self.model is not None:
            blk = rep_block(series_frame(seq, self.kp_conf), d["start_frame"], d["end_frame"])
            sc = float(self.model.score(blk[None])[0])
            ev["model_score"] = round(sc, 3)
            ev["model_flag"] = "주의" if sc > self.model_tau else ""
        return [ev]


__all__ = ["OnlineJudge", "estimate_view", "load_runtime_config", "RUNTIME_DEFAULTS", "MESSAGES"]
