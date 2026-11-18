"""실시간(한 Frame씩) 1회 판정 엔진 — team_21

녹화 영상 판정(team_09)은 영상 전체를 본 뒤 '서 있을 때 높이'와 1회 구간을 계산합니다.
실시간에서는 미래 Frame이 없으므로 두 단계로 나눕니다.
  1) 감지(가벼움, 매 Frame): 다리 높이(leg_ratio)를 최근 몇 초의 '서 있을 때 값'과 비교하는 상태 머신
       STAND → (서 있을 때 × 0.85 아래) DOWN → (다시 위로 0.4초 유지) 완료
  2) 측정·판정(1회가 끝날 때 한 번): 그 1회 앞뒤를 포함한 최근 구간을 team_09와 같은 함수
       (build_frame_features → rep_metrics → judge_rep)로 계산 → 녹화 판정과 같은 기준·메시지
     1회로 인정되는 깊이에 못 미쳤으면 '얕은 시도' NG (shallow_attempt_ng 동작만)
모델(team_20)이 있으면 같은 1회에 '주의' 점수를 붙입니다 (판정은 바꾸지 않음).
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
    MESSAGES, build_frame_features, curve_distance, metrics_for, rep_metrics, shallow_attempts,
    view_group,
)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("team09", ROOT / "scripts" / "team_09_judge.py")
team09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(team09)

ENTER, NEED = 0.85, 0.80           # find_reps와 같은 값
EXIT_HOLD_S, MIN_LEN_S = 0.4, 0.3  # merge_gap_s, min_len_s
PRE_S, POST_S = 1.2, 0.5           # 측정할 때 1회 앞뒤로 붙이는 여유
BUFFER_S, STAND_WIN_S, WARMUP_S = 20.0, 8.0, 1.5


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
                 standards: str = "team", use_model: bool = True):
        self.cfg, self.aid, self.fps = cfg, action_id, float(fps)
        self.action = cfg["actions"][action_id]
        self.kp_conf = cfg["keypoint_confidence"]
        self.check_ratio = cfg["standard"]["low_conf_check_ratio"]
        self.combine = self.action.get("leg_combine", "mean")
        self.p, self.std_kind = p, standards
        self.n_buf = int(BUFFER_S * fps)
        self.fid, self.t = deque(maxlen=self.n_buf), deque(maxlen=self.n_buf)
        self.xy, self.conf = deque(maxlen=self.n_buf), deque(maxlen=self.n_buf)
        self.lr = deque(maxlen=self.n_buf)        # 감지용 leg_ratio (몸통 길이 기준)
        self.view_fixed = view if view != "auto" else None
        self.view = view if view != "auto" else None
        self.state, self.rep_no, self.shallow_no = "WARMUP", 0, 0
        self.total = 0  # 지금까지 받은 Frame 수 (버퍼가 차서 앞쪽이 지워져도 위치를 절대값으로 관리)
        self.down_start, self.down_min, self.up_since = None, np.inf, None
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
    def _leg_ratio(xy: np.ndarray) -> float:
        sh = np.nanmean(xy[[LEFT_SHOULDER, RIGHT_SHOULDER]], 0)
        hp = np.nanmean(xy[[LEFT_HIP, RIGHT_HIP]], 0)
        an = np.nanmean(xy[[LEFT_ANKLE, RIGHT_ANKLE]], 0)
        torso = np.linalg.norm(sh - hp)
        if not (np.isfinite(torso) and torso > 1e-6 and np.isfinite(an[1]) and np.isfinite(hp[1])):
            return np.nan
        return float((an[1] - hp[1]) / torso)

    def _smoothed(self, k: int = 5) -> float:
        v = np.array(list(self.lr)[-k:], float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan

    def _stand(self) -> float:
        n = int(STAND_WIN_S * self.fps)
        v = np.array(list(self.lr)[-n:], float)
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 90)) if len(v) else np.nan

    # ------------------------------------------------------------------
    def push(self, frame_id: int, t: float, xy: np.ndarray, conf: np.ndarray) -> list[dict]:
        """한 Frame 입력 → 이번 Frame에서 끝난 1회 판정 이벤트 목록"""
        self.fid.append(frame_id)
        self.t.append(t)
        self.xy.append(np.asarray(xy, float))
        self.conf.append(np.asarray(conf, float))
        self.lr.append(self._leg_ratio(np.asarray(xy, float)))
        self.total += 1
        out = []
        if self.state == "WARMUP":
            if len(self.lr) >= WARMUP_S * self.fps and np.isfinite(self._stand()):
                if self.view is None:
                    self.view = estimate_view(np.array(self.xy))
                    self._load_standard()
                self.state = "STAND"
            return out
        lr, stand = self._smoothed(), self._stand()
        if not (np.isfinite(lr) and np.isfinite(stand)):
            return out
        i = self.total - 1  # 절대 위치
        if self.state == "STAND":
            if lr < stand * ENTER:
                self.state, self.down_start, self.down_min, self.up_since = "DOWN", i, lr, None
        else:  # DOWN
            self.down_min = min(self.down_min, lr)
            if lr < stand * ENTER:
                self.up_since = None
            else:
                self.up_since = i if self.up_since is None else self.up_since
                if (i - self.up_since) >= EXIT_HOLD_S * self.fps:
                    out += self._finish(self.down_start, self.up_since, stand)
                    self.state = "STAND"
        self.events += out
        return out

    def flush(self) -> list[dict]:
        """영상 끝: 내려가던 중 끝났으면 버림 (녹화 판정에서 끝에 걸친 구간을 버리는 것과 같음)"""
        return []

    # ------------------------------------------------------------------
    def _finish(self, s_abs: int, e_abs: int, stand: float) -> list[dict]:
        if (e_abs - s_abs) < MIN_LEN_S * self.fps:
            return []
        first = self.total - len(self.fid)            # 버퍼 첫 칸의 절대 위치
        s, e = max(0, s_abs - first), e_abs - first   # 버퍼 안 위치
        a = max(0, s - int(PRE_S * self.fps))
        b = min(len(self.fid), e + int(POST_S * self.fps))
        fid = np.array(list(self.fid)[a:b])
        t = np.array(list(self.t)[a:b])
        xy = np.array(list(self.xy)[a:b])
        conf = np.array(list(self.conf)[a:b])
        view = self.view or "front"
        seq = build_frame_features(fid, t, xy, conf, view, self.kp_conf)
        reps, curves = rep_metrics(seq, view_group(view, self.cfg), self.fps, self.kp_conf, self.combine)
        lo, hi = s - a, e - a
        best, best_ov = None, 0
        for k, rr in reps.iterrows():
            ov = min(hi, rr["end_frame"]) - max(lo, rr["start_frame"] - 1)
            if ov > best_ov:
                best, best_ov = k, ov
        base = dict(start_frame=int(fid[lo]), end_frame=int(fid[min(hi, len(fid) - 1)]),
                    start_s=round(float(t[lo]), 2), end_s=round(float(t[min(hi, len(t) - 1)]), 2))
        if best is None:
            if self.action.get("shallow_attempt_ng") and shallow_attempts(
                    seq, view, self.fps, self.kp_conf, self.combine):
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
        self.counts[v] += 1
        ev = dict(kind="rep", rep=self.rep_no, verdict=v, out_of_range="; ".join(bad),
                  message=" / ".join(dict.fromkeys(msgs)) if msgs else ("좋습니다" if v == "OK" else ""),
                  depth=d.get("depth"), knee_min=d.get("knee_min"), **base)
        if self.model is not None:
            blk = rep_block(series_frame(seq, self.kp_conf), d["start_frame"], d["end_frame"])
            sc = float(self.model.score(blk[None])[0])
            ev["model_score"] = round(sc, 3)
            ev["model_flag"] = "주의" if sc > self.model_tau else ""
        return [ev]


__all__ = ["OnlineJudge", "estimate_view", "MESSAGES"]
