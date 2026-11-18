"""정석 자세 이상 탐지 모델 (team_20) — 정석만 학습하고, 정석에서 벗어난 정도를 점수로 냅니다.

1회 동작 Feature (48 시점 × 8 시계열 = 384차원)
  knee_lo / knee_hi   : 좌우 무릎 각도 중 작은 값 / 큰 값  (좌우·앞뒤다리 바뀜에 영향 없음)
  hip_lo  / hip_hi    : 좌우 고관절 각도
  elbow_lo / elbow_hi : 좌우 팔꿈치 각도
  leg_ratio           : 골반이 발목보다 위에 있는 정도 (몸통 길이 기준)
  body_line           : 플랭크에서 어깨–발목 선 대비 골반 높이
  Confidence가 낮은 관절이 들어간 각도는 NaN (VICON은 NaN Confidence라 그대로 사용)

모델: 가중 PCA (numpy만 사용)
  - 정석 1회들의 평균·표준편차로 표준화 → 주성분 k개 (설명 분산 90%, 최대 8개)
  - T²  : 주성분 공간 안에서 평균에서 먼 정도 (예: 정석보다 훨씬 얕게 앉음)
  - SPE : 주성분으로 설명되지 않는 모양 (예: 정석에 없는 관절 움직임 조합)
  - score = max(T² / c_T2, SPE / c_SPE), c는 실제 정석 1회를 '사람 단위로 빼고 예측한' 점수의 95% 분위
    → 처음 보는 사람의 정석이면 대략 1 이하
  - 어느 시계열·시점이 벗어났는지 contribution()으로 확인 가능
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.pose_utils import COCO_KEYPOINT_NAMES
from src.team_feature_utils import TEMPLATE_LEN

SERIES = ["knee_lo", "knee_hi", "hip_lo", "hip_hi", "elbow_lo", "elbow_hi", "leg_ratio", "body_line"]
_ANGLE_JOINTS = {
    "knee": {"left": ("left_hip", "left_knee", "left_ankle"), "right": ("right_hip", "right_knee", "right_ankle")},
    "hip": {"left": ("left_shoulder", "left_hip", "left_knee"), "right": ("right_shoulder", "right_hip", "right_knee")},
    "elbow": {"left": ("left_shoulder", "left_elbow", "left_wrist"), "right": ("right_shoulder", "right_elbow", "right_wrist")},
}


def resample_block(a: np.ndarray, n: int = TEMPLATE_LEN) -> np.ndarray:
    """(T, F) → (n, F) 선형 보간 (NaN은 열마다 건너뛰고 보간, 유효값 2개 미만이면 NaN)"""
    T = len(a)
    if T < 2:
        return np.full((n, a.shape[1]), np.nan)
    src, idx = np.linspace(0, T - 1, n), np.arange(T)
    out = np.empty((n, a.shape[1]))
    for j in range(a.shape[1]):
        ok = np.isfinite(a[:, j])
        out[:, j] = np.interp(src, idx[ok], a[ok, j]) if ok.sum() >= 2 else np.nan
    return out


def _masked_angle(seq: pd.DataFrame, kind: str, side: str, kp_conf: float) -> np.ndarray:
    v = seq[f"{side}_{kind}_angle"].to_numpy(float).copy()
    cols = [f"{j}_conf" for j in _ANGLE_JOINTS[kind][side]]
    if all(c in seq for c in cols):
        c = seq[cols].to_numpy(float)
        v[np.any(c < kp_conf, axis=1)] = np.nan  # NaN(VICON) < kp_conf 는 False → 유지
    return v


def series_frame(seq: pd.DataFrame, kp_conf: float = 0.5) -> np.ndarray:
    """Sequence 전체 → (T, 8) 시계열"""
    cols = []
    for kind in ("knee", "hip", "elbow"):
        L, R = _masked_angle(seq, kind, "left", kp_conf), _masked_angle(seq, kind, "right", kp_conf)
        with np.errstate(invalid="ignore"):
            cols += [np.fmin(L, R), np.fmax(L, R)]
    cols.append(seq["leg_ratio"].to_numpy(float))
    bl = seq[["left_body_line_offset", "right_body_line_offset"]].to_numpy(float)
    with np.errstate(invalid="ignore"):
        cols.append(np.nanmean(np.where(np.isfinite(bl), bl, np.nan), axis=1)
                    if np.isfinite(bl).any() else np.full(len(seq), np.nan))
    return np.stack(cols, axis=1)


def rep_block(series: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    """rep_metrics의 start_frame(1부터)·end_frame → (48, 8)"""
    return resample_block(series[int(start_frame) - 1:int(end_frame)])


class FormModel:
    def __init__(self, var_ratio: float = 0.90, max_k: int = 8):
        self.var_ratio, self.max_k = var_ratio, max_k

    def _core(self, X: np.ndarray, w: np.ndarray) -> dict:
        w = w / w.sum()
        fin = np.isfinite(X)
        mu = (np.where(fin, X, 0) * w[:, None]).sum(0) / np.maximum((fin * w[:, None]).sum(0), 1e-12)
        Xf = np.where(fin, X, mu)
        sd = np.sqrt(((Xf - mu) ** 2 * w[:, None]).sum(0))
        col_scale = pd.Series(sd).groupby(np.arange(len(sd)) % len(SERIES)).transform("median").to_numpy()
        sd = np.maximum(sd, 0.1 * col_scale + 1e-6)  # 거의 안 움직이는 칸이 점수를 지배하지 않게
        Z = (Xf - mu) / sd
        _, S, Vt = np.linalg.svd(Z * np.sqrt(w)[:, None], full_matrices=False)
        var = S ** 2
        k = int(np.searchsorted(np.cumsum(var) / var.sum(), self.var_ratio) + 1)
        k = int(max(1, min(k, self.max_k, len(X) - 1)))
        return dict(mu=mu, sd=sd, P=Vt[:k], lam=np.maximum(var[:k], 1e-9), k=k)

    def fit(self, X: np.ndarray, w: np.ndarray, is_real: np.ndarray, groups=None) -> "FormModel":
        """groups(실제 1회의 사람 ID)를 주면 기준값 c를 '사람 단위 교차 예측' 점수로 정함.
        학습에 쓴 점수로 정하면 표본이 적을 때 c가 너무 작아져(과적합) 처음 보는 사람을 NG로 판정함."""
        X = X.reshape(len(X), -1).astype(float)
        self.__dict__.update(self._core(X, w))
        t2, spe = self._raw(X)
        ref = is_real if is_real.sum() >= 5 else np.ones(len(X), bool)
        t2_ref, spe_ref = t2[ref], spe[ref]
        if groups is not None:
            groups = np.asarray(groups, dtype=object)
            uniq = [g for g in pd.unique(groups[is_real]) if g is not None]
            if len(uniq) >= 3:
                t2_cv, spe_cv = [], []
                full = self.__dict__.copy()
                for g in uniq:
                    hold = is_real & (groups == g)
                    self.__dict__.update(self._core(X[~hold], w[~hold]))
                    a, b = self._raw(X[hold])
                    t2_cv.append(a)
                    spe_cv.append(b)
                self.__dict__.update(full)
                t2_ref, spe_ref = np.concatenate(t2_cv), np.concatenate(spe_cv)
        self.c_t2 = float(np.quantile(t2_ref, 0.95)) or 1.0
        self.c_spe = float(np.quantile(spe_ref, 0.95)) or 1.0
        return self

    def _z(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(len(X), -1).astype(float)
        Z = (np.where(np.isfinite(X), X, self.mu) - self.mu) / self.sd
        return Z

    def _raw(self, X: np.ndarray):
        Z = self._z(X)
        T = Z @ self.P.T
        t2 = (T ** 2 / self.lam).sum(1) / self.k
        spe = ((Z - T @ self.P) ** 2).mean(1)
        return t2, spe

    def score(self, X: np.ndarray) -> np.ndarray:
        t2, spe = self._raw(X)
        return np.maximum(t2 / self.c_t2, spe / self.c_spe)

    def contribution(self, x: np.ndarray) -> pd.DataFrame:
        """1회 동작 하나에서 시계열·구간별로 얼마나 벗어났는지 (표준화 편차의 제곱 평균)"""
        z = self._z(x[None])[0].reshape(TEMPLATE_LEN, len(SERIES))
        phases = np.array_split(np.arange(TEMPLATE_LEN), 4)
        rows = [{"series": s, **{f"phase{i + 1}": float((z[ph, j] ** 2).mean()) for i, ph in enumerate(phases)}}
                for j, s in enumerate(SERIES)]
        return pd.DataFrame(rows)

    def to_npz(self, path, **extra) -> None:
        np.savez_compressed(path, mu=self.mu, sd=self.sd, P=self.P, lam=self.lam, k=self.k,
                            c_t2=self.c_t2, c_spe=self.c_spe, series=np.array(SERIES), **extra)


__all__ = ["SERIES", "FormModel", "rep_block", "resample_block", "series_frame", "COCO_KEYPOINT_NAMES"]
