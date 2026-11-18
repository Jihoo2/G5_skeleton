"""VICON GLB 정석 동작 → 여러 촬영 조건의 2D Keypoint로 증강 (team_18)

GLB 한 Take는 '정석 동작 1명분'뿐이므로, 실제 웹캠에서 달라지는 요소를 무작위로 바꿔 투영합니다.
  - 카메라 방향(yaw)        : 0°(정면) ~ 90°(측면), 좌우 반전(사람이 왼쪽/오른쪽을 봄)
  - 카메라 높이(pitch)      : 아래에서 올려보기 ~ 위에서 내려보기
  - 카메라 거리(원근)        : 가까울수록 원근 왜곡이 커짐 (project_glb는 원근 없는 정사영)
  - 체형                    : 허벅지·정강이·위팔·아래팔·몸통 길이 비율
  - 동작 속도               : 0.8 ~ 1.25배
  - YOLO 흔들림·누락        : 시간적으로 이어지는 잡음(AR(1)) + 손목·발목 등이 가끔 빠짐
좌표 규칙은 project_glb와 같음: VICON z = 몸 좌우, x = 앞뒤, y = 위.
VICON에는 Confidence가 없으므로 conf = NaN (1.0으로 채우지 않음).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.feature_utils import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)
from src.team_feature_utils import COCO_TO_VICON

DISTAL = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_ANKLE, RIGHT_ANKLE, LEFT_KNEE, RIGHT_KNEE]


@dataclass
class Variant:
    variant: int
    yaw_deg: float
    mirror: int
    pitch_deg: float
    cam_dist_m: float
    thigh: float
    shin: float
    upper_arm: float
    forearm: float
    torso: float
    speed: float
    noise: float
    dropout: float

    def as_dict(self) -> dict:
        return asdict(self)


def sample_variant(rng: np.random.Generator, i: int, yaw_range=(0.0, 90.0)) -> Variant:
    return Variant(
        variant=i,
        yaw_deg=float(rng.uniform(*yaw_range)),
        mirror=int(rng.integers(0, 2)),
        pitch_deg=float(rng.uniform(-5.0, 20.0)),
        cam_dist_m=float(rng.uniform(2.5, 6.0)),
        thigh=float(rng.uniform(0.92, 1.08)),
        shin=float(rng.uniform(0.92, 1.08)),
        upper_arm=float(rng.uniform(0.9, 1.1)),
        forearm=float(rng.uniform(0.9, 1.1)),
        torso=float(rng.uniform(0.92, 1.08)),
        speed=float(rng.uniform(0.8, 1.25)),
        noise=float(rng.uniform(0.005, 0.02)),
        dropout=float(rng.uniform(0.0, 0.02)),
    )


def glb_coco3d(names: list[str], P: np.ndarray) -> np.ndarray:
    """evaluate() 결과 → COCO 17 순서의 3D 좌표 (얼굴 0~4는 VICON에 없어 NaN)"""
    out = np.full((len(P), 17, 3), np.nan)
    for cid, bone in COCO_TO_VICON.items():
        out[:, cid] = P[:, names.index(bone)]
    return out


def reshape_body(X: np.ndarray, v: Variant) -> np.ndarray:
    """골반 중심을 고정하고 몸통·팔·다리 길이를 비율만큼 바꿈 (모든 Frame 같은 비율)"""
    Y = X.copy()
    hc = np.nanmean(X[:, [LEFT_HIP, RIGHT_HIP]], axis=1, keepdims=True)
    for s in (LEFT_SHOULDER, RIGHT_SHOULDER):
        Y[:, s] = hc[:, 0] + v.torso * (X[:, s] - hc[:, 0])
    for S, E, W in ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)):
        Y[:, E] = Y[:, S] + v.upper_arm * (X[:, E] - X[:, S])
        Y[:, W] = Y[:, E] + v.forearm * (X[:, W] - X[:, E])
    for H, K, A in ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)):
        Y[:, K] = Y[:, H] + v.thigh * (X[:, K] - X[:, H])
        Y[:, A] = Y[:, K] + v.shin * (X[:, A] - X[:, K])
    return Y


def retime(X: np.ndarray, speed: float) -> np.ndarray:
    """동작 속도 변경 (speed > 1 이면 빨라져 Frame 수가 줄어듦) — 선형 보간"""
    n = len(X)
    m = max(int(round(n / speed)), 2)
    src = np.linspace(0, n - 1, m)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0)[:, None, None]
    return X[i0] * (1 - w) + X[i1] * w


def project(X: np.ndarray, v: Variant) -> np.ndarray:
    """3D → 2D 원근 투영 → 이미지 좌표계(y 아래로 +), 단위는 임의 px (정규화에서 사라짐)"""
    th, ph = np.radians(v.yaw_deg), np.radians(v.pitch_deg)
    z, x, y = X[..., 2], X[..., 0], X[..., 1]
    u = z * np.cos(th) + x * np.sin(th)            # 화면 가로
    w = -z * np.sin(th) + x * np.cos(th)           # 카메라 쪽 깊이
    center_y = np.nanmedian(y)
    yc = y - center_y
    up = yc * np.cos(ph) + w * np.sin(ph)          # 카메라가 위에서 내려다보면(pitch+) 깊이가 세로에 섞임
    depth = -yc * np.sin(ph) + w * np.cos(ph)
    scale = 1000.0 / np.maximum(v.cam_dist_m - depth, 0.5)
    xy = np.stack([u * scale, -up * scale], axis=-1)
    if v.mirror:
        xy[..., 0] *= -1
    return xy


def add_detector_noise(xy: np.ndarray, v: Variant, rng: np.random.Generator) -> np.ndarray:
    """YOLO처럼 Frame 사이에 이어지는 흔들림 + 끝쪽 관절이 가끔 빠짐"""
    out = xy.copy()
    sh = np.nanmean(xy[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], axis=1)
    hp = np.nanmean(xy[:, [LEFT_HIP, RIGHT_HIP]], axis=1)
    torso = np.nanpercentile(np.linalg.norm(sh - hp, axis=1), 90)
    sigma, rho = v.noise * torso, 0.7
    e = np.zeros(xy.shape[1:])
    for t in range(len(xy)):
        e = rho * e + np.sqrt(1 - rho ** 2) * rng.normal(0, sigma, size=e.shape)
        out[t] += e
    drop = rng.random(len(xy)) < v.dropout
    for t in np.where(drop)[0]:
        out[t, rng.choice(DISTAL)] = np.nan
    return out


def make_variant_xy(X3: np.ndarray, v: Variant, rng: np.random.Generator) -> np.ndarray:
    X = retime(reshape_body(X3, v), v.speed)
    return add_detector_noise(project(X, v), v, rng)
