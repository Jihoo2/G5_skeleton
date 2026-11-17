"""
팀 프로젝트 공통 함수 (교과 10 3일차 src/feature_utils.py 확장)

- 38-D Frame Feature  : 교과 feature_utils의 함수를 그대로 호출 (열 이름·순서 동일)
- 팀 추가 Feature     : 고관절 각도, leg_ratio, body_line_offset, 관절별 Confidence
- 1회(rep) 구간 찾기   : leg_ratio가 '서 있을 때 값 × 0.6' 아래로 내려간 구간
- 1회 단위 측정값      : knee_min, hip_min, elbow_min, depth, body_line_min/max, duration_s
- GLB 투영            : 웹캠 각도(0=정면, 90=측면)에 맞춰 VICON 3D를 2D로 투영

좌표 규칙: 교과와 같이 이미지 좌표계(y가 아래로 갈수록 커짐)를 사용합니다.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_utils import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    calculate_angle, center_keypoints, normalize_by_shoulder_width, selected_joint_angles,
)
from src.pose_utils import COCO_KEYPOINT_NAMES

# NaN만 있는 구간(가려진 관절)의 평균 경고는 결과에 영향이 없으므로 숨김
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

ROOT = Path(__file__).resolve().parents[1]

# 한쪽 몸의 관절 묶음: 어깨, 팔꿈치, 손목, 골반, 무릎, 발목
SIDE_JOINTS = {
    "L": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "R": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}
# 정규화에 꼭 필요한 관절 (교과: 골반 중심 + 어깨 폭)
REQUIRED_FOR_NORM = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# COCO 번호 ↔ VICON Bone (어깨는 LeftShoulder(쇄골)가 아니라 LeftArm)
COCO_TO_VICON = {
    5: "LeftArm", 6: "RightArm", 7: "LeftForeArm", 8: "RightForeArm",
    9: "LeftHand", 10: "RightHand", 11: "LeftUpLeg", 12: "RightUpLeg",
    13: "LeftLeg", 14: "RightLeg", 15: "LeftFoot", 16: "RightFoot",
}

REP_METRICS = ["knee_min", "hip_min", "elbow_min", "depth",
               "body_line_min", "body_line_max", "knee_spread_bottom", "duration_s"]
TEMPLATE_LEN = 48  # 1회 구간을 48개 시점으로 맞춤 (교과 4일차 예시와 동일)


# ------------------------------------------------------------------
# 설정·경로
# ------------------------------------------------------------------
def load_team_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "team_project.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def view_group(view: str, cfg: dict) -> str:
    """front / diag / side / degNN → front·diag·side 중 하나 (0~22° 정면, 23~67° 사선, 68° 이상 측면)"""
    if view in ("front", "diag", "side"):
        return view
    deg = float(view.replace("deg", "")) if view.startswith("deg") else float(cfg["views_deg"][view])
    deg = abs(deg) % 180
    deg = min(deg, 180 - deg)
    return "front" if deg <= 22 else ("diag" if deg < 68 else "side")


def metrics_for(action: dict, view: str, cfg: dict) -> list[str]:
    """동작 × 각도에서 판정에 쓸 측정값.
    굽힘 각도(무릎·고관절·팔꿈치)와 몸통 일직선은 몸 옆면(시상면)에서 일어나는 움직임이라
    정면에서는 원근으로 찌그러져 믿을 수 없음 → 정면은 깊이·무릎 벌어짐처럼 정면에서 보이는 값만 사용."""
    by_view = action.get("metrics_by_view")
    if by_view is None:
        return list(action.get("key_metrics", []))
    return list(by_view.get(view_group(view, cfg), []))


def team_paths(cfg: dict) -> dict:
    tid = cfg["team_id"]
    data = ROOT / "data" / "team_projects" / tid
    rep = ROOT / "reports" / "team_projects" / tid
    paths = {
        "data": data,
        "rgb_videos": data / "rgb" / "videos",
        "ext_videos": data / "ext" / "videos",
        "vicon": data / "vicon",
        "manifest": ROOT / "configs" / f"{tid}_clip_manifest.csv",
        "reports": rep,
        "keypoints": rep / "keypoints",
        "sequences": rep / "sequences",
        "tables": rep / "tables",
        "reference": rep / "reference",
        "standards": rep / "standards",
        "judgments": rep / "judgments",
    }
    for k in ("keypoints", "sequences", "tables", "reference", "standards", "judgments"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


# ------------------------------------------------------------------
# 입력: 웹캠 Keypoint CSV / VICON GLB 투영
# ------------------------------------------------------------------
def load_keypoints_csv(path: str | Path):
    """교과 10번 형식 keypoints CSV → frame_id, time_sec, xy(n,17,2), conf(n,17)
    사람이 검출되지 않은 Frame은 NaN / conf 0 으로 채웁니다."""
    df = pd.read_csv(path)
    n = int(df["frame_id"].max())
    xy = np.full((n, 17, 2), np.nan, dtype=np.float64)
    conf = np.zeros((n, 17), dtype=np.float64)
    f = df["frame_id"].to_numpy() - 1
    k = df["keypoint_id"].to_numpy()
    xy[f, k, 0] = df["pixel_x"].to_numpy()
    xy[f, k, 1] = df["pixel_y"].to_numpy()
    conf[f, k] = df["confidence"].to_numpy()
    frame_id = np.arange(1, n + 1)
    times = df.groupby("frame_id")["time_sec"].first()
    time_sec = np.array([times.get(i, np.nan) for i in frame_id])
    return frame_id, time_sec, xy, conf


def project_glb(glb_path: str | Path, view_deg: float, fps: float = 30.0):
    """VICON GLB를 웹캠 각도로 투영 → 이미지 좌표계(y 아래로 +)의 xy(n,17,2).
    view_deg: 0 = 정면, 90 = 측면, 45 = 사선
    VICON에는 YOLO Confidence가 없으므로 conf는 NaN (1.0으로 채우지 않음)."""
    from src.glb_utils import evaluate  # 사용할 때만 불러옴

    names, P, _ = evaluate(glb_path, fps=fps)
    th = np.radians(view_deg)
    xy = np.full((len(P), 17, 2), np.nan, dtype=np.float64)
    for cid, bone in COCO_TO_VICON.items():
        p = P[:, names.index(bone)]
        # 이번 VICON 촬영본: z축 = 몸의 좌우 방향, x축 = 앞뒤 방향, y = 위
        xy[:, cid, 0] = p[:, 2] * np.cos(th) + p[:, 0] * np.sin(th)
        xy[:, cid, 1] = -p[:, 1]  # 위가 + → 이미지 좌표(아래가 +)로 변환
    conf = np.full((len(P), 17), np.nan)
    frame_id = np.arange(1, len(P) + 1)
    return frame_id, frame_id / fps - 1 / fps, xy, conf


# ------------------------------------------------------------------
# Frame Feature: 교과 38-D + 팀 추가 Feature
# ------------------------------------------------------------------
def _ok(conf_row, ids, kp_conf):
    c = conf_row[list(ids)]
    return bool(np.all(np.isnan(c)) or np.all(c >= kp_conf))  # VICON(NaN)은 통과


def torso_reference(xy: np.ndarray) -> float:
    """서 있을 때의 몸통 길이(어깨 중심~골반 중심) — Clip 전체에서 90번째 백분위수"""
    sh = np.nanmean(xy[:, [LEFT_SHOULDER, RIGHT_SHOULDER]], axis=1)
    hp = np.nanmean(xy[:, [LEFT_HIP, RIGHT_HIP]], axis=1)
    d = np.linalg.norm(sh - hp, axis=1)
    return float(np.nanpercentile(d, 90)) if np.isfinite(d).any() else float("nan")


def choose_scale_mode(xy: np.ndarray, view: str) -> str:
    """측면에서는 양쪽 어깨가 겹쳐 어깨 폭이 0에 가까워짐 → 몸통 길이로 정규화.
    정면·사선은 교과 방식(어깨 폭) 그대로 사용."""
    if view == "side":
        return "torso"
    sw = np.linalg.norm(xy[:, LEFT_SHOULDER] - xy[:, RIGHT_SHOULDER], axis=1)
    ratio = np.nanmedian(sw) / max(torso_reference(xy), 1e-6)
    return "shoulder" if ratio >= 0.35 else "torso"


def feature_columns() -> list[str]:
    """교과 16번 make_feature_names()와 같은 38개 열"""
    names = []
    for n in COCO_KEYPOINT_NAMES:
        names += [f"{n}_x", f"{n}_y"]
    return names + ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]


def build_frame_features(frame_id, time_sec, xy, conf, view: str,
                         kp_conf: float = 0.50, scale_mode: str = "auto") -> pd.DataFrame:
    """Frame별 38-D(교과) + 팀 Feature + 품질 정보"""
    mode = choose_scale_mode(xy, view) if scale_mode == "auto" else scale_mode
    torso_ref = torso_reference(xy)
    cols38 = feature_columns()
    rows = []
    for i in range(len(xy)):
        kp = xy[i]
        cf = conf[i]
        row = {"frame_id": int(frame_id[i]), "time_sec": float(time_sec[i])}
        detected = np.isfinite(kp[REQUIRED_FOR_NORM, :]).all()
        valid = detected and _ok(cf, REQUIRED_FOR_NORM, kp_conf)
        row["valid_pose"] = int(valid)
        has_conf = not np.all(np.isnan(cf))
        row["mean_keypoint_conf"] = float(np.mean(cf)) if has_conf else np.nan
        row["low_keypoint_count"] = int(np.sum(cf < kp_conf)) if has_conf else np.nan

        # ---- 교과 38-D (정규화 34 + 각도 4) ----
        feat = {c: np.nan for c in cols38}
        if detected:
            kp_filled = np.nan_to_num(kp)  # 얼굴(VICON에는 없음)은 0 → 아래에서 NaN 복원
            try:
                if mode == "shoulder":
                    norm, _, scale = normalize_by_shoulder_width(kp_filled)
                else:
                    centered, _ = center_keypoints(kp_filled)
                    scale = torso_ref
                    norm = centered / scale
                norm[~np.isfinite(kp).all(axis=1)] = np.nan
                for j, n in enumerate(COCO_KEYPOINT_NAMES):
                    feat[f"{n}_x"], feat[f"{n}_y"] = float(norm[j, 0]), float(norm[j, 1])
                feat.update(selected_joint_angles(kp))
                row["norm_scale_px"] = float(scale)
            except ValueError:  # 어깨 폭 0 등 정규화 불가
                row["valid_pose"] = 0
                row["norm_scale_px"] = np.nan
        else:
            row["norm_scale_px"] = np.nan
        row.update(feat)
        row["norm_scale_mode"] = mode

        # ---- 팀 추가 Feature ----
        for s, (S, E, W, H, K, A) in SIDE_JOINTS.items():
            pre = "left" if s == "L" else "right"
            row[f"{pre}_hip_angle"] = calculate_angle(kp[S], kp[H], kp[K]) if np.isfinite(kp[[S, H, K]]).all() else np.nan
            # 골반이 발목보다 얼마나 위에 있는가 (서 있음 ≈ 1.5, 스쿼트 바닥 ≈ 0.5, 플랭크 ≈ 0)
            row[f"{pre}_leg_ratio"] = (kp[A, 1] - kp[H, 1]) / torso_ref
            # 몸이 수평일 때 어깨-발목 직선 대비 골반 높이 (+ 엉덩이 들림, − 허리 처짐)
            dx = kp[A, 0] - kp[S, 0]
            if np.isfinite(dx) and abs(dx) > 0.5 * torso_ref:
                t = (kp[H, 0] - kp[S, 0]) / dx
                line_y = kp[S, 1] + t * (kp[A, 1] - kp[S, 1])
                row[f"{pre}_body_line_offset"] = (line_y - kp[H, 1]) / torso_ref
            else:
                row[f"{pre}_body_line_offset"] = np.nan
        row["leg_ratio"] = np.nanmean([row["left_leg_ratio"], row["right_leg_ratio"]])

        # ---- 관절별 Confidence (교과 4일차 권장: 품질 정보 별도 보존) ----
        for j, n in enumerate(COCO_KEYPOINT_NAMES):
            row[f"{n}_conf"] = float(cf[j]) if np.isfinite(cf[j]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 1회(rep) 구간과 측정값
# ------------------------------------------------------------------
def smooth(x, k: int = 7):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(k, center=True, min_periods=1).median().rolling(k, center=True, min_periods=1).mean().to_numpy()


def find_reps(leg_ratio, fps: float = 30.0, enter: float = 0.85, need: float = 0.80,
              merge_gap_s: float = 0.4, min_len_s: float = 0.3):
    """1회 구간 찾기.
    leg_ratio(엉덩이–발목 세로 거리 / 몸통 길이)가 '서 있을 때 값 × enter' 아래로 내려간 구간을 후보로 잡고,
    그 안의 최저값이 '서 있을 때 × need' 아래까지 내려간 경우만 1회로 인정.
    → 얕게 앉은 동작도 1회로 잡힌 뒤 깊이(depth) 기준으로 NG가 남 (구간을 못 찾아 CHECK로 빠지지 않음)."""
    r = smooth(leg_ratio)
    if not np.isfinite(r).any():
        return [], np.nan
    stand = np.nanpercentile(r, 90)
    rr = np.nan_to_num(r, nan=stand)
    low = rr < stand * enter
    reps, i, n = [], 0, len(r)
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            reps.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s_, e in reps:
        if merged and s_ - merged[-1][1] < merge_gap_s * fps:
            merged[-1][1] = e
        else:
            merged.append([s_, e])
    return [(s_, e) for s_, e in merged
            if e - s_ >= min_len_s * fps and rr[s_:e].min() < stand * need], float(stand)


def near_side(seq: pd.DataFrame) -> str:
    """측면 영상에서 카메라에 가까운 쪽 (팔·다리 Confidence 평균이 높은 쪽)"""
    def m(s):
        cols = [f"{COCO_KEYPOINT_NAMES[j]}_conf" for j in SIDE_JOINTS[s]]
        return np.nanmean(seq[cols].to_numpy())
    l, r = m("L"), m("R")
    if not np.isfinite(l) or not np.isfinite(r):
        return "LR"  # VICON: 양쪽 모두 사용
    return "L" if l >= r else "R"


def resolved_series(seq: pd.DataFrame, view: str, kp_conf: float = 0.50,
                    combine: str = "mean") -> dict:
    """1회 측정에 쓸 시계열. Confidence가 낮은 관절이 들어간 각도는 NaN 처리.
    combine="mean": 측면은 카메라 쪽 다리·팔만, 정면·사선은 좌우 평균 (양발이 같이 움직이는 동작)
    combine="min" : 모든 각도에서 좌우 중 작은 각도 = 앞으로 내디딘 다리 (런지처럼 다리가 앞뒤로 갈리는 동작,
                    번갈아 런지면 앞다리가 매번 바뀌므로 '작은 쪽'을 앞다리로 봄)"""
    side = near_side(seq) if (view == "side" and combine == "mean") else "LR"
    sides = ["L", "R"] if side == "LR" else [side]

    def masked(angle_col, joint_ids, s):
        v = seq[angle_col].to_numpy(dtype=float).copy()
        cols = [f"{COCO_KEYPOINT_NAMES[j]}_conf" for j in joint_ids]
        c = seq[cols].to_numpy(dtype=float)
        bad = np.any(c < kp_conf, axis=1)  # NaN(VICON)은 비교 결과 False → 유지
        v[bad] = np.nan
        return v

    out = {"knee": [], "hip": [], "elbow": [], "leg_ratio": [], "body_line": []}
    for s in sides:
        pre = "left" if s == "L" else "right"
        S, E, W, H, K, A = SIDE_JOINTS[s]
        out["knee"].append(masked(f"{pre}_knee_angle", (H, K, A), s))
        out["hip"].append(masked(f"{pre}_hip_angle", (S, H, K), s))
        out["elbow"].append(masked(f"{pre}_elbow_angle", (S, E, W), s))
        out["leg_ratio"].append(seq[f"{pre}_leg_ratio"].to_numpy(dtype=float))
        out["body_line"].append(seq[f"{pre}_body_line_offset"].to_numpy(dtype=float))
    res = {}
    for k, v in out.items():
        arr = np.vstack(v)
        agg = np.nanmin if (combine == "min" and k in ("knee", "hip")) else np.nanmean
        with np.errstate(all="ignore"):
            res[k] = smooth(agg(arr, axis=0)) if np.isfinite(arr).any() else np.full(arr.shape[1], np.nan)
    # 무릎 벌어짐 = 두 무릎 가로 간격 / 두 발목 가로 간격 (정면용, 1보다 작으면 무릎이 안쪽으로 모임)
    def xcol(j):
        v = seq[f"{COCO_KEYPOINT_NAMES[j]}_x"].to_numpy(dtype=float).copy()
        c = seq[f"{COCO_KEYPOINT_NAMES[j]}_conf"].to_numpy(dtype=float)
        v[c < kp_conf] = np.nan
        return v
    with np.errstate(all="ignore"):
        knee_w = np.abs(xcol(LEFT_KNEE) - xcol(RIGHT_KNEE))
        ankle_w = np.abs(xcol(LEFT_ANKLE) - xcol(RIGHT_ANKLE))
        spread = knee_w / ankle_w
    spread[~(ankle_w > 0.25)] = np.nan  # 발이 겹쳐 보이면(측면 등) 계산하지 않음
    res["knee_spread"] = spread

    # 몸통 일직선(body_line)은 플랭크처럼 몸통이 누워 있는 Frame에서만 의미가 있음
    # → 어깨중점–엉덩이중점 선이 수평에서 35° 이내인 Frame만 남김 (버피의 서기·웅크리기 구간 제외)
    def mid(a, b, ax):
        return (seq[f"{COCO_KEYPOINT_NAMES[a]}_{ax}"].to_numpy(dtype=float)
                + seq[f"{COCO_KEYPOINT_NAMES[b]}_{ax}"].to_numpy(dtype=float)) / 2
    with np.errstate(all="ignore"):
        dx = mid(LEFT_SHOULDER, RIGHT_SHOULDER, "x") - mid(LEFT_HIP, RIGHT_HIP, "x")
        dy = mid(LEFT_SHOULDER, RIGHT_SHOULDER, "y") - mid(LEFT_HIP, RIGHT_HIP, "y")
        lying = np.abs(dy) < np.abs(dx) * np.tan(np.radians(35))
        # 다리도 펴져 있어야 플랭크 (웅크린 상태에서 다리를 뒤로 뻗기 직전 자세 제외)
        knees = np.nanmean(np.vstack([seq["left_knee_angle"].to_numpy(dtype=float),
                                      seq["right_knee_angle"].to_numpy(dtype=float)]), axis=0)
        lying &= knees > 140
        # 다리(엉덩이중점→발목중점)도 수평에서 25° 이내 = 발이 뒤로 뻗어 있음 (몸을 숙여 손을 짚는 순간 제외)
        lx = mid(LEFT_ANKLE, RIGHT_ANKLE, "x") - mid(LEFT_HIP, RIGHT_HIP, "x")
        ly = mid(LEFT_ANKLE, RIGHT_ANKLE, "y") - mid(LEFT_HIP, RIGHT_HIP, "y")
        lying &= np.abs(ly) < np.abs(lx) * np.tan(np.radians(25))
    bl = res["body_line"].copy()
    bl[~lying] = np.nan
    res["body_line"] = bl
    res["lying"] = lying
    res["side"] = side
    return res


def _nanstat(fn, x):
    x = np.asarray(x, dtype=float)
    return float(fn(x)) if np.isfinite(x).any() else np.nan


def _bottom_median(x, leg_ratio, q: float = 20):
    """가장 낮게 내려간 구간(leg_ratio 하위 20%)에서의 중앙값 — 착지·점프 순간 흔들림 제외"""
    x, lr = np.asarray(x, float), np.asarray(leg_ratio, float)
    if not np.isfinite(lr).any():
        return np.nan
    m = lr <= np.nanpercentile(lr, q)
    return _nanstat(np.nanmedian, x[m])


def resample(x, n: int = TEMPLATE_LEN):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full(n, np.nan)
    x = np.interp(idx, idx[ok], x[ok])
    return np.interp(np.linspace(0, len(x) - 1, n), idx, x)


def rep_metrics(seq: pd.DataFrame, view: str, fps: float = 30.0, kp_conf: float = 0.50,
                combine: str = "mean"):
    """Clip 하나 → 1회 단위 측정값 표 + 1회 곡선(48점)"""
    ser = resolved_series(seq, view, kp_conf, combine)
    reps, stand = find_reps(ser["leg_ratio"], fps)
    rows, curves = [], []
    for n, (s, e) in enumerate(reps, 1):
        seg = slice(s, e)
        key_cols = [c for c in ser if c not in ("side",)]
        low_ratio = float(np.mean(~np.isfinite(ser["knee"][seg]) | ~np.isfinite(ser["hip"][seg])))
        rows.append(dict(
            rep=n, start_frame=int(s + 1), end_frame=int(e), side=ser["side"],
            duration_s=round((e - s) / fps, 3),
            knee_min=_nanstat(np.nanmin, ser["knee"][seg]),
            hip_min=_nanstat(np.nanmin, ser["hip"][seg]),
            elbow_min=_nanstat(np.nanmin, ser["elbow"][seg]),
            # 깊이 = 가장 낮을 때 다리 높이 / 그 사람이 서 있을 때 다리 높이 (0.4 = 서 있을 때의 40%까지 내려감)
            # 사람마다 다리·몸통 비율이 달라도 같은 기준으로 비교되도록 각자 서 있는 자세로 나눔
            depth=_nanstat(np.nanmin, ser["leg_ratio"][seg]) / stand,
            body_line_min=_nanstat(np.nanmin, ser["body_line"][seg]),
            body_line_max=_nanstat(np.nanmax, ser["body_line"][seg]),
            knee_spread_bottom=_bottom_median(ser["knee_spread"][seg], ser["leg_ratio"][seg]),
            low_conf_ratio=round(low_ratio, 3),
        ))
        curves.append({k: resample(ser[k][seg]) for k in ("knee", "hip", "leg_ratio")})
    return pd.DataFrame(rows), curves


def curve_distance(curve: dict, template: dict) -> float:
    """1회 곡선과 기준(GLB) 곡선의 평균 각도 차이(°) — 무릎·고관절 평균"""
    d = []
    for k in ("knee", "hip"):
        a, b = np.asarray(curve[k]), np.asarray(template[k])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= TEMPLATE_LEN // 2:
            d.append(np.mean(np.abs(a[ok] - b[ok])))
    return float(np.mean(d)) if d else np.nan


# ------------------------------------------------------------------
# 교정 메시지
# ------------------------------------------------------------------
MESSAGES = {
    ("knee_min", "high"): "무릎을 더 굽혀 더 깊이 앉으세요",
    ("knee_min", "low"): "무릎을 너무 많이 굽혔습니다",
    ("hip_min", "high"): "엉덩이(고관절)를 더 접으세요",
    ("hip_min", "low"): "상체를 너무 숙였습니다",
    ("elbow_min", "high"): "팔꿈치를 더 굽히세요",
    ("elbow_min", "low"): "팔꿈치를 너무 굽혔습니다",
    ("depth", "high"): "몸을 더 낮게 내리세요",
    ("depth", "low"): "너무 낮게 내려갔습니다",
    ("body_line_min", "low"): "허리가 처졌습니다. 몸을 일직선으로 유지하세요",
    ("body_line_min", "high"): "몸이 일직선이 되도록 엉덩이를 내리세요",
    ("body_line_max", "high"): "엉덩이가 너무 높습니다",
    ("body_line_max", "low"): "허리가 처졌습니다",
    ("duration_s", "high"): "동작이 기준보다 느립니다",
    ("duration_s", "low"): "동작이 기준보다 빠릅니다",
    ("curve_dist", "high"): "동작 흐름이 정석 자세와 다릅니다",
    ("knee_spread_bottom", "low"): "무릎이 안쪽으로 모입니다. 무릎을 발끝 방향으로 벌리세요",
    ("knee_spread_bottom", "high"): "무릎이 너무 벌어졌습니다",
}
