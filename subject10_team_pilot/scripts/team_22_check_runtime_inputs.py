"""
[팀 22] 실시간 판정·업로드 웹 실행 전 입력 점검 (교과 7일차 41_check_day07_inputs 자리)

웹(web/app.py)·실시간(team_21)·replay 는 Git에 없는 파일(data/, reports/)을 씁니다.
다른 PC에서 실행하기 전에 필요한 파일이 모두 있는지 먼저 확인합니다.

  FAIL : 없으면 판정이 안 됨 (실행 전에 반드시 해결)
  WARN : 없어도 실행은 되지만 일부 기능이 빠짐 (주의 점수, 곡선 비교, replay 등)
  OK   : 정상

실행
  python scripts/team_22_check_runtime_inputs.py
출력
  tables/runtime_input_check.csv  (FAIL 이 하나라도 있으면 종료 코드 1)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.realtime_judge import RUNTIME_DEFAULTS, RUNTIME_PATH, load_runtime_config  # noqa: E402
from src.team_feature_utils import load_team_config, metrics_for, team_paths  # noqa: E402

VIEWS = ["front", "diag", "side"]


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    rows: list[dict] = []

    def add(group: str, item: str, status: str, note: str = "") -> None:
        rows.append(dict(group=group, item=item, status=status, note=note))

    # 1) 설정 파일
    cfg = load_team_config()
    p = team_paths(cfg)
    add("설정", "team_project.json", "OK", f"team_id={cfg.get('team_id', '')}")
    margin = cfg["standard"].get("boundary_margin")
    add("설정", "standard.boundary_margin", "OK" if margin is not None else "WARN",
        f"{margin}" if margin is not None else "없음 → 기본 0.02 사용")
    if RUNTIME_PATH.exists():
        try:
            rt = load_runtime_config()
            add("설정", RUNTIME_PATH.name, "OK", ", ".join(f"{k}={v}" for k, v in rt.items()))
        except (ValueError, json.JSONDecodeError) as e:
            add("설정", RUNTIME_PATH.name, "FAIL", str(e))
    else:
        add("설정", RUNTIME_PATH.name, "WARN", f"없음 → 기본값 사용 {RUNTIME_DEFAULTS}")

    # 2) Pose 모델
    try:
        from src.common import load_config
        pc = load_config()
        pm = Path(pc["pose_model"])
        found = pm.exists() or (ROOT / pm).exists()
        add("Pose", f"pose_model ({pm.name})", "OK" if found else "WARN",
            "" if found else "로컬에 없음 → 첫 실행 때 ultralytics가 내려받음 (인터넷 필요)")
    except Exception as e:  # noqa: BLE001
        add("Pose", "configs/project.json", "FAIL", f"읽기 실패: {e}")

    # 3) 동작·각도별 판정 기준 / 곡선 템플릿 / 이상 탐지 모델
    std_dir, ref_dir = p["standards"], p["reference"]
    model_dir = p["reports"] / "models"
    for aid, action in cfg["actions"].items():
        label = f"{aid} {action.get('label', '')}".strip()
        judged = [v for v in VIEWS if metrics_for(action, v, cfg)]
        if not judged:
            add("기준", label, "WARN", "판정하는 각도가 없음")
            continue
        for v in judged:
            sp = std_dir / f"{aid}_{v}.json"
            if not sp.exists():
                add("기준", f"{label} {v}", "FAIL", f"{rel(sp)} 없음 → team_08_fit_standard.py")
            else:
                n = len(json.loads(sp.read_text(encoding="utf-8")).get("metrics", {}))
                add("기준", f"{label} {v}", "OK", f"측정값 {n}개")
            tp = ref_dir / f"{aid}_{v}_template.json"
            add("곡선", f"{label} {v}", "OK" if tp.exists() else "WARN",
                "" if tp.exists() else "템플릿 없음 → 웹 곡선 비교·curve_dist 생략 (team_07_build_reference.py)")
        mp = model_dir / f"form_model_{aid}.npz"
        add("모델", label, "OK" if mp.exists() else "WARN",
            "" if mp.exists() else "이상 탐지 모델 없음 → '주의' 표시 생략 (team_20_form_model_loso.py)")

    # 4) replay 검증에 필요한 파일
    add("replay", "manifest", "OK" if p["manifest"].exists() else "WARN", str(rel(p["manifest"])))
    rj = p["judgments"] / "rep_judgments.csv"
    if rj.exists():
        has_b = "boundary" in pd.read_csv(rj, nrows=1).columns
        add("replay", "rep_judgments.csv", "OK" if has_b else "WARN",
            "" if has_b else "boundary 컬럼 없음 → team_09_judge.py 다시 실행")
    else:
        add("replay", "rep_judgments.csv", "WARN", "없음 → team_09_judge.py (replay 검증 불가)")
    kp = p["keypoints"]
    n_kp = len(list(kp.glob("*_keypoints.csv"))) if kp.exists() else 0
    add("replay", "keypoints", "OK" if n_kp else "WARN", f"{n_kp}개" if n_kp else "없음 → team_03_extract_keypoints.py")

    # 5) 화면 표시용 한글 글꼴
    fonts = [Path("/mnt/c/Windows/Fonts/malgun.ttf"), Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")]
    ok_font = next((f for f in fonts if f.exists()), None)
    add("화면", "한글 글꼴", "OK" if ok_font else "WARN", str(ok_font) if ok_font else "없음 → 화면 글자가 영어로만 나옴")

    df = pd.DataFrame(rows)
    out = p["tables"] / "runtime_input_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    n_fail, n_warn = int((df.status == "FAIL").sum()), int((df.status == "WARN").sum())
    print(f"\nFAIL {n_fail}개, WARN {n_warn}개 → {rel(out)}")
    if n_fail:
        print("FAIL 항목을 해결한 뒤 실시간 판정·웹을 실행하세요.")
        sys.exit(1)
    print("실행 준비 완료")


if __name__ == "__main__":
    main()
