"""
[팀 16] 교과 5일차 31번 — Split CSV의 Data Leakage 독립 검사

입력  reports/.../day05/splits/{train,val,val_ext,test,test_ext}.csv  (team_15)
      configs/<team>_ext_sources.csv  (외부 영상 원본 출처, 있으면)
검사
  1) Subject overlap   : 같은 사람이 두 split에 있으면 FAIL
  2) Clip overlap      : 같은 clip_id가 두 split에 있으면 FAIL
  3) Source overlap    : 같은 원본 영상(source_group)에서 자른 외부 Clip이 두 split에 있으면 FAIL
                         (Subject ID가 달라도 같은 사람·같은 영상일 수 있음)
  4) Action coverage   : 판정 각도별 Clip 수를 다시 세어 표로 출력 (비면 WARN)
문제가 하나라도 있으면 [FAIL] 출력 후 종료 코드 1 → 다음 단계(학습·평가)로 넘어가지 않음

실행
  python scripts/team_16_check_split_leakage.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import SPLIT_NAMES, day05_paths  # noqa: E402
from src.team_feature_utils import load_team_config  # noqa: E402


def main() -> None:
    cfg = load_team_config()
    d = day05_paths(cfg)
    splits = {}
    for name in SPLIT_NAMES:
        path = d["splits"] / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"[중지] {path} 없음 → team_15 먼저 실행")
        splits[name] = pd.read_csv(path, dtype=str).fillna("")

    src_path = ROOT / "configs" / f"{cfg['team_id']}_ext_sources.csv"
    src = (pd.read_csv(src_path, dtype=str).fillna("").set_index("clip_id")["source_group"].to_dict()
           if src_path.exists() else {})

    errors = []
    for a, b in itertools.combinations(SPLIT_NAMES, 2):
        A, B = splits[a], splits[b]
        if A.empty or B.empty:
            continue
        s = set(A["subject"]) & set(B["subject"])
        c = set(A["clip_id"]) & set(B["clip_id"])
        g = {src[x] for x in A["clip_id"] if x in src} & {src[x] for x in B["clip_id"] if x in src}
        if s:
            errors.append(f"{a}/{b} Subject Leakage: {sorted(s)}")
        if c:
            errors.append(f"{a}/{b} Clip Leakage: {sorted(c)}")
        if g:
            errors.append(f"{a}/{b} 같은 원본 영상: {sorted(g)}")

    for name, df in splits.items():
        print(f"{name:9s} Subjects: {sorted(df['subject'].unique()) if len(df) else '[]'}")
    print()
    print("[PASS] Subject overlap 없음" if not any("Subject" in e for e in errors) else "[FAIL] Subject overlap")
    print("[PASS] Clip overlap 없음" if not any("Clip" in e for e in errors) else "[FAIL] Clip overlap")
    print("[PASS] 원본 영상 overlap 없음" if not any("원본" in e for e in errors) else "[FAIL] 원본 영상 overlap")
    if not src:
        print("       (ext_sources 없음 → 원본 영상 검사는 생략)")
    elif len(src) < sum(len(df[df["subject"].str.startswith("EXT_")]) for df in splits.values()):
        print(f"       (출처 기록 {len(src)}개만 검사 — 나머지 외부 Clip도 {src_path.name}에 기록 권장)")

    cov_path = d["tables"] / "split_coverage.csv"
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
        empty = cov[(cov["train"] == 0) | ((cov["val"] == 0) & (cov["val_ext"] == 0))]
        print("[PASS] 판정 각도 Coverage" if empty.empty
              else f"[WARN] 판정 각도 Coverage 부족 {len(empty)}칸 (tables/split_coverage.csv)")

    if errors:
        print()
        for e in errors:
            print("[FAIL]", e)
        raise SystemExit(1)
    print("\nLeakage check: OK")


if __name__ == "__main__":
    main()
