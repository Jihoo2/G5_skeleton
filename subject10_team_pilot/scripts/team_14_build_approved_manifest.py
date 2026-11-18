"""
[팀 14] 교과 5일차 28번 + 29번 — Human QA 입력 검사 → Approved Manifest

28번 검사 (교과 수정본 기준: CHECK만 사람 확인 필수)
  - human_decision 값은 PASS / RESHOOT / EXCLUDE / UNREVIEWED 만 허용
  - qa_status=CHECK 인데 UNREVIEWED → 필수 Human QA 미완료로 중지
  - AUTO_PASS + UNREVIEWED 는 허용 (팀은 22개 전부 사람이 확정했으므로 실제로는 없음)
29번 승인 기준
  AUTO_PASS + UNREVIEWED → 승인 /  PASS → 승인 /  RESHOOT·EXCLUDE → 제외

출력  reports/.../day05/manifests/approved_clip_manifest.csv
      화면: Subject별·동작별·(동작, 각도)별 승인 Clip 수

실행
  python scripts/team_14_build_approved_manifest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.day05_common import day05_paths  # noqa: E402
from src.team_feature_utils import load_team_config  # noqa: E402

ALLOWED = {"PASS", "RESHOOT", "EXCLUDE", "UNREVIEWED"}


def main() -> None:
    cfg = load_team_config()
    d = day05_paths(cfg)
    if not d["review"].exists():
        raise SystemExit("[중지] keypoint_clip_review.csv 없음 → team_13 먼저 실행")
    df = pd.read_csv(d["review"], dtype=str).fillna("")
    dec = df["human_decision"].str.strip().str.upper()

    bad = df[~dec.isin(ALLOWED)]
    if len(bad):
        print(bad[["clip_id", "qa_status", "human_decision"]].to_string(index=False))
        raise SystemExit("[INVALID HUMAN DECISION] PASS / RESHOOT / EXCLUDE 만 사용하세요.")
    todo = df[(df["qa_status"] == "CHECK") & (dec == "UNREVIEWED")]
    if len(todo):
        print(todo[["clip_id", "qa_status", "qa_reason"]].to_string(index=False))
        raise SystemExit(f"[HUMAN QA NOT COMPLETE] CHECK Clip {len(todo)}개를 영상으로 확인한 뒤 "
                         "Manifest review_status를 채우고 team_13부터 다시 실행하세요.")

    print("[Required Human QA Result]")
    print(pd.crosstab(df["qa_status"], dec).to_string())
    print("Required human review: OK\n")

    approved = df[(dec == "PASS") | ((df["qa_status"] == "AUTO_PASS") & (dec == "UNREVIEWED"))].copy()
    if approved.empty:
        raise SystemExit("[중지] 승인된 Clip이 없습니다.")
    approved.to_csv(d["approved"], index=False, encoding="utf-8-sig")

    print(f"Approved clips: {len(approved)} / {len(df)}")
    print("\n[By Subject]")
    print(approved.groupby("subject").size().to_string())
    print("\n[By Action × View]")
    print(approved.groupby(["action_label", "view"]).size().unstack(fill_value=0).to_string())
    print(f"\nSaved: {d['approved']}")


if __name__ == "__main__":
    main()
