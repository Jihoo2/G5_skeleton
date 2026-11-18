#!/usr/bin/env bash
# 교과 5일차 형식 산출물 만들기 (team_03 Keypoint 추출이 끝난 상태에서)
#   bash scripts/run_day05.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/team_02_check_videos.py             # 25번: 영상 검사 (해상도·Frame 수)
python scripts/team_12_export_day05_keypoints.py   # 26번: 1행=1Frame 95열 Keypoint CSV + Manifest
python scripts/team_13_keypoint_qa.py              # 27번: 자동 QA + Human QA 작업표
python scripts/team_14_build_approved_manifest.py  # 28·29번: Human QA 검사 + Approved Manifest
python scripts/team_15_build_subject_split.py      # 30번: Subject-wise Split CSV
python scripts/team_16_check_split_leakage.py      # 31번: Leakage 검사 (FAIL이면 여기서 멈춤)
python scripts/team_17_demo_frame_split_leakage.py # 32번: Random Frame Split 비교 실험
