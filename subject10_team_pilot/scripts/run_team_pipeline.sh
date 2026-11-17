#!/usr/bin/env bash
# 팀 프로젝트 전체 실행 (subject10_pose_action 루트에서)
#   bash scripts/run_team_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/team_01_make_manifest.py
python scripts/team_02_check_videos.py
python scripts/team_03_extract_keypoints.py
python scripts/team_04_build_sequences.py
python scripts/team_05_sequence_qa.py
python scripts/team_06_make_split.py
python scripts/team_07_build_reference.py
python scripts/team_08_fit_standard.py
python scripts/team_09_judge.py
