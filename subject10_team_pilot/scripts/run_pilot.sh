#!/usr/bin/env bash
# 사전촬영 9개 영상 → YOLO Pose(전체 Frame) → 모캡 GLB와 비교
# 실행 위치: ~/ai_vision/subject10_team_pilot (가상환경 활성화 상태)
#
# 파일 배치
#   data/pilot/videos/Front_A001.mp4, Side_A001.mp4, A001_up.mp4 ... (A006, A008)
#   data/pilot/mocap/261110_B001_T002_SHG.glb, 261110_B006_T001_SHG.glb, 261110_B008_T002_SHG.glb
set -e

declare -A GLB=(
  [A001]=261110_B001_T002_SHG
  [A006]=261110_B006_T001_SHG
  [A008]=261110_B008_T002_SHG
)

for act in A001 A006 A008; do
  for pair in "Front_${act}:front" "Side_${act}:side" "${act}_up:up"; do
    stem=${pair%%:*}; view=${pair##*:}
    video=data/pilot/videos/${stem}.mp4
    [ -f "$video" ] || { echo "skip: $video"; continue; }

    # ① 2일차 코드로 프레임별 Keypoint·Confidence 추출 (영상 전체)
    python scripts/10_pose_video_quality.py --source "$video" --max-frames 0

    # ② 모캡과 관절각도 비교
    python scripts/11_pilot_compare_mocap.py \
      --keypoints reports/pilot/tables/${stem}_keypoints.csv \
      --glb data/pilot/mocap/${GLB[$act]}.glb \
      --view "$view"
  done
done

echo "완료: reports/pilot/pilot_summary.csv 와 *_compare.png 확인"
