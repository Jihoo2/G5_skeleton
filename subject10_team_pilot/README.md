# subject10_team_pilot — VICON 모캡 vs 웹캠 사전 테스트

## 목적
사전촬영 영상(Front / Side / up)을 VICON 모캡(GLB)과 비교해서
동작마다 **어느 카메라 방향이 판정에 적합한지** 정합니다.

## 실행 순서
```bash
cd ~/ai_vision/subject10_team_pilot
uv venv
source .venv/bin/activate
uv sync
bash scripts/run_pilot.sh
```

## 파일 배치
- `data/pilot/videos/` : Front_A001.mp4, Side_A001.mp4, A001_up.mp4 … (A006, A008)
- `data/pilot/mocap/`  : 261110_B001_T002_SHG.glb, 261110_B006_T001_SHG.glb, 261110_B008_T002_SHG.glb

## 결과
- `reports/pilot/videos/*_pose.mp4`      : Skeleton이 그려진 영상
- `reports/pilot/tables/*_keypoints.csv` : 영상별 Keypoint·Confidence
- `reports/pilot/*_compare.png`          : 관절각도 모캡 vs 웹캠 그래프
- `reports/pilot/pilot_summary.csv`      : valid_ratio(Confidence 통과 비율), dtw_mean_diff(평균 각도 차이)

## 판단 기준
동작별로 valid_ratio가 높고 dtw_mean_diff가 작은 방향 → 본촬영 카메라 방향
