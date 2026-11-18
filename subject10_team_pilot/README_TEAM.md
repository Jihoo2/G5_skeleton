# 팀 프로젝트 add-on — 맨몸운동 정석 자세 판정

`~/ai_vision/subject10_pose_action` (교과 10 수업 프로젝트) 위에 **덮어쓰는** 추가 파일입니다.
수업 파일(`src/common.py`, `src/pose_utils.py`, `src/feature_utils.py`)은 건드리지 않고 import만 합니다.
→ 3일차 `src/feature_utils.py` 가 있어야 동작합니다.

## 추가되는 파일
```
configs/team_project.json         팀 번호·FPS·Confidence·각도·동작·기준 설정·split
configs/team_action_classes.csv   동작 목록 (A001, A006, A008)
src/glb_utils.py                  GLB 뼈대 계산 (VICON → 3D 관절 위치)
src/team_feature_utils.py         팀 Feature·1회 구간·측정값·교정 메시지
scripts/team_01 ~ team_09, run_team_pipeline.sh
docs/교과10_1-3일차_정리.md, docs/촬영요청서.md
data/team_projects/team05/{rgb/videos, ext/videos, vicon}
```

## 실행 순서
| 순서 | 스크립트 | 교과 대응 | 결과 |
|---|---|---|---|
| 1 | `team_01_make_manifest.py` | 4일차 Clip Manifest | `configs/team05_clip_manifest.csv` |
| 2 | `team_02_check_videos.py` | 4일차 영상 점검 | `tables/video_inventory.csv` |
| 3 | `team_03_extract_keypoints.py` | 1·2일차 YOLO Pose | `keypoints/<clip_id>_keypoints.csv` |
| 4 | `team_04_build_sequences.py` | 3일차 Feature + 4일차 Sequence | `sequences/<clip_id>_sequence.csv` |
| 5 | `team_05_sequence_qa.py` | 4·5일차 QA | `tables/sequence_qa.csv` → **Manifest의 review_status 직접 기입** |
| 6 | `team_06_make_split.py` | 5일차 Subject-wise Split | Manifest split 열(train·val·val_ext·test·test_ext), `tables/dataset_split.csv` |
| 7 | `team_07_build_reference.py` | VICON 별도 실험 | `reference/<동작>_<각도>_template.json` |
| 8 | `team_08_fit_standard.py` | (팀) 정석 허용 범위 | `standards/<동작>_<각도>.json` |
| 9 | `team_09_judge.py` | 6일차 평가 자리 | `judgments/rep_judgments.csv`, `tables/judgment_summary.csv` (기본 val·val_ext, test 계열은 `--final`) |
| 10 | `team_10_render_overlay.py --clip <id>` | 10일차 실패 분석 | `overlays/<clip_id>_overlay.mp4` (뼈대 + 1회 판정 표시, 선택 실행) |
| 12 | `team_12_export_day05_keypoints.py` | 5일차 26번 | `day05/keypoints/<subject>/<action>/<clip>.csv` (1행=1Frame, 95열), `day05/manifests/keypoint_dataset_manifest.csv` |
| 13 | `team_13_keypoint_qa.py` | 5일차 27번 | `day05/qa/keypoint_clip_qa.csv`, `keypoint_clip_review.csv` (human_decision ← Manifest review_status) |
| 14 | `team_14_build_approved_manifest.py` | 5일차 28·29번 | `day05/manifests/approved_clip_manifest.csv` |
| 15 | `team_15_build_subject_split.py` | 5일차 30번 | `configs/day05_subject_split.json`(자동 생성), `day05/splits/{train,val,val_ext,test,test_ext}.csv`, `day05/tables/split_coverage.csv` |
| 16 | `team_16_check_split_leakage.py` | 5일차 31번 | Subject·Clip·원본 영상 overlap 검사 (FAIL 시 종료 코드 1) |
| 17 | `team_17_demo_frame_split_leakage.py` | 5일차 32번 | Random Frame Split 비교 (`day05/tables/random_frame_split_demo.csv`) |
| 18 | `team_18_glb_augment.py` | (포트폴리오) GLB 메인 | `glb_aug/<action>_reps.csv·.npz` — 카메라·체형·속도·YOLO 흔들림을 바꾼 GLB 정석 1회 동작 |
| 19 | `team_19_calibrate_glb.py` | (포트폴리오) 영상 보정 | `standards_glb/`, `tables/glb_calibration.csv` → `team_09_judge.py --standards glb` |
| 20 | `team_20_form_model_loso.py` | (포트폴리오) 정석 이상 탐지 모델 | 실제 정석(train+val_ext) 중심 + GLB 증강 보충, 사람 단위 LOSO 평가 → `tables/form_model_loso_summary.csv`, `models/form_model_<action>.npz` |
| 21 | `team_21_realtime_judge.py` | (포트폴리오) 실시간 판정 | 웹캠·IP Webcam·영상 → 1회마다 OK/NG + 교정 메시지 화면 표시, `--replay-all` 로 녹화 판정과 일치율 검증 |

5일차 형식 한 번에: `bash scripts/run_day05.sh` → 결과 폴더 `reports/team_projects/team05/day05/`

```bash
cd ~/ai_vision/subject10_pose_action
source .venv/bin/activate
cp ~/Downloads/vicon/261110_B00{1,6,8}_*.glb data/team_projects/team05/vicon/   # GLB 위치
bash scripts/run_team_pipeline.sh
```
결과 폴더: `reports/team_projects/team05/`

## 판정 규칙
- 정석 영상(train, form_label=OK)의 1회 구간 측정값 → `평균 ± max(2.5·표준편차, 최소폭)` 이 허용 범위.
- 범위 밖 → **NG + 교정 메시지**, 관절이 잘 안 보이면 **CHECK**(자세 문제로 판단하지 않음).
- **각도별 판정 항목** (`configs/team_project.json` → `actions.*.metrics_by_view`)
  | 동작 | 측면·사선 | 정면 |
  |---|---|---|
  | A001 버피 | 깊이, 고관절, 몸통 일직선, 동작 흐름 | 판정 안 함(CHECK) |
  | A006 | 무릎, 고관절, 동작 흐름 | 판정 안 함(CHECK) |
  | A008 스쿼트 점프 | 깊이, 무릎, 고관절, 동작 흐름 | 깊이, **무릎 모임(외반)** |
  - 무릎·고관절 굽힘과 몸통 일직선은 몸 옆면에서 일어나는 움직임이라 정면에서는 원근으로 찌그러짐 → 정면에서는 쓰지 않음.
  - 대신 정면에서만 보이는 "무릎이 안쪽으로 모이는지"(앉은 순간 무릎 간격 / 발목 간격)를 판정.
- 1회 구간은 서 있을 때 대비 85% 아래로 내려간 구간 중 최저점이 80% 아래인 것 → 얕게 앉아도 1회로 잡히고 깊이로 NG 판정.
- 그 각도에서 정석끼리도 크게 흔들리는 항목은 `unreliable` 로 판정에서 자동 제외.
- 팀 영상이 없는 각도는 GLB 값만으로는 판정하지 않음 (`judge_glb_only=false`).
- `k_std`, `min_halfwidth`, `max_halfwidth` 는 val 결과의 오판정률(false NG)을 보고 조정 — 한 번에 한 값만 바꾸기.
- 일부러 벗어난 자세 영상이 몇 개 생기면 Manifest의 `form_label` 을 `NG` 로 적으면 검출률도 같이 나옵니다(필수 아님).

## 주의
- 영상·keypoint·결과 CSV는 `.gitignore` 대상. 얼굴이 나오는 영상·외부 영상은 GitHub에 올리지 않기.
- `team_03` 은 YOLO(torch)가 필요, `team_04~09` 는 numpy/pandas만 사용.
- 파일 이름은 수업 스크립트(11_, 12_ …)와 겹치지 않도록 `team_` 접두어 사용.
