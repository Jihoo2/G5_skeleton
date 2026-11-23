# subject10_team_pilot — 맨몸운동 정석 자세 판정 (team05)

카메라로 찍은 맨몸운동 영상을 **VICON 모션캡처(GLB) 정석 동작**과 비교해서
회차(rep)마다 **OK / NG / CHECK**를 판정하고, NG이면 교정 메시지를 보여 주는 프로그램입니다.

- 교과 10(자세추정·동작인식) 팀 프로젝트 · 팀 저장소 `G5_skeleton/subject10_team_pilot`
- 입력: 업로드 영상, 브라우저 웹캠, 휴대폰 IP Webcam
- 판정: **규칙 기반**(정석 범위를 벗어나면 NG) + 정석 이상 탐지 모델의 **'주의' 표시(판정에는 반영 안 함)**

---

## 1. 판정 대상 동작

| action_id | 동작 | GLB 정석 | 판정에 쓰는 카메라 방향 |
|---|---|---|---|
| A001 | burpee (버피) | `261110_B001_T002_SHG.glb` | 측면 (사선 보류) |
| A004 | lunge (런지) | `261110_B014_T001_SHG.glb` | 측면 (사선은 각도·깊이 규칙 미적용) |
| A008 | squat_jump (스쿼트 점프) | `261110_B008_T002_SHG.glb` | 측면 · 사선 · 정면 |
| A006 | plank_kick_through | `261110_B006_T001_SHG.glb` | 제외 (회차 구분이 어려워 보류) |

카메라 방향은 `front 0° / diag 45° / side 90°`로 구분하고, 영상마다 방향을 기록합니다.
같은 자세라도 카메라 각도에 따라 2D 관절 각도가 달라 보이므로(투영 왜곡) **방향마다 쓰는 지표를 따로 정했습니다.**

### 왜 '정석에서 벗어나면 NG'인가
틀린 자세는 종류가 너무 많아서 NG 데이터를 다 모을 수 없습니다.
그래서 **정석 자세의 범위를 기준으로 잡고, 그 범위를 벗어나면 NG**로 판정합니다.
틀린 자세 데이터가 적기 때문에 분류 모델(Random Forest 등) 대신 규칙 판정을 메인으로 썼습니다.

---

## 2. 판정 규칙 (`configs/team_project.json`)

> 모든 값은 **제안값**이며, 최종 평가 전에 `rules-v1` 태그로 동결합니다.

| 동작 | 지표 | 기준 | 적용 방향 | 의미 |
|---|---|---|---|---|
| 버피 | `body_line_min` | ≥ −0.15 | 측면 | 플랭크 구간에서 엉덩이가 어깨–발목 선보다 몸통 길이의 15% 이상 처지지 않기 (허리 처짐) |
| 버피 | `body_line_max` | ≤ 0.55 | 측면 | 엉덩이가 55% 이상 들리지 않기 (엉덩이 들림) |
| 버피 | `curve_dist` | ≤ 30° | 측면 | 무릎·고관절 움직임 곡선이 GLB 정석과 평균 30° 이상 다르지 않기 |
| 런지 | `knee_min` | 60° ~ 115° | 측면 | 앞무릎 각도 (약 90° 목표) |
| 런지 | `hip_min` | ≥ 60° | 측면 | 앞다리 고관절 각도 |
| 런지 | `depth` | ≤ 0.55 | 측면 | 서 있을 때 다리 높이 대비 내려간 비율 |
| 스쿼트 점프 | `depth` | ≤ 0.55 | 전 방향 | 서 있을 때 다리 높이의 55% 이하까지 앉기 |
| 스쿼트 점프 | `knee_spread_bottom` | ≥ 0.80 | 정면 | 앉은 순간 무릎 간격 ≥ 발목 간격의 80% (무릎이 안으로 무너지지 않기) |

추가 규칙
- **얕은 시도 NG**: 1회로 인정될 만큼 내려가지 않은 동작도 NG (런지는 측면만, 스쿼트 점프는 전 방향)
- **스쿼트 점프 최소 1회 길이 0.4초**: 점프 후 착지하면서 무릎을 굽힌 동작을 1회로 세지 않음
- **런지 앞다리 판별**(`leg_combine: front`): 두 다리 평균이 아니라 굽힌 앞다리만 측정
- **CHECK**: Keypoint 신뢰도(0.5 미만) 프레임이 30% 이상이거나 측정할 수 없는 회차
- **경계 표시**: 기준값 ±2% 안에 든 회차는 화면에 경계 회차로 표시

### 규칙 근거 (val_ext 기준)
- 런지 측면 깊이 0.55: 외부 영상 4명 11회에서 정석 0.41~0.50, 20% 덜 내려가기 0.51~0.61, 40% 덜 내려가기 0.61~0.72 → **정석 오판정 0%, 40% 덜 내려가기 전부 검출**
- 런지 사선은 정석 깊이 범위(0.43~0.78)가 덜 내려가기(0.63~0.77)와 겹쳐서 깊이로 판정하지 않음
- 버피 사선은 팀 정석 영상도 `curve_dist`로 NG가 나와서 보류 (사선 정석 영상이 3명 이상 모이면 재검토)

---

## 3. 전체 흐름

```
[0] 사전 테스트(pilot) — 카메라 방향 정하기
      │
[1] 데이터 준비 — manifest → 영상 점검 → Keypoint 추출 → Sequence·Feature → QA
      │
[2] 데이터 분할 — 사람 단위(Subject-wise) Split · Leakage 검사
      │
[3] 정석 기준 — GLB 기준 곡선 → GLB 증강·보정 → 규칙 기준(standards)
      │
[4] 판정 — 회차 검출 → 규칙 OK/NG/CHECK + 교정 메시지 (+ 이상 탐지 '주의')
      │
[5] 서비스 — 오버레이 영상 · 실시간 판정 · 업로드 분석 웹 · 녹화 재생
```

### [0] 사전 테스트 (pilot)
사전촬영 영상(Front / Side / up)을 GLB와 비교해서 동작마다 판정에 맞는 카메라 방향을 정했습니다.
- `scripts/run_pilot.sh` → `reports/pilot/` (`*_pose.mp4`, `*_keypoints.csv`, `*_compare.png`, `pilot_summary.csv`)
- 기준: `valid_ratio`(Confidence 통과 비율)가 높고 `dtw_mean_diff`(평균 각도 차이)가 작은 방향
- 결론: 무릎·고관절처럼 앞뒤로 움직이는 지표는 측면이 가장 안정적, 무릎 모임은 정면에서만 측정 가능

---

## 4. 스크립트 목록

| 단계 | 스크립트 | 하는 일 |
|---|---|---|
| 데이터 | `team_01_make_manifest.py` | 파일명으로 clip manifest 생성 |
| | `team_02_check_videos.py` | 영상 상태(fps·길이·해상도) 점검 |
| | `team_03_extract_keypoints.py` | YOLO Pose Keypoint 추출 |
| | `team_04_build_sequences.py` | Sequence·Feature 생성 (관절 각도, 다리 높이 비율, 몸통 선, 무릎 간격) |
| | `team_05_sequence_qa.py` | Sequence 품질 검사 |
| | `team_06_make_split.py` | 사람 단위 train/val/test 분할 |
| 정석 기준 | `team_07_build_reference.py` | GLB를 카메라 방향별로 투영한 정석 기준 곡선 |
| | `team_08_fit_standard.py` | 규칙·분포로 판정 기준(standards) 생성 |
| | `team_18_glb_augment.py` | GLB 정석 증강 (카메라 방향·높이·거리, 체형, 속도, YOLO 흔들림) |
| | `team_19_calibrate_glb.py` | GLB 증강 분포를 실제 정석 영상으로 보정 (사람 3명 이상일 때) |
| 판정 | `team_09_judge.py` | 회차별 OK/NG/CHECK 판정·교정 메시지 (`--final`: test 계열 1회 평가) |
| | `team_10_render_overlay.py` | 뼈대 + 회차별 판정 막대를 영상에 그림 |
| | `team_11_ng_sensitivity.py` | NG 검출 민감도 실험 (덜 내려가기 등 변형 비율별) |
| | `team_20_form_model_loso.py` | 정석 이상 탐지 모델 (가중 PCA T²/SPE, LOSO 평가) |
| 5일차 형식 | `team_12_export_day05_keypoints.py` | 95열 Keypoint CSV |
| | `team_13_keypoint_qa.py` | 자동 QA + Human QA |
| | `team_14_build_approved_manifest.py` | Approved Manifest |
| | `team_15_build_subject_split.py` | Subject-wise Split (`configs/day05_subject_split.json`) |
| | `team_16_check_split_leakage.py` | Data Leakage 검사 |
| | `team_17_demo_frame_split_leakage.py` | Random Frame Split과 비교 |
| 실시간 | `team_21_realtime_judge.py` | 실시간 판정 (상태 머신 회차 감지, `--replay-all`로 녹화 판정과 일치율 검증) |
| | `team_22_check_runtime_inputs.py` | 실행 전 입력 파일 점검 |
| | `team_23_plot_rep_signal.py` | 회차 신호 진단 그래프 |
| 일괄 실행 | `run_team_pipeline.sh` | 데이터 → 기준 → 판정 전체 실행 |

### 설정 파일 (`configs/`)
| 파일 | 내용 |
|---|---|
| `team_project.json` | 동작·GLB·방향별 지표·판정 규칙·Split |
| `team_realtime.json` | 실시간 실행 설정 (warmup 1.0s, buffer 20s 등) — 판정 기준이 아님 |
| `team_action_classes.csv` | 동작 ID·이름·설명·GLB |
| `team05_clip_manifest.csv` | 팀 촬영 clip 목록 |
| `team05_ext_sources.csv` | 외부 영상 출처 기록 |
| `team05_rep_labels.csv` | 회차 단위 사람 확인 라벨 (`checked_by`) |
| `day05_subject_split.json` | 5일차 형식 Subject split (team_15 자동 생성) |

---

## 5. 실행 방법

```bash
cd ~/ai_vision/G5_skeleton/subject10_team_pilot
uv venv && source .venv/bin/activate      # 프롬프트: (subject10-team-pilot)
uv sync

# 데이터 → 기준 → 판정 일괄 실행
bash scripts/run_team_pipeline.sh

# 웹 서버 (업로드 분석 + 실시간 판정 + 녹화 재생)
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

### 웹 화면·API (`web/app.py`)
| 경로 | 기능 |
|---|---|
| `/` | 영상 업로드 분석 |
| `POST /api/jobs` → `GET /api/jobs/{job}` | 분석 작업 등록 → 상태 조회 (백그라운드 판정) |
| `/jobs/{job}` | 결과 화면 — 뼈대 캔버스, 회차 타임라인, 측정값 vs 기준, 곡선 vs GLB 정석 |
| `/live` + `WS /ws/live` | 실시간 판정 (브라우저 웹캠 / 휴대폰 IP Webcam, 세로 영상은 서버에서 회전) |
| `/records`, `/records/{id}` | 실시간 녹화 목록·재생 (뼈대, OK/NG 배너, 회차 타임라인) |
| `POST /api/records/{id}/save` | 임시 녹화본 저장 (미저장분은 다음 시작·서버 재시작 때 삭제) |

업로드 분석·실시간·터미널 판정은 **모두 같은 판정 엔진**을 써서 결과가 어긋나지 않게 했습니다.

---

## 6. 검증 방식

| 구분 | 사람 | 용도 |
|---|---|---|
| train | S01, S02 | 기준 생성 |
| val | S03 | 규칙 조정 |
| val_ext | EXT_S01·S02·S04~S15 (14명) | 외부 영상, 규칙 조정·LOSO |
| test / test_ext | (비어 있음) | **규칙 동결 후 `team_09_judge.py --final`로 한 번만 평가** |

- 사람 단위로 나눠서 같은 사람이 학습·평가에 함께 들어가지 않게 했습니다 (5일차 Subject-wise Split).
- 규칙 조정에 한 번이라도 쓴 외부 영상은 test_ext에서 val_ext로 옮겼습니다 (Test 누수 방지).
- 이상 탐지 모델은 사람 하나씩 빼고 평가하는 LOSO로 검증합니다.

### 주요 결과
| 항목 | 결과 |
|---|---|
| 스쿼트 점프 착지 동작 오검출 수정 (`min_rep_s` 0.4) | 정석 오판정 3회 → 1회, 실시간 vs 녹화 판정 일치율 97% |
| 런지 얕은 시도 NG를 측면에만 적용 | 실시간에만 잡히는 회차 0개 |
| EXT_S05 회차 단위 라벨 (사람 확인) | 정석 오판정 0/50, 틀린 자세 검출 1/2 |
| 런지 측면 깊이 0.55 | 정석 오판정 0%, 40% 덜 내려가기 전부 검출 |

> 최종 성능은 test_ext 평가 후 추가합니다.

### 관련 문서
- 오류 원인 추적 사례: [`docs/failure_cases.md`](docs/failure_cases.md)
- 외부 영상 출처: `configs/team05_ext_sources.csv` · 제외 기록: `configs/team05_ext_excluded.csv` (외부 영상은 학습·검증에만 쓰고 저장소·발표 자료에 올리지 않음)
- EXT_S03은 점프 없는 버피라 제외해서 번호가 비어 있음

---

## 7. 수업 일차 ↔ 팀 코드

| 교과 10 일차 | 수업 내용 | 팀 프로젝트 적용 |
|---|---|---|
| 1~2일차 | YOLO Pose, Keypoint·Confidence | `team_02`, `team_03`, pilot (`run_pilot.sh`) — Confidence 0.5 기준 |
| 3일차 | 관절 각도, 정규화, FrameFeature | `team_04` — 관절 각도 + 팀 추가 지표(다리 높이 비율, 몸통 선, 무릎 간격), 측면은 어깨 폭 대신 몸통 길이로 정규화 |
| 4일차 | ActionClip, Sequence Dataset | `team_01`, `team_04`, `team_05` — Feature 명세·데이터 준비 보고서 |
| 5일차 | Keypoint Dataset QA, Subject-wise Split | `team_12`~`team_17`, `team_06` |
| 6일차 | Action Recognition Baseline 성능평가 | 분류 모델 대신 규칙 판정 `team_07`~`team_09`, `team_11` + 이상 탐지 `team_18`~`team_20` |
| 7일차 | Sliding Window, IP Webcam 실시간 인식 | `team_21`~`team_23`, `configs/team_realtime.json`, 웹 `/live` |
| 8일차 | 다중 인물 Pose Tracking | 미적용 — 1인 판정 기준 (한계 참고) |

**6일차를 다르게 한 이유**: 수업은 동작 *종류*를 맞히는 분류(Random Forest)이고,
팀 과제는 같은 동작 안에서 *자세가 맞는지*를 판정하는 문제입니다. 틀린 자세 데이터가 적어서
정석 범위 규칙을 메인으로 쓰고, 이상 탐지 모델은 보조 '주의' 표시로만 붙였습니다.

---

## 8. 알려진 한계

1. **경계 회차 불일치**: 기준값 근처(예: 깊이 0.547·0.552 vs 기준 0.55) 회차는 사람 판단과 어긋날 수 있습니다. ±2% 경계 표시로 사용자에게 알립니다.
2. **사선 런지 회차 검출**: 사선 영상은 깊이 신호가 넓게 퍼져 회차 인정선 근처에서 검출이 흔들릴 수 있습니다.
3. **회차 라벨이 회차 번호 기준**: `team05_rep_labels.csv`는 시간이 아닌 회차 번호로 저장해서, 회차 검출 규칙이 바뀌면 라벨이 다른 회차에 붙을 수 있습니다.
4. **방향 제한**: 버피는 측면만, 무릎 모임은 정면에서만 판정합니다.
5. **1인 기준**: 화면에 여러 명이 있으면 판정하지 않습니다 (8일차 Tracking 미적용).
6. **규칙은 제안값**: 외부 영상 수가 적어서, test_ext 최종 평가 전까지는 일반화 성능이 확인되지 않았습니다.
7. **A006 제외**: 플랭크 킥스루는 회차 구분이 어려워 판정 대상에서 뺐습니다.
