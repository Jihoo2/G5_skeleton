# Subject 9 Machine Vision

OpenCV 기반 제조 머신비전 검사기술을 10일간 단계적으로 학습하는 교과 폴더입니다.
`main`에는 폴더 구조만 있고, 실습 내용은 팀원별 브랜치에 있습니다.

## 환경 준비

```bash
cd subject9_machinevision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install opencv-python numpy
python -m pip freeze > requirements.txt
```

모든 코드는 **이 폴더(`subject9_machinevision/`)에서** `python src/파일명.py`로 실행합니다.

## 폴더 구조

```text
subject9_machinevision/
├── data/          # calibration · train · validation · test · challenge (Git 제외)
├── samples/       # 실습 이미지·영상 (Git 제외)
├── outputs/       # 실행 결과 (Git 제외)
├── src/           # 실습 코드 00 ~
├── reports/       # 문제 정의 · 파라미터 표 · 판단 기록
├── .gitignore
└── README.md
```

## 일차별 구성

| 일차 | 주제 | 파일 |
|---|---|---|
| Day 01 | 검사문제 설계 · ROI · 영상/실시간 입력 | `00 ~ 08` |
| Day 02 | Image Enhancement · Threshold / Otsu | `09 ~ 17` |
| Day 03 | Morphology · Contour · Rule 판정 | `18 ~ 23` |
| Day 04 | GrabCut · Template · Feature Matching · Homography | `24 ~ 32` |
| Day 05 | Calibration · Rule Freeze · Test | `33 ~ 39` |
| Day 06 | OCR | `40 ~ 49` |
| Day 07 | Anomaly Detection | `50 ~ 61` |
| Day 08 | One-shot / Few-shot | `62 ~ 69` |
| Day 09 | 재현성 · Experiment · Hybrid | `70 ~ 79` |
| Day 10 | 서비스화 (FastAPI · Dashboard) | `80 ~ 83` |

## Day 01 — 검사 문제 정의와 ROI 설정

병뚜껑 유무를 검사 대상으로 정하고, 뚜껑이 있는 상태를 OK, 없는 상태를 NG로 구분했다. 먼저 정지 이미지에서 ROI를 정한 뒤 여러 이미지와 영상에 같은 검사 영역을 적용해 보았다.

- `03_roi_crop.py`에서는 `(170, 120, 470, 450)` 영역을 사용했다. `04_batch_roi.py`에서는 병 입구까지 포함되도록 아래쪽 경계를 500으로 늘려 5개 이미지에 적용했다.
- `05_video_input.py`, `06_video_roi.py`로 동영상 프레임을 읽고 ROI를 추출했다. 영상에서는 병이 움직이면 고정된 ROI와 실제 병뚜껑 위치가 어긋날 수 있다는 점을 확인했다.
- `08_phone_camera_roi.py`에서는 스마트폰 실시간 영상에서 직접 ROI를 지정하고 `cap_on.jpg`, `cap_off.jpg`를 저장했다. 이 사진을 2일차 기준 이미지로 활용했다.

병뚜껑을 보려면 병 전체보다 상단의 뚜껑과 병 입구가 함께 들어오는 영역이 중요했다. 다만 1일차에는 ROI를 확보하는 데 집중했으며, 자동 OK/NG 판정 기준은 만들지 않았다.

- 코드: `src/00_check_env.py` ~ `src/08_phone_camera_roi.py`
- 보고서: `reports/day01_problem_definition.md`, `reports/dataset_roles.csv`

## Day 02 — 영상 전처리 및 Otsu 비교

1일차에서 정한 ROI를 이용해 Color Space, 밝기·대비 조정, Gamma, CLAHE, Gaussian/Median/Bilateral Filter, Sharpening, Threshold와 Otsu를 실습했다. 파라미터가 바뀌면 이미지의 선명도와 밝기는 달라지지만, 그것이 곧바로 객체 분리 개선을 의미하지는 않았다.

Mini Challenge에서는 밝은 병뚜껑과 회색 배경의 명도가 비슷해 Otsu만으로는 분리가 잘되지 않는 사진을 사용했다. 아래 조건으로 비교했다.

| 방법 | 비교한 조건 | 관찰 |
|---|---|---|
| CLAHE | `clipLimit=1.0/2.0/3.0` | 3.0에서는 표면과 배경 잡음이 지나치게 강조됨 |
| Gamma | `1.2/1.5` | 전체 밝기는 달라졌지만 뚜껑과 배경의 분리는 개선되지 않음 |
| Gaussian | `5×5` | 작은 잡음은 줄었으나 밝은 객체와 배경을 구분하는 문제는 남음 |

최종적으로 `17_day02_challenge.py`에는 **`result = image.copy()`**를 적용해 전처리를 사용하지 않았다. 전처리 없이도 분리가 잘됐다는 뜻이 아니라, 시험한 방법들이 분리 문제를 충분히 개선하지 못했기 때문이다. Gaussian 5×5는 이후 형태 측정 단계에서 다시 비교할 후보로 남겼다.

- 코드: `src/09_color_space.py` ~ `src/17_day02_challenge.py`
- 보고서: `reports/day02_parameter_table.csv`, `reports/day02_preprocess_reason.md`

## Day 03 — 진행 상태

3일차의 Morphology, Contour, Rule 판정은 이번 정리에서 완료한 실습으로 기록하지 않았다. 별도로 보완할 예정이다.

## Day 04 — 객체 분리 및 이미지 매칭

광천김 포장지로 GrabCut, MeanShift, Template Matching, ORB, SIFT, Homography를 실습했다. GrabCut에서는 초기 사각형 크기에 따른 전경 분리 결과를, MeanShift에서는 `sp`, `sr` 값에 따른 색상 단순화 결과를 비교했다. 이후 같은 로고에 대해 Template Matching의 Score와 위치, ORB·SIFT의 Good Match, Homography의 Inlier를 확인했다.

기본 실습에서는 `24_day04_prepare.py`로 정면 원본 한 장에서 `normal`, `shift`, `rot15`, `scale90` 이미지를 생성했다. 회전하거나 크기가 달라졌을 때 Template Matching의 결과가 달라지는 점을 확인했고, ORB·SIFT와 Homography 결과도 비교했다.

### Mini Challenge

Mini Challenge에서는 비타민 음료병을 새로 촬영해 `VITAMIN C 1000 mg` 글자 영역을 찾았다. 실제 촬영에서는 위치뿐 아니라 회전·거리·반사에 따라 글자의 크기와 방향도 달라지므로 **SIFT + RANSAC + Homography**를 선택했다.

| 사진 | Good Match | Inlier | Inlier Ratio | 검출 결과 |
| --- | ---: | ---: | ---: | --- |
| `shift.jpg` | 44 | 32 | 72.73% | 정상 |
| `rotate.jpg` | 73 | 60 | 82.19% | 정상 |
| `distance.jpg` | 48 | 41 | 85.42% | 정상 |
| `tilt_glare.jpg` | 63 | 61 | 96.83% | 정상 |

네 조건 모두 결과 이미지의 녹색 사각형이 실제 글자 영역을 감싸고 있었다. 다만 같은 제품의 정상 로고만 검사했기 때문에 잘못된 로고에 대한 오검출 여부는 확인하지 못했다. 최종 OK/NG 기준은 5일차 Calibration에서 다룬다.

### 실행 및 결과 파일

프로젝트 폴더에서 다음 명령어를 실행한다.

```bash
python src/32_day04_challenge.py
```

실행 후 `full_front.jpg`에서 ROI를 선택하고 `SPACE` 또는 `ENTER`로 확정한다.

- 코드: `src/24_day04_prepare.py` ~ `src/32_day04_challenge.py`
- 기본 실습 비교표: `reports/day04_template_results.csv`, `day04_orb_results.csv`, `day04_sift_results.csv`, `day04_homography_results.csv`, `day04_rotation_compare.csv`
- Mini Challenge 결과: `reports/day04_challenge_results.csv`
- Mini Challenge 분석: `reports/day04_method_selection.md`
- 촬영 사진: `samples/day04/my_challenge/`
- 결과 이미지: `outputs/day04/my_challenge/`
