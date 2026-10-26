# Subject 9 Machine Vision

OpenCV 기반 제조 머신비전 검사기술을 단계적으로 학습하는 프로젝트입니다.

- 환경: WSL2 Ubuntu + VS Code + Python `.venv`
- 주요 라이브러리: `opencv-python`, `numpy` (세부 버전은 `requirements.txt`)
- 실행 위치: 모든 코드는 **프로젝트 최상위 폴더**에서 `python src/파일명.py`로 실행

```bash
cd ~/ai_vision/subject9_machinevision
source .venv/bin/activate
python src/00_check_env.py   # Environment OK 확인
```

---

## Day 01 — 머신비전 검사문제 설계와 ROI

완성된 OK / NG 검사기를 만드는 것이 아니라,
이미지와 영상에서 **어디를 검사할 것인지 정하고**
정상·불량의 차이를 **측정 가능한 특징으로 바꾸는 방법**을 생각했다.

### 핵심 흐름

```text
검사 문제 확인 → 이미지 읽기 → ROI 설정
→ 여러 이미지에 같은 ROI 적용 → 영상 Frame 처리 → Frame에 ROI 적용
→ 스마트폰 실시간 영상 연결 → 실시간 Frame에 ROI 적용
→ 결과와 판단 기록
```

### 작성한 코드

| 파일 | 역할 |
|---|---|
| `00_check_env.py` | Python · OpenCV · NumPy 환경 확인 |
| `01_read_image.py` | 이미지 한 장 읽기·저장 |
| `02_image_info.py` | 이미지 크기·채널·자료형 확인 |
| `03_roi_crop.py` | 이미지 한 장에서 ROI 설정·저장 |
| `04_batch_roi.py` | 여러 이미지에 동일한 ROI 적용 |
| `05_video_input.py` | 저장 영상을 Frame으로 읽고 30 Frame마다 저장 |
| `06_video_roi.py` | 저장 영상 Frame에 ROI 적용·저장 |
| `07_webcam_input.py` | 스마트폰 IP Webcam 영상 입력 확인 |
| `08_phone_camera_roi.py` | 실시간 영상에서 마우스로 ROI 선택, `cap_on` / `cap_off` 저장 |

### 주요 결과

- `outputs/roi/` — ROI 이미지와 Preview
- `outputs/frames/`, `outputs/video_roi/` — 영상 Frame과 Frame ROI
- `outputs/live_roi/cap_on.jpg`, `cap_off.jpg` — 실시간 ROI 저장 결과
- `reports/day01_problem_definition.md` — 검사 문제 정의와 ROI 판단 근거
- `reports/dataset_roles.csv` — 데이터 폴더별 역할

### 배운 점

- 한 장에서 잘 맞는 ROI ≠ 모든 이미지에서 잘 맞는 ROI
- ROI가 대상을 놓친 것(검사조건 문제)과 ROI 안에 대상이 없는 것(제품 불량)은 구분해야 한다
- 이미지·저장 영상·실시간 카메라는 OpenCV에서 결국 Image / Frame 단위로 처리된다

---

## Day 02 — Image Enhancement와 검사 영상 품질 개선

1일차에 확보한 ROI를 이용하여
촬영조건과 영상 품질이 검사 결과에 어떤 영향을 주는지 확인했다.

### 실습 내용

| 파일 | 내용 |
|---|---|
| `09_color_space.py` | BGR / Gray / HSV 비교 |
| `10_brightness_contrast.py` | Brightness (beta) 비교 |
| `11_gamma.py` | Gamma Correction 0.6 ~ 2.0 비교 |
| `12_clahe.py` | CLAHE clipLimit 1.0 ~ 4.0 비교 |
| `13_filter_compare.py` | Gaussian / Median / Bilateral Filter 비교 |
| `14_sharpen.py` | Blur 생성 후 Sharpening 효과와 한계 확인 |
| `15_threshold_otsu.py` | Threshold 80 · 120 · 160 / Otsu 비교 |
| `16_preprocess_compare.py` | Original · Gaussian · CLAHE + Otsu 종합 비교 |
| `17_day02_challenge.py` | Mini Challenge — 전처리 직접 선택 |

### 핵심 결과

- `outputs/day02/compare/preprocess_compare.png`
- `outputs/day02/challenge/challenge_result.jpg`
- `reports/day02_parameter_table.csv`
- `reports/day02_preprocess_reason.md`

### 오늘의 핵심

전처리 기술을 먼저 선택하는 것이 아니라

```text
문제 확인
→ 필요한 전처리 선택
→ Parameter 비교
→ Threshold / Otsu 결과 확인
→ 사용 여부 결정
```

순서로 판단한다.
전처리를 하지 않는 것이 더 안정적이면 **미사용도 올바른 결론**이고,
강한 반사(Glare)처럼 정보가 이미 손실된 경우는 **촬영환경 개선**이 먼저다.

### 다음 학습

Day 03에서는 Binary 이미지에 Morphology와 Contour를 적용하여
객체의 면적·개수·위치를 측정하고 Rule 기반 OK / NG 판정으로 연결한다.

---

## 폴더 구조

```text
subject9_machinevision/
├── data/        # calibration · train · validation · test · challenge
├── samples/     # 실습 이미지·영상 (Git 제외)
│   └── day02/   # cap_on_reference, dark, low_contrast, noise, blur, glare, challenge
├── outputs/     # 실행 결과 (Git 제외)
├── src/         # 00 ~ 17 실습 코드
├── reports/     # 문제 정의 · 파라미터 표 · 선택 이유
├── roi_temp/    # challenge.jpg ROI 자르기 도구
├── requirements.txt
└── README.md
```
