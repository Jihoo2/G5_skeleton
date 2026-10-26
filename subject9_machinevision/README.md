# Subject 9 Machine Vision — 김지후 (jihoo2/subject9)

OpenCV 기반 제조 머신비전 검사기술을 단계적으로 학습한 실습 기록입니다.
검사 대상은 **병뚜껑 유무(OK / NG)**이며, 규칙 기반(Rule Vision)으로
"사람 눈에 보이는 차이 → 컴퓨터가 측정할 수 있는 숫자"로 바꾸는 과정을 다룹니다.

---

## 환경

- WSL2 Ubuntu + VS Code
- Python 가상환경 `.venv`, OpenCV(`opencv-python`), NumPy
- 모든 코드는 **이 폴더(`subject9_machinevision/`)에서** 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/00_check_env.py      # Environment OK 확인
```

## 폴더 구조

```text
subject9_machinevision/
├── data/        # calibration · train · validation · test · challenge (Git 제외)
├── samples/     # 실습 이미지·영상, samples/day02/ 문제 이미지 (Git 제외)
├── outputs/     # 실행 결과 (Git 제외)
├── src/         # 실습 코드 00 ~ 17
├── roi_temp/    # challenge.jpg ROI 자르기 도구
├── reports/     # 문제 정의 · 파라미터 표 · 판단 기록
├── requirements.txt
└── README.md
```

> 데이터·결과 이미지는 `.gitignore`로 제외되어 있어 코드와 기록만 저장소에 올라갑니다.

---

## Day 01 — 검사문제 설계와 ROI

어디를 검사할지 정하고(ROI), 정상·불량 차이를 어떤 숫자로 측정할지 생각하는 단계.

| 파일 | 역할 |
|---|---|
| `00_check_env.py` | Python · OpenCV · NumPy 환경 확인 |
| `01_read_image.py` | 이미지 한 장 읽기·저장 |
| `02_image_info.py` | Width · Height · Channel 확인 (`shape = (H, W, C)`) |
| `03_roi_crop.py` | 이미지 한 장에서 ROI 설정·저장 + Preview |
| `04_batch_roi.py` | 여러 이미지에 같은 ROI 적용 → 실패 조건 발견 |
| `05_video_input.py` | 저장 영상을 Frame으로 읽고 30 Frame마다 저장 |
| `06_video_roi.py` | 영상 Frame에 같은 ROI 적용 |
| `07_webcam_input.py` | 스마트폰 IP Webcam 실시간 영상 입력 |
| `08_phone_camera_roi.py` | 실시간 영상에서 마우스로 ROI 선택, `cap_on` / `cap_off` 저장 |

**핵심**
- 좌표는 `(x, y)`, 슬라이싱은 `image[y1:y2, x1:x2]`
- 한 장에 딱 맞는 ROI ≠ 모든 이미지에 맞는 ROI → 위치 변화는 허용하고 배경은 줄이는 범위 선택
- "ROI가 제품을 놓침(검사조건 문제)"과 "ROI는 맞는데 뚜껑이 없음(제품 불량)"을 구분
- 이미지 · 저장 영상 · 실시간 영상 모두 결국 **Frame 단위**로 같은 처리를 적용

---

## Day 02 — Image Enhancement와 검사 이미지 품질 개선

ROI가 컴퓨터가 검사하기 좋은 상태인지 확인하고,
**문제를 먼저 정의 → 필요한 전처리만 선택 → Otsu 분리 결과로 검증**하는 단계.

| 파일 | 역할 | 결과 위치 |
|---|---|---|
| `09_color_space.py` | BGR · Gray · H/S/V 비교 → 병뚜껑 유무는 **Gray 중심** | `outputs/day02/color` |
| `10_brightness_contrast.py` | `dark.jpg` 생성, beta 0/20/50/80 비교 | `outputs/day02/brightness` |
| `11_gamma.py` | Gamma 0.6 ~ 2.0 비교 (이 구현은 gamma > 1 → 밝아짐) | `outputs/day02/gamma` |
| `12_clahe.py` | `low_contrast.jpg` 생성, CLAHE clipLimit 1~4 비교 | `outputs/day02/clahe` |
| `13_filter_compare.py` | `noise.jpg` 생성, Gaussian · Median · Bilateral 비교 | `outputs/day02/filter` |
| `14_sharpen.py` | `blur.jpg` 생성, Sharpening 효과와 한계 | `outputs/day02/sharpen` |
| `15_threshold_otsu.py` | Threshold 80/120/160 vs Otsu | `outputs/day02/threshold` |
| `16_preprocess_compare.py` | Original / Gaussian / CLAHE + Otsu 2×2 비교 | `outputs/day02/compare` |
| `17_day02_challenge.py` | Mini Challenge — 후보 6개 비교 후 최종 선택 | `outputs/day02/challenge` |
| `roi_temp/crop_challenge.py` | 원본 `challenge.jpg`에서 병뚜껑 ROI를 마우스로 잘라 저장 | `samples/day02/challenge.jpg` |

### Mini Challenge 결과 (`17_day02_challenge.py`)

**문제** — 흰 병뚜껑과 밝은 회색 배경의 밝기 차이가 작고 경계가 약간 흐림
→ Otsu에서 배경이 함께 선택될 수 있음

| 후보 | Otsu | white % | 판단 |
|---|---:|---:|---|
| original | 109.0 | 83.7% | 기준 — 아래 링에 검은 얼룩 |
| **clahe_1.0** | **105.0** | **89.1%** | **최종 선택** — 뚜껑·링이 끊기지 않고 얼룩 최소 |
| clahe_2.0 | 104.0 | 88.8% | 점 Noise 증가 |
| clahe_3.0 | 128.0 | 57.4% | ❌ 표면 Texture가 검출되어 뚜껑 몸통이 깨짐 |
| gamma_0.6 | 68.0 | 75.0% | ❌ 뚜껑 왼쪽 위가 검게 뚫림 |
| gamma_0.8 | 89.0 | 82.8% | 개선 효과 미미 |

**결론 — 3일차 추가 확인**
- CLAHE 1.0으로 Otsu 결과는 깨끗해졌지만, 이 ROI는 밝기 분포가
  "어두운 틈·병목" vs "밝은 뚜껑+배경"으로 나뉘어 **뚜껑과 배경은 전처리만으로 분리되지 않음**
- 흰 제품을 밝은 배경에서 찍은 **촬영환경 문제** → 어두운 배경 사용 검토
- 뚜껑이 있을 때만 생기는 **링 사이 검정 띠**를 Day 03에서 Contour로 측정하는 방법 검토

### 기록

- `reports/day02_parameter_table.csv` — 파라미터별 좋아진 점 · 나빠진 점 · 사용 여부
- `reports/day02_preprocess_reason.md` — 선택 이유, 실패한 시도, Glare 사례

**Day 02 핵심** — 전처리는 이미지를 예쁘게 만드는 것이 아니라
다음 검사 단계(Threshold → Contour)가 안정적으로 동작하도록 입력을 준비하는 것.
도움이 되지 않으면 **사용하지 않는 것도 올바른 판단**이다.

---

## 일차별 진행

| 일차 | 주제 | 파일 | 상태 |
|---|---|---|---|
| Day 01 | 검사문제 설계 · ROI · 영상/실시간 입력 | `00 ~ 08` | ✅ |
| Day 02 | Image Enhancement · Threshold / Otsu | `09 ~ 17` | ✅ |
| Day 03 | Morphology · Contour · Rule 판정 | `18 ~ 23` | ⬜ |
| Day 04 | GrabCut · Template · Feature Matching · Homography | `24 ~ 32` | ⬜ |
| Day 05 | Calibration · Rule Freeze · Test | `33 ~ 39` | ⬜ |
| Day 06 | OCR | `40 ~ 49` | ⬜ |
| Day 07 | Anomaly Detection | `50 ~ 61` | ⬜ |
| Day 08 | One-shot / Few-shot | `62 ~ 69` | ⬜ |
| Day 09 | 재현성 · Experiment · Hybrid | `70 ~ 79` | ⬜ |
| Day 10 | 서비스화 (FastAPI · Dashboard) | `80 ~ 83` | ⬜ |
