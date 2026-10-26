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
