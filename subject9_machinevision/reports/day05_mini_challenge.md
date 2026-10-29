# Day 05 Mini Challenge — 비타민 음료 로고 검사

## 1. 실험 목적

4일차에는 비타민 음료병의 로고를 SIFT와 Homography로 찾았다. 이번에는 정상 사진과 로고 일부를 가린 사진을 구분할 수 있는 기준을 정하고, 그 기준이 새로운 사진에서도 유지되는지 확인했다.

검사 영역은 `VITAMIN C 1000 mg` 글자 주변이며, 4일차에 만든 `reference_label.jpg`를 사용했다. 로고가 가려지지 않은 사진을 OK, 일부가 가려진 사진을 NG로 정했다.

## 2. 데이터 구성

| 구분 | OK | NG | 합계 |
|---|---:|---:|---:|
| Calibration | 3 | 3 | 6 |
| Test | 2 | 2 | 4 |

Calibration OK는 위치·거리·기울기를 바꿔 촬영했다. NG는 검은 펜으로 로고 일부를 가렸다. Test에서는 새로 촬영한 정상 사진 2장과 명함으로 로고를 가린 사진 2장을 사용했다. Test 사진은 Threshold를 정할 때 사용하지 않았다.

- 기준 이미지: `samples/day05/mini_challenge/reference/reference_label.jpg`
- Calibration: `samples/day05/mini_challenge/calibration/`
- Test: `samples/day05/mini_challenge/test/`

## 3. Calibration 결과

`day05_logo_features.py`의 공통 측정 함수를 이용해 Template Score, SIFT Good Match, Homography Inlier, Inlier Ratio를 측정했다.

| 파일 | 실제 | Template Score | Good Match | Inlier | Inlier Ratio |
|---|---|---:|---:|---:|---:|
| `ok_01.jpg` | OK | 0.4680 | 69 | 59 | 0.8551 |
| `ok_02.jpg` | OK | 0.4026 | 51 | 46 | 0.9020 |
| `ok_03.jpg` | OK | 0.3371 | 56 | 50 | 0.8929 |
| `ng_01.jpg` | NG | 0.3368 | 47 | 37 | 0.7872 |
| `ng_02.jpg` | NG | 0.3742 | 25 | 17 | 0.6800 |
| `ng_03.jpg` | NG | 0.3191 | 22 | 17 | 0.7727 |

Template Score는 `ok_03.jpg`(0.3371)보다 `ng_02.jpg`(0.3742)가 높아서 단독 판정 기준으로 쓰기 어려웠다. 반면 Good Match는 OK가 51~69개, NG가 22~47개였고, Inlier Ratio는 OK가 0.8551~0.9020, NG가 0.6800~0.7872로 두 그룹의 범위가 겹치지 않았다.

## 4. Baseline Rule 결정

Calibration에서 확인한 범위 사이에 기준값을 정했다.

- Homography Found = 1
- SIFT Good Match ≥ 50
- Inlier Ratio ≥ 0.82

세 조건을 모두 만족하면 OK, 하나라도 만족하지 못하면 NG로 판정했다. Good Match 기준 50은 NG 최댓값 47과 OK 최솟값 51 사이에서 정했고, Inlier Ratio 기준 0.82는 NG 최댓값 0.7872와 OK 최솟값 0.8551 사이에서 정했다.

Calibration 6장에서는 TP 3, TN 3, FP 0, FN 0으로 모두 맞혔다. 이 값을 `day05-vitamin-v1.0`으로 저장하고 Test에는 변경하지 않고 사용했다.

## 5. Test 결과

| 파일 | 실제 | 예측 | Good Match | Inlier Ratio | 분류 |
|---|---|---|---:|---:|---|
| `test_ok_01.jpg` | OK | NG | 67 | 0.7612 | FP |
| `test_ok_02.jpg` | OK | NG | 56 | 0.7857 | FP |
| `test_ng_01.jpg` | NG | NG | 36 | 0.6667 | TP |
| `test_ng_02.jpg` | NG | NG | 7 | 0.8571 | TP |

Test 결과는 TP 2, TN 0, FP 2, FN 0이었다.

| 지표 | 결과 |
|---|---:|
| Accuracy | 0.5000 |
| NG Precision | 0.5000 |
| NG Recall | 1.0000 |

## 6. 실패 분석

정상 사진 두 장 모두 Good Match는 50개를 넘었지만 Inlier Ratio가 기준 0.82보다 낮았다. 따라서 두 사진 모두 NG로 판정되었다. 이번 Test에서 발생한 FP 2건은 Inlier Ratio 조건 때문에 생겼다.

`test_ng_02.jpg`는 Inlier Ratio가 0.8571로 기준을 넘었지만 Good Match가 7개뿐이었다. 비율만 봤다면 OK로 판단할 수 있는 값이지만, Good Match 조건을 함께 적용해 NG로 판정되었다.

Calibration에서는 정상과 불량의 값이 분리됐지만 Test 정상 사진의 Inlier Ratio는 Calibration OK 범위보다 낮았다. 촬영 조건 차이가 영향을 줬을 가능성은 있으나, 이번 측정값만으로 정확한 원인을 특정하기는 어렵다.

## 7. 정리

Calibration에서는 6장을 모두 구분했지만 Test 정확도는 50%였다. NG 2장은 놓치지 않았으나 정상 2장도 NG로 판정했다. 따라서 이번 기준은 정상 사진을 통과시키는 데 문제가 있었다.

Test 결과에 맞춰 Threshold를 수정하지 않고 기존 Rule을 유지했다. 이후 개선한다면 정상 촬영 조건을 더 다양하게 확보하고, 기존 Rule과 구분되는 새 버전에서 기준값과 판정 조건을 다시 비교할 필요가 있다.

## 8. 관련 파일

- `src/39_day05_mini_challenge.py` — Calibration 측정
- `src/41_day05_mini_test.py` — 고정 Rule을 이용한 Test
- `configs/day05_vitamin_rules.json` — Baseline Rule v1.0
- `reports/day05_mini_calibration.csv` — Calibration 측정값
- `reports/day05_mini_test_results.csv` — Test 판정 결과
