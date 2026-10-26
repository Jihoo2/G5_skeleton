''' 전체 코드흐름
이미지 파일 경로 확인
        ↓
이미지 읽기
        ↓
이미지 크기 확인
        ↓
ROI 좌표 설정
        ↓
좌표가 이미지 범위를 벗어나는지 확인
        ↓
ROI 이미지 자르기
        ↓
ROI 이미지 직접 확인
        ↓
원본 이미지에 ROI 사각형 표시
        ↓
이미지 파일로 저장
        ↓
최종 코드 전체 흐름 이해

## 의사코드
[BLOCK 1 — 입력(Input) + 예외 처리]

1. 입력 이미지 경로를 정한다.

2. ROI 결과를 저장할 폴더 경로를 정한다.

3. 이미지를 읽는다.

4. 이미지가 없으면 오류를 발생시킨다.


[BLOCK 2 — 이미지 정보 확인 + ROI 설정]

5. 이미지의 가로와 세로 크기를 확인한다.

6. 자르고 싶은 ROI 좌표를 정한다.

   왼쪽 위    : (x1, y1)
   오른쪽 아래 : (x2, y2)


[BLOCK 3 — 좌표 보정 + 예외 처리]

7. ROI 좌표가 이미지 바깥으로 나가지 않도록 보정한다.

8. 좌표가 잘못되었는지 확인한다.

   만약 x1 >= x2 또는 y1 >= y2 이면
   오류를 발생시킨다.


[BLOCK 4 — 핵심 로직(ROI 처리)]

9. 이미지에서 ROI 영역만 잘라낸다.


[BLOCK 5 — 출력 및 ROI 저장]

10. 결과 저장 폴더가 없으면 만든다.

11. 잘라낸 ROI 이미지를 저장한다.


[BLOCK 6 — Preview 생성 및 저장]

12. 원본 이미지를 복사한다.

13. 복사한 이미지에 ROI 위치를 사각형으로 표시한다.

14. ROI 위치가 표시된 Preview 이미지를 저장한다.


[BLOCK 7 — 실행 결과 확인]

15. 다음 정보를 화면에 출력한다.

    - 원본 이미지 크기
    - ROI 좌표
    - ROI 크기
    - ROI 저장 경로
    - Preview 저장 경로
'''

from pathlib import Path

import cv2


image_path = Path("samples/sample01.jpg")
output_dir = Path("outputs/roi")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

height, width = image.shape[:2]

# 자신의 이미지에 맞게 수정합니다.
x1 = 170
y1 = 120
x2 = 470
y2 = 450

# 좌표가 이미지 범위를 벗어나지 않도록 제한합니다.
x1 = max(0, min(x1, width))
x2 = max(0, min(x2, width))
y1 = max(0, min(y1, height))
y2 = max(0, min(y2, height))

if x1 >= x2 or y1 >= y2:
    raise ValueError(
        f"잘못된 ROI 좌표입니다: {(x1, y1, x2, y2)}"
    )

roi = image[y1:y2, x1:x2]

output_dir.mkdir(parents=True, exist_ok=True)

roi_path = output_dir / "sample01_roi.jpg"
preview_path = output_dir / "sample01_roi_preview.jpg"

# 잘라낸 ROI 저장
cv2.imwrite(str(roi_path), roi)

# 원본에서 ROI 위치를 확인할 Preview 저장
preview = image.copy()
cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.imwrite(str(preview_path), preview)

print("원본 크기 :", image.shape)
print("ROI 좌표  :", (x1, y1, x2, y2))
print("ROI 크기  :", roi.shape)
print("ROI 저장  :", roi_path)
print("Preview   :", preview_path)