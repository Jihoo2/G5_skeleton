''' 의사코드
1. 입력 이미지 폴더를 정한다.
   → samples

2. 결과를 저장할 폴더를 정한다.
   → outputs/roi

3. 결과 폴더가 없으면 만든다.

4. 사용할 ROI 좌표를 정한다.
   → x1, y1, x2, y2

5. samples 폴더에서
   jpg, jpeg, png 이미지를 모두 찾는다.

6. 찾은 이미지 목록을 정렬한다.

7. 전체 이미지 개수를 출력한다.

8. 이미지가 하나도 없으면 오류를 발생시킨다.

9. 성공적으로 처리한 이미지 개수를 0으로 시작한다.

10. 이미지 목록을 하나씩 반복한다.

    10-1. 현재 이미지를 읽는다.

    10-2. 이미지 읽기에 실패하면
          실패 메시지를 출력하고
          다음 이미지로 넘어간다.

    10-3. 현재 이미지의 가로와 세로 크기를 확인한다.

    10-4. ROI 좌표가 이미지 밖으로 나가지 않도록
          현재 이미지 크기에 맞게 보정한다.

    10-5. ROI 좌표가 잘못되었다면
          실패 메시지를 출력하고
          다음 이미지로 넘어간다.

    10-6. 현재 이미지에서 ROI 영역을 잘라낸다.

    10-7. 잘라낸 ROI 이미지의 저장 경로를 만든다.

    10-8. ROI 위치를 표시한 Preview 이미지의
          저장 경로를 만든다.

    10-9. 잘라낸 ROI 이미지를 저장한다.

    10-10. 원본 이미지를 복사한다.

    10-11. 복사한 이미지에
           ROI 위치를 사각형으로 표시한다.

    10-12. Preview 이미지를 저장한다.

    10-13. 성공 개수를 1 증가시킨다.

    10-14. 어떤 이미지가 처리되었는지 출력한다.

11. 모든 이미지 처리가 끝나면
    성공 개수 / 전체 이미지 개수를 출력한다.
'''

from pathlib import Path

import cv2


input_dir = Path("samples")
output_dir = Path("outputs/roi")
output_dir.mkdir(parents=True, exist_ok=True)

# 자신의 이미지에 맞게 수정합니다.
x1 = 170
y1 = 120
x2 = 470
y2 = 500

extensions = ("*.jpg", "*.jpeg", "*.png")

image_files = []
for pattern in extensions:
    image_files.extend(input_dir.glob(pattern))

image_files = sorted(image_files)

print("전체 이미지 수:", len(image_files))

if not image_files:
    raise RuntimeError(
        "samples 폴더에 jpg/jpeg/png 이미지가 없습니다."
    )

success_count = 0

for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        print("읽기 실패:", image_path.name)
        continue

    height, width = image.shape[:2]

    cx1 = max(0, min(x1, width))
    cx2 = max(0, min(x2, width))
    cy1 = max(0, min(y1, height))
    cy2 = max(0, min(y2, height))

    if cx1 >= cx2 or cy1 >= cy2:
        print("ROI 실패:", image_path.name)
        continue

    roi = image[cy1:cy2, cx1:cx2]

    roi_path = output_dir / f"{image_path.stem}_roi.jpg"
    preview_path = output_dir / f"{image_path.stem}_preview.jpg"

    cv2.imwrite(str(roi_path), roi)

    preview = image.copy()
    cv2.rectangle(
        preview,
        (cx1, cy1),
        (cx2, cy2),
        (0, 255, 0),
        3,
    )
    cv2.imwrite(str(preview_path), preview)

    success_count += 1
    print(image_path.name, "→", roi_path.name)

print("처리 완료:", success_count, "/", len(image_files))