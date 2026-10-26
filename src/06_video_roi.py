''' 의사코드
1. sample.mp4 영상을 연다.

2. 검사할 ROI 좌표를 정한다.

3. 영상에서 Frame을 한 장 읽는다.

4. 현재 Frame에 같은 ROI 좌표를 적용한다.

5. 현재 Frame 번호가
   30의 배수인지 확인한다.

6. 30의 배수이면

   전체 Frame
   → ROI 사각형 표시
   → Preview 저장

   ROI 영역
   → 잘라내기
   → ROI 이미지 저장

7. 다음 Frame을 읽는다.

8. 영상이 끝날 때까지 반복한다.

9. 전체 Frame 수와
   저장한 ROI 수를 출력한다.

## 전체 흐름

16. 영상 파일
        ↓
영상 열기
        ↓
Frame 읽기
        ↓
30 Frame마다 이미지 저장
        ↓
같은 Frame에 ROI 적용
        ↓
Preview + ROI 이미지 저장
        ↓
완료
'''

from pathlib import Path

import cv2


video_path = Path("samples/sample.mp4")

output_dir = Path("outputs/video_roi")
output_dir.mkdir(parents=True, exist_ok=True)


# 검사할 ROI 좌표
# 실제 영상에 맞게 수정합니다.
x1 = 420
y1 = 110
x2 = 730
y2 = 300


cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        f"영상을 열 수 없습니다: {video_path}"
    )


frame_count = 0
saved_count = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break


    frame_count += 1

    height, width = frame.shape[:2]


    # ROI 좌표가 영상 밖으로 나가지 않도록 조정
    cx1 = max(0, min(x1, width))
    cx2 = max(0, min(x2, width))

    cy1 = max(0, min(y1, height))
    cy2 = max(0, min(y2, height))


    if cx1 >= cx2 or cy1 >= cy2:
        print("잘못된 ROI 좌표입니다.")
        break


    # 30 Frame마다 결과 저장
    if frame_count % 30 == 0:

        # ROI 자르기
        roi = frame[
            cy1:cy2,
            cx1:cx2
        ]


        # ROI 이미지 저장
        roi_path = (
            output_dir
            / f"frame_{frame_count:05d}_roi.jpg"
        )

        cv2.imwrite(
            str(roi_path),
            roi
        )


        # 원본 Frame 복사
        preview = frame.copy()


        # ROI 위치 표시
        cv2.rectangle(
            preview,
            (cx1, cy1),
            (cx2, cy2),
            (0, 255, 0),
            3
        )


        # ROI 표시된 전체 Frame 저장
        preview_path = (
            output_dir
            / f"frame_{frame_count:05d}_preview.jpg"
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )


        saved_count += 1

        print(
            "저장:",
            roi_path.name
        )


cap.release()


print("전체 Frame:", frame_count)
print("ROI 저장 Frame:", saved_count)