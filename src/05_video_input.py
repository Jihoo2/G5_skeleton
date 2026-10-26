''' 의사코드
1. 사용할 영상 파일의 위치를 정한다.
   → samples/sample.mp4

2. Frame 이미지를 저장할 폴더를 정한다.
   → outputs/frames

3. 저장 폴더가 없으면 만든다.

4. OpenCV로 영상을 연다.

5. 영상이 정상적으로 열리지 않으면
   오류를 발생시키고 종료한다.

6. 영상의 정보를 확인한다.
   → FPS
   → 가로 크기
   → 세로 크기

7. 전체 Frame 개수를 세기 위한 값을 0으로 시작한다.

8. 저장한 Frame 개수를 세기 위한 값을 0으로 시작한다.

9. 영상을 처음부터 끝까지 반복해서 읽는다.

   9-1. 현재 Frame을 한 장 읽는다.

   9-2. 더 이상 읽을 Frame이 없으면
        반복을 끝낸다.

   9-3. 전체 Frame 개수를 1 증가시킨다.

   9-4. 현재 Frame 번호가 30의 배수인지 확인한다.

        예:
        30
        60
        90
        120
        ...

   9-5. 30의 배수라면
        현재 Frame을 JPG 이미지로 저장한다.

   9-6. 저장한 Frame 개수를 1 증가시킨다.

10. 영상 사용을 종료한다.

11. 전체 Frame 수와
    실제 저장한 Frame 수를 출력한다.

## 핵심 흐름

영상 열기
   ↓
영상 정보 확인
   ↓
Frame 한 장 읽기
   ↓
Frame 번호 증가
   ↓
30번째 Frame인가?
   ↓
YES → 이미지 저장
NO  → 다음 Frame
   ↓
영상 끝까지 반복
   ↓
전체 결과 출력
'''

from pathlib import Path

import cv2


video_path = Path("samples/sample.mp4")
output_dir = Path("outputs/frames")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        f"영상을 열 수 없습니다: {video_path}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("영상 :", video_path)
print("FPS  :", fps)
print("크기 :", width, "x", height)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # 약 30 Frame마다 한 장 저장하는 예제
    if frame_count % 30 == 0:
        output_path = (
            output_dir / f"frame_{frame_count:05d}.jpg"
        )

        cv2.imwrite(str(output_path), frame)
        saved_count += 1

        print("저장:", output_path.name)

cap.release()

print("전체 Frame:", frame_count)
print("저장 Frame:", saved_count)