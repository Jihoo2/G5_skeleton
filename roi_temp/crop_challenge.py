# 교재 28장 — challenge.jpg 원본에서 병뚜껑 ROI를 마우스로 잘라 덮어쓰는 도구
# 원본은 roi_temp/challenge_original.jpg 로 한 번만 백업됩니다.
# 이미 ROI로 잘랐다면 다시 실행하지 마세요 (잘린 이미지를 또 자르게 됩니다).
from pathlib import Path
import shutil

import cv2


image_path = Path("samples/day02/challenge.jpg")
temp_dir = Path("roi_temp")
backup_path = temp_dir / "challenge_original.jpg"
temp_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(f"이미지를 읽지 못했습니다: {image_path}")

if not backup_path.exists():
    shutil.copy2(image_path, backup_path)
    print("원본 백업:", backup_path)

height, width = image.shape[:2]
scale = min(1.0, 900 / width, 700 / height)
preview = cv2.resize(image, (int(width * scale), int(height * scale)))

start_point = None
end_point = None
dragging = False
roi_selected = False


def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, dragging, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = end_point = (x, y)
        dragging = True
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        dragging = False
        roi_selected = True
        print("ROI 선택:", start_point, "→", end_point)


window_name = "ROI - Drag and press ENTER"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window_name, mouse_callback)

print("드래그로 ROI 선택 → ENTER/SPACE 저장, C 다시 선택, ESC 취소")

while True:
    canvas = preview.copy()
    if start_point is not None and end_point is not None:
        cv2.rectangle(canvas, start_point, end_point, (0, 255, 0), 2)
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        cv2.destroyAllWindows()
        print("ROI 선택 취소")
        raise SystemExit
    if key in (ord("c"), ord("C")):
        start_point = end_point = None
        dragging = roi_selected = False
        print("ROI 다시 선택")
    if key in (13, 10, 32):
        if not roi_selected:
            print("먼저 ROI를 드래그하세요.")
            continue
        break

cv2.destroyAllWindows()

(x1, y1), (x2, y2) = start_point, end_point
left, right = int(min(x1, x2) / scale), int(max(x1, x2) / scale)
top, bottom = int(min(y1, y2) / scale), int(max(y1, y2) / scale)

roi = image[top:bottom, left:right]
if roi.size == 0:
    raise ValueError("ROI 영역이 올바르지 않습니다.")

cv2.imwrite(str(image_path), roi)
print("ROI 저장 완료:", image_path, (left, top, right, bottom))
