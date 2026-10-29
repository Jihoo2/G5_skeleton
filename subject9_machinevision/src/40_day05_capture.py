
from pathlib import Path
import cv2

# --------------------------------------
# 1. 휴대폰 카메라 연결
# --------------------------------------
stream_url = "http://192.168.40.114:4747/video"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    raise RuntimeError("스마트폰 카메라 연결 실패")

# --------------------------------------
# 2. 저장 경로
# --------------------------------------
base_dir = Path("samples/day05/mini_challenge")

save_paths = {
    ord("1"): base_dir / "calibration/ok/ok_01.jpg",
    ord("2"): base_dir / "calibration/ok/ok_02.jpg",
    ord("3"): base_dir / "calibration/ok/ok_03.jpg",

    ord("4"): base_dir / "calibration/ng/ng_01.jpg",
    ord("5"): base_dir / "calibration/ng/ng_02.jpg",
    ord("6"): base_dir / "calibration/ng/ng_03.jpg",

    ord("7"): base_dir / "test/ok/test_ok_01.jpg",
    ord("8"): base_dir / "test/ok/test_ok_02.jpg",

    ord("9"): base_dir / "test/ng/test_ng_01.jpg",
    ord("0"): base_dir / "test/ng/test_ng_02.jpg",
}

for path in save_paths.values():
    path.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# 3. 미리보기 설정
# --------------------------------------
window_name = "Day05 Mini Challenge"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

# 화면을 크게 표시
preview_width = 900
preview_height = 1200

cv2.resizeWindow(
    window_name,
    preview_width,
    preview_height
)

print("=" * 45)
print("Day05 사진 촬영")
print("=" * 45)
print("1~3 : Calibration OK")
print("4~6 : Calibration NG")
print("7~8 : Test OK")
print("9,0 : Test NG")
print("q   : 종료")
print("=" * 45)

# --------------------------------------
# 4. 실시간 영상 + 촬영
# --------------------------------------
try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("카메라 프레임 읽기 실패")
            break

        # 시계 방향으로 90도 회전
        frame = cv2.rotate(
            frame,
            cv2.ROTATE_90_CLOCKWISE
        )

        # 미리보기 화면
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        # 종료
        if key == ord("q") or key == 27:
            print("촬영 종료")
            break

        # 숫자키를 누르면 사진 저장
        if key in save_paths:

            save_path = save_paths[key]

            success = cv2.imwrite(
                str(save_path),
                frame
            )

            if success:
                print(f"[저장 완료] {save_path}")
            else:
                print(f"[저장 실패] {save_path}")

except KeyboardInterrupt:
    print("촬영 중단")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("카메라 연결 종료")
