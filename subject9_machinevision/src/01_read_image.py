''' 의사코드
먼저 읽어올 이미지 파일의 위치를 정합니다.

지정한 경로에서 이미지를 읽습니다.

이미지를 정상적으로 읽지 못했다면
    파일 경로나 이미지 파일에 문제가 있다고 판단하고 오류를 발생시킵니다.

이미지를 저장할 폴더가 없다면
    저장할 폴더를 새로 만듭니다.

읽어온 이미지를 지정한 출력 경로에 저장합니다.

마지막으로
    어떤 이미지를 읽었는지,
    어디에 저장했는지,
    저장이 정상적으로 되었는지를 화면에 출력하여 확인합니다.
'''

from pathlib import Path

import cv2


image_path = Path("samples/sample01.jpg")
output_path = Path("outputs/read_image_check.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"이미지를 읽지 못했습니다: {image_path}"
    )

output_path.parent.mkdir(parents=True, exist_ok=True)

saved = cv2.imwrite(str(output_path), image)

print("입력 :", image_path)
print("저장 :", output_path)
print("저장 성공:", saved)