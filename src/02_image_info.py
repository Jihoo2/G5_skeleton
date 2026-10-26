''' 의사코드
먼저 읽어올 이미지 파일의 경로를 정합니다.

지정한 경로에서 이미지를 읽습니다.

이미지를 정상적으로 읽지 못했다면
    이미지 파일을 찾을 수 없다고 판단하고 오류를 발생시킵니다.

정상적으로 이미지를 읽었다면
    이미지의 세로 크기,
    가로 크기,
    채널 수를 확인합니다.

마지막으로
    이미지 파일 이름,
    가로 크기,
    세로 크기,
    채널 수,
    전체 이미지 배열의 크기,
    이미지 데이터의 자료형을
    화면에 출력하여 확인합니다.
'''

from pathlib import Path

import cv2


image_path = Path("samples/sample01.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

height, width, channels = image.shape

print("파일명   :", image_path.name)
print("Width    :", width)
print("Height   :", height)
print("Channels :", channels)
print("Shape    :", image.shape)
print("DataType :", image.dtype)