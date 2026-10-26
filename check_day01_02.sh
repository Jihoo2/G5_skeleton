#!/usr/bin/env bash
# 1·2일차 산출물 점검 — 프로젝트 최상위에서 실행: bash check_day01_02.sh
cd "$(dirname "$0")" || exit 1
miss=0
check() { if [ -e "$1" ]; then echo "  [OK]   $1"; else echo "  [없음] $1"; miss=$((miss+1)); fi; }

echo "== 환경 =="
if [ -x .venv/bin/python ]; then
  .venv/bin/python src/00_check_env.py 2>&1 | sed 's/^/  /' || echo "  [오류] venv 깨짐 → .venv 재생성 필요"
else
  echo "  [없음] .venv/bin/python"; miss=$((miss+1))
fi

echo "== 코드 =="
for f in 00_check_env 01_read_image 02_image_info 03_roi_crop 04_batch_roi 05_video_input \
         06_video_roi 07_webcam_input 08_phone_camera_roi 09_color_space 10_brightness_contrast \
         11_gamma 12_clahe 13_filter_compare 14_sharpen 15_threshold_otsu \
         16_preprocess_compare 17_day02_challenge; do check "src/$f.py"; done

echo "== 문서 =="
for f in README.md .gitignore requirements.txt reports/day01_problem_definition.md \
         reports/dataset_roles.csv reports/day02_parameter_table.csv reports/day02_preprocess_reason.md; do check "$f"; done

echo "== 1일차 결과 =="
check outputs/read_image_check.jpg
check outputs/roi/sample01_roi_preview.jpg
check outputs/live_roi/cap_on.jpg
check outputs/live_roi/cap_off.jpg
echo "  frames: $(ls outputs/frames 2>/dev/null | wc -l)장, video_roi: $(ls outputs/video_roi 2>/dev/null | wc -l)장"

echo "== 2일차 입력 =="
for f in cap_on_reference cap_off_reference dark low_contrast noise blur glare challenge; do check "samples/day02/$f.jpg"; done

echo "== 2일차 결과 =="
for d in color brightness gamma clahe filter sharpen threshold compare challenge; do
  echo "  outputs/day02/$d: $(ls outputs/day02/$d 2>/dev/null | wc -l)개"
done
check outputs/day02/compare/preprocess_compare.png
check outputs/day02/challenge/challenge_result.jpg

echo "== 정리 대상 =="
[ -d subject9_machinevision ] && echo "  [주의] 중첩 폴더 subject9_machinevision/ 남아 있음"
n=$(find . -name '*:Zone.Identifier' -not -path './.venv/*' | wc -l)
[ "$n" -gt 0 ] && echo "  [주의] Zone.Identifier ${n}개 → find . -name '*:Zone.Identifier' -delete"
[ -f Untitled.ipynb ] && echo "  [참고] Untitled.ipynb — 필요 없으면 삭제 또는 이름 변경"
grep -l "✏️" reports/*.md reports/*.csv 2>/dev/null | sed 's/^/  [작성 필요] /'

echo
[ $miss -eq 0 ] && echo "누락 없음" || echo "누락 ${miss}개"
