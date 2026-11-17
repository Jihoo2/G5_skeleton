#!/usr/bin/env bash
# 1회 단위 비교 (run_pilot.sh를 먼저 실행해서 reports/pilot/tables/*_keypoints.csv가 있어야 함)
set -e
declare -A GLB=([A001]=261110_B001_T002_SHG [A006]=261110_B006_T001_SHG [A008]=261110_B008_T002_SHG)
for act in A001 A006 A008; do
  for pair in "Front_${act}:front" "Side_${act}:side" "${act}_up:up"; do
    stem=${pair%%:*}; view=${pair##*:}
    kp=reports/pilot/tables/${stem}_keypoints.csv
    [ -f "$kp" ] || { echo "skip: $kp"; continue; }
    python scripts/12_rep_compare_mocap.py --keypoints "$kp" \
      --glb data/pilot/mocap/${GLB[$act]}.glb --view "$view"
  done
done
python - <<'PY'
import glob, pandas as pd
fs = sorted(glob.glob("reports/pilot/reps/*_reps.csv"))
fs = [f for f in fs if not f.endswith("all_reps.csv")]
pd.concat([pd.read_csv(f) for f in fs]).to_csv("reports/pilot/reps/all_reps.csv", index=False, encoding="utf-8-sig")
print("완료: reports/pilot/reps/all_reps.csv, *_reps.png")
PY
