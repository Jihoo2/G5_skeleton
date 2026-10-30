import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "reports/day06_mini_challenge_results.csv"
OUTPUT = ROOT / "reports/day06_mini_regex_results.csv"

pattern = r"^[A-Z]{2}[0-9]{6}-[0-9][A-Z]$"

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

results = []

for row in rows:
    ocr_text = row["ocr_text"].strip()
    expected_text = row["expected_text"].strip()

    format_ok = re.fullmatch(pattern, ocr_text) is not None
    exact_match = ocr_text == expected_text

    results.append({
        "file": row["file"],
        "expected_text": expected_text,
        "ocr_text": ocr_text,
        "format_ok": format_ok,
        "exact_match": exact_match,
        "status": "OK" if format_ok and exact_match else "NG",
    })

with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

for row in results:
    print(row)

print("Saved:", OUTPUT)