"""Deduplicate degradation_curve.csv: keep first occurrence of each (n, rep, gate)."""
import csv
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "results_branch_a2", "degradation_curve.csv")
CSV_PATH = os.path.normpath(CSV_PATH)

if not os.path.exists(CSV_PATH):
    print(f"CSV not found: {CSV_PATH}")
    exit(0)

rows = []
with open(CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        rows.append(r)

seen = set()
deduped = []
for r in rows:
    key = (int(r["n_per_class"]), int(r["repeat_idx"]), r["gate_type"])
    if key not in seen:
        seen.add(key)
        deduped.append(r)

removed = len(rows) - len(deduped)
print(f"Rows: {len(rows)} -> {len(deduped)} (removed {removed} duplicates)")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(deduped)
    f.flush()
    os.fsync(f.fileno())

print("CSV deduplicated successfully.")