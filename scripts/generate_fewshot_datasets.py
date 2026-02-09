"""Generate CSV of datasets eligible for few-shot benchmarking.

Reads data/FLAb_0/FLAb1_structure_used_csv_check.csv and filters by N>=30.
Few-shot benchmark uses FLAb1 (FLAb_0) datasets only, which have structure data.
"""

import csv
import os

FLAB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK_CSV = os.path.join(FLAB_DIR, "data", "FLAb_0", "FLAb1_structure_used_csv_check.csv")
OUTPUT_CSV = os.path.join(FLAB_DIR, "data", "fewshot_eligible_datasets.csv")

MIN_SIZE = 30


def main():
    with open(CHECK_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    eligible = []
    for row in rows:
        size = int(row["Fasta_count"])
        if size < MIN_SIZE:
            continue

        csv_path = row["expected_csv_path"]
        structure_path = row["Structure_path"]
        folder_name = row["Folder_name"]
        category = row["Category"]

        # Verify CSV exists
        csv_abs = os.path.join(FLAB_DIR, csv_path)
        if not os.path.exists(csv_abs):
            print(f"WARNING: CSV not found: {csv_path}")
            continue

        # Verify structure directory exists
        structure_abs = os.path.join(FLAB_DIR, structure_path)
        if not os.path.isdir(structure_abs):
            print(f"WARNING: Structure dir not found: {structure_path}")
            continue

        eligible.append({
            "csv_path": csv_path,
            "category": category,
            "dataset_name": folder_name,
            "size": size,
            "structure_dir": structure_path,
        })

    # Sort by category then dataset_name
    eligible.sort(key=lambda x: (x["category"], x["dataset_name"]))

    fieldnames = ["csv_path", "category", "dataset_name", "size", "structure_dir"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in eligible:
            writer.writerow(row)

    print(f"Eligible datasets: {len(eligible)} (from {len(rows)} total, filtered by N>={MIN_SIZE})")
    print(f"Output: {OUTPUT_CSV}")

    # Summary by category
    from collections import Counter
    cat_counts = Counter(r["category"] for r in eligible)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
