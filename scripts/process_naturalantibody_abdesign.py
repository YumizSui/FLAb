#!/usr/bin/env python3
"""
Process Natural Antibody AbDesign data into FLAb binding format.

Input: data/additionals/naturalantibody/AbDesign-20260110T123750Z-1-001.zip
Outputs (18 files in data/binding/):
- krawczyk2025naturalantibody_{PDB}_bind.csv for each expected PDB
"""

import pandas as pd
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ABDESIGN_ZIP = DATA_DIR / "additionals" / "naturalantibody" / "AbDesign-20260110T123750Z-1-001.zip"

# Expected PDBs and their datapoint counts from flab_paper.csv
EXPECTED_PDBS = {
    "1BJ1": 68,
    "1MHP": 131,
    "1N8Z": 42,
    "1NMB": 8,
    "1VFB": 65,
    "1XGR": 4,
    "1XGT": 4,
    "1YY9": 21,
    "3HFM": 46,
    "5F9W": 58,
    "5GGS": 51,
    "5GGV": 66,
    "6MFP": 36,
    "6NMS": 48,
    "6NMU": 46,
    "7BEJ": 34,
    "7JMO": 60,
    "7KF0": 36
}


def load_abdesign_data():
    """Load datasets_mut.csv from the zip file."""
    print(f"Loading {ABDESIGN_ZIP}...")

    with zipfile.ZipFile(ABDESIGN_ZIP) as z:
        with z.open("AbDesign/datasets_mut.csv") as f:
            df = pd.read_csv(f)

    print(f"Loaded {len(df)} total mutation records")
    return df


def process_pdb(df: pd.DataFrame, pdb: str, expected_count: int):
    """Process a single PDB into FLAb format CSV."""
    output_path = DATA_DIR / "binding" / f"krawczyk2025naturalantibody_{pdb}_bind.csv"

    # Filter for this PDB
    pdb_df = df[df["pdb_name"] == pdb].copy()
    actual_count = len(pdb_df)

    if actual_count == 0:
        print(f"  WARNING: No data found for PDB {pdb}")
        return 0

    if actual_count != expected_count:
        print(f"  NOTE: Expected {expected_count} rows, got {actual_count}")

    # Create output DataFrame
    # Use heavy_sequence_variable and light_sequence_variable for Fv regions
    # Use affinity as the binding measurement (elisa_mut_to_wt_ratio)
    result = pd.DataFrame({
        "heavy": pdb_df["heavy_sequence_variable"],
        "light": pdb_df["light_sequence_variable"],
        "bind": pdb_df["affinity"],
        "fitness": pdb_df["affinity"]
    })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"  Created {output_path} ({actual_count} rows)")
    return actual_count


def main():
    print("=" * 60)
    print("Natural Antibody AbDesign Data Processing")
    print("=" * 60)
    print()

    # Load data
    df = load_abdesign_data()
    print()

    # Show available PDBs
    available_pdbs = df["pdb_name"].unique()
    print(f"Available PDBs in data: {len(available_pdbs)}")
    print()

    # Process each expected PDB
    results = {}
    missing = []
    count_mismatches = []

    for pdb, expected_count in EXPECTED_PDBS.items():
        print(f"Processing {pdb}...")

        if pdb not in available_pdbs:
            print(f"  WARNING: PDB {pdb} not found in data")
            missing.append(pdb)
            continue

        actual_count = process_pdb(df, pdb, expected_count)
        results[pdb] = actual_count

        if actual_count != expected_count:
            count_mismatches.append({
                "pdb": pdb,
                "expected": expected_count,
                "actual": actual_count
            })

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()

    print(f"Files created: {len(results)}")
    for pdb, count in results.items():
        expected = EXPECTED_PDBS[pdb]
        status = "✓" if count == expected else f"(expected {expected})"
        print(f"  {pdb}: {count} {status}")

    if missing:
        print()
        print(f"Missing PDBs ({len(missing)}):")
        for pdb in missing:
            print(f"  - {pdb}")

    if count_mismatches:
        print()
        print(f"Count mismatches ({len(count_mismatches)}):")
        for m in count_mismatches:
            print(f"  - {m['pdb']}: expected {m['expected']}, got {m['actual']}")

    print()
    print("AbDesign data processing complete!")


if __name__ == "__main__":
    main()
