#!/usr/bin/env python3
"""
Process Ginkgo GDPa1 Excel file into FLAb standard format.

Input: data/additionals/ginkgo/GDPa1_v1.3_20251027_full.xlsx
Outputs (8 files):
- data/thermostability/ginkgo2025gdpa1_tm1_nanodsf_avg.csv (237 pts)
- data/aggregation/ginkgo2025gdpa1_sec_pctmonomer_avg.csv (244 pts)
- data/aggregation/ginkgo2025gdpa1_smac_rt_avg.csv (244 pts)
- data/aggregation/ginkgo2025gdpa1_hic_rt_avg.csv (244 pts)
- data/aggregation/ginkgo2025gdpa1_acsins_dLmax_ph74_avg.csv (244 pts)
- data/aggregation/ginkgo2025gdpa1_acsins_dLmax_ph60_avg.csv (244 pts)
- data/polyreactivity/ginkgo2025gdpa1_hac_rt_avg.csv (94 pts)
- data/polyreactivity/ginkgo2025gdpa1_polyreactivity_prscore_cho_avg.csv (197 pts)
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
GINKGO_XLSX = DATA_DIR / "additionals" / "ginkgo" / "GDPa1_v1.3_20251027_full.xlsx"

# Mapping from Excel columns to output files
GINKGO_MAPPING = {
    "tm1_nanodsf_avg": {
        "output": "thermostability/ginkgo2025gdpa1_tm1_nanodsf_avg.csv",
        "measurement_name": "Tm1 (nanoDSF)",
        "expected_count": 237
    },
    "sec_%monomer_avg": {
        "output": "aggregation/ginkgo2025gdpa1_sec_pctmonomer_avg.csv",
        "measurement_name": "SEC %Monomer",
        "expected_count": 244
    },
    "smac_rt_avg": {
        "output": "aggregation/ginkgo2025gdpa1_smac_rt_avg.csv",
        "measurement_name": "SMAC RT",
        "expected_count": 244
    },
    "hic_rt_avg": {
        "output": "aggregation/ginkgo2025gdpa1_hic_rt_avg.csv",
        "measurement_name": "HIC RT",
        "expected_count": 244
    },
    "acsins_dLmax_ph7.4_avg": {
        "output": "aggregation/ginkgo2025gdpa1_acsins_dLmax_ph74_avg.csv",
        "measurement_name": "AC-SINS dLmax pH7.4",
        "expected_count": 244
    },
    "acsins_dLmax_ph6.0_avg": {
        "output": "aggregation/ginkgo2025gdpa1_acsins_dLmax_ph60_avg.csv",
        "measurement_name": "AC-SINS dLmax pH6.0",
        "expected_count": 244
    },
    "hac_rt_avg": {
        "output": "polyreactivity/ginkgo2025gdpa1_hac_rt_avg.csv",
        "measurement_name": "HAC RT",
        "expected_count": 94
    },
    "polyreactivity_prscore_cho_avg": {
        "output": "polyreactivity/ginkgo2025gdpa1_polyreactivity_prscore_cho_avg.csv",
        "measurement_name": "Polyreactivity PRScore CHO",
        "expected_count": 197
    }
}


def load_ginkgo_data():
    """Load and merge Sequences and Assay Data sheets."""
    print(f"Loading {GINKGO_XLSX}...")

    xlsx = pd.ExcelFile(GINKGO_XLSX)

    # Load sequences
    seq_df = pd.read_excel(xlsx, "Sequences")
    seq_df = seq_df[["antibody_id", "vh_protein_sequence", "vl_protein_sequence"]]
    seq_df = seq_df.rename(columns={
        "vh_protein_sequence": "heavy",
        "vl_protein_sequence": "light"
    })

    # Load assay data averages
    avg_df = pd.read_excel(xlsx, "Assay Data - average")

    # Merge on antibody_id
    merged = avg_df.merge(seq_df, on="antibody_id", how="left")

    print(f"Loaded {len(merged)} samples")
    return merged


def process_measurement(merged_df: pd.DataFrame, column: str, config: dict):
    """Process a single measurement into FLAb format CSV."""
    output_path = DATA_DIR / config["output"]
    measurement_name = config["measurement_name"]
    expected_count = config["expected_count"]

    # Filter rows with non-null measurement values
    df = merged_df[merged_df[column].notna()].copy()

    # Select and rename columns
    result = pd.DataFrame({
        "heavy": df["heavy"],
        "light": df["light"],
        measurement_name: df[column],
        "fitness": df[column]
    })

    # Verify count
    actual_count = len(result)
    if actual_count != expected_count:
        print(f"  WARNING: Expected {expected_count} rows, got {actual_count}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"  Created {output_path} ({actual_count} rows)")
    return actual_count


def main():
    print("=" * 60)
    print("Ginkgo Data Processing")
    print("=" * 60)
    print()

    # Load data
    merged = load_ginkgo_data()
    print()

    # Process each measurement
    results = {}
    for column, config in GINKGO_MAPPING.items():
        print(f"Processing {column}...")
        count = process_measurement(merged, column, config)
        results[config["output"]] = count

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    for output, count in results.items():
        print(f"  {output}: {count} rows")

    print()
    print("Ginkgo data processing complete!")


if __name__ == "__main__":
    main()
