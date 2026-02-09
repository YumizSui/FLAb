#!/usr/bin/env python3
"""
Process Natural Antibody Therapeutics data into FLAb immunogenicity format.

Input: data/additionals/naturalantibody/NA_therapeutics-20260110T123737Z-1-001.zip
Outputs (3 files in data/immunogenicity/):
- naturalantibody2025therapeutics_ada_prevalence_all.csv (3456 pts expected)
- naturalantibody2025therapeutics_ada_incidence_all.csv (820 pts expected)
- naturalantibody2025therapeutics_ada_baseline_all.csv (579 pts expected)
"""

import pandas as pd
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
NA_THERAPEUTICS_ZIP = DATA_DIR / "additionals" / "naturalantibody" / "NA_therapeutics-20260110T123737Z-1-001.zip"

# Expected counts from flab_paper.csv
EXPECTED_COUNTS = {
    "prevalence": 3456,
    "incidence": 820,
    "baseline": 579
}


def load_sequence_data(z: zipfile.ZipFile):
    """Load and prepare sequence data from variable_domains_airr.csv."""
    with z.open("NA_therapeutics/Therapeutics - variable_domains_airr.csv") as f:
        domains = pd.read_csv(f, on_bad_lines="skip")

    # Pivot to get heavy and light chain sequences for each therapeutic
    heavy_chains = domains[domains["locus"] == "igh"][["therapeutic", "sequence_aa"]]
    heavy_chains = heavy_chains.rename(columns={"sequence_aa": "heavy"})
    # Take the first heavy chain for each therapeutic
    heavy_chains = heavy_chains.groupby("therapeutic").first().reset_index()

    light_chains = domains[domains["locus"].isin(["igk", "igl"])][["therapeutic", "sequence_aa"]]
    light_chains = light_chains.rename(columns={"sequence_aa": "light"})
    # Take the first light chain for each therapeutic
    light_chains = light_chains.groupby("therapeutic").first().reset_index()

    # Merge heavy and light chains
    sequences = heavy_chains.merge(light_chains, on="therapeutic", how="inner")

    print(f"Loaded sequences for {len(sequences)} therapeutics")
    return sequences


def load_measurement_data(z: zipfile.ZipFile, measurement_type: str):
    """Load prevalence, incidence, or baseline data."""
    filename = f"NA_therapeutics/{measurement_type}.csv"

    with z.open(filename) as f:
        df = pd.read_csv(f)

    # The therapeutic column might be named differently
    if "therapeutic" in df.columns:
        therapeutic_col = "therapeutic"
    elif "measurement_therapeutic" in df.columns:
        therapeutic_col = "measurement_therapeutic"
    else:
        # Try to find a column with 'therapeutic' in the name
        therapeutic_cols = [c for c in df.columns if "therapeutic" in c.lower()]
        if therapeutic_cols:
            therapeutic_col = therapeutic_cols[0]
        else:
            raise ValueError(f"No therapeutic column found in {filename}")

    # Rename to standard name
    df = df.rename(columns={therapeutic_col: "therapeutic"})

    # The measurement value column
    if "measurement_value" in df.columns:
        value_col = "measurement_value"
    else:
        value_cols = [c for c in df.columns if "value" in c.lower()]
        if value_cols:
            value_col = value_cols[0]
        else:
            raise ValueError(f"No measurement value column found in {filename}")

    df = df.rename(columns={value_col: "ada_value"})

    print(f"Loaded {len(df)} {measurement_type} records")
    return df[["therapeutic", "ada_value"]]


def process_measurement(sequences: pd.DataFrame, measurements: pd.DataFrame,
                        measurement_type: str, expected_count: int):
    """Process a measurement type into FLAb format."""
    output_path = DATA_DIR / "immunogenicity" / f"naturalantibody2025therapeutics_ada_{measurement_type}_all.csv"

    # Convert therapeutic names to lowercase for matching
    sequences_lower = sequences.copy()
    sequences_lower["therapeutic_lower"] = sequences_lower["therapeutic"].str.lower()

    measurements_lower = measurements.copy()
    measurements_lower["therapeutic_lower"] = measurements_lower["therapeutic"].str.lower()

    # Join measurements with sequences
    merged = measurements_lower.merge(
        sequences_lower[["therapeutic_lower", "heavy", "light"]],
        on="therapeutic_lower",
        how="inner"
    )

    actual_count = len(merged)
    print(f"  Matched {actual_count} records with sequences (expected {expected_count})")

    if actual_count == 0:
        print(f"  WARNING: No matches found!")
        return 0

    # Create output DataFrame
    result = pd.DataFrame({
        "heavy": merged["heavy"],
        "light": merged["light"],
        "%ADA": merged["ada_value"],
        "fitness": merged["ada_value"]
    })

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"  Created {output_path} ({actual_count} rows)")
    return actual_count


def main():
    print("=" * 60)
    print("Natural Antibody Therapeutics Data Processing")
    print("=" * 60)
    print()

    with zipfile.ZipFile(NA_THERAPEUTICS_ZIP) as z:
        # Load sequence data
        print("Loading sequence data...")
        sequences = load_sequence_data(z)
        print()

        results = {}

        for measurement_type in ["prevalence", "incidence", "baseline"]:
            print(f"Processing {measurement_type}...")

            # Load measurement data
            measurements = load_measurement_data(z, measurement_type)

            # Process and save
            count = process_measurement(
                sequences,
                measurements,
                measurement_type,
                EXPECTED_COUNTS[measurement_type]
            )
            results[measurement_type] = count
            print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()

    for measurement_type, count in results.items():
        expected = EXPECTED_COUNTS[measurement_type]
        status = "✓" if count == expected else f"(expected {expected})"
        print(f"  {measurement_type}: {count} {status}")

    print()
    print("Therapeutics data processing complete!")


if __name__ == "__main__":
    main()
