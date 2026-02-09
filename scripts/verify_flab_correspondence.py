#!/usr/bin/env python3
"""
Verify correspondence between flab_paper.csv and actual data files.
Generates a report of mismatches and creates filename mappings.
"""

import os
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"


def load_flab_paper():
    """Load flab_paper.csv and return as DataFrame."""
    return pd.read_csv(DATA_DIR / "flab_paper.csv")


def check_file_existence(flab_paper: pd.DataFrame):
    """Check if each file in flab_paper.csv exists."""
    found = []
    not_found = []

    for _, row in flab_paper.iterrows():
        category = row["Category"]
        filename = row["Name"]
        path = DATA_DIR / category / filename

        if path.exists():
            found.append({
                "filename": filename,
                "category": category,
                "path": str(path),
                "expected_datapoints": row["Datapoints"]
            })
        else:
            not_found.append({
                "filename": filename,
                "category": category,
                "expected_path": str(path),
                "expected_datapoints": row["Datapoints"]
            })

    return found, not_found


def check_datapoint_counts(found_files: list):
    """Check if actual datapoint counts match expected counts."""
    matches = []
    mismatches = []

    for f in found_files:
        path = Path(f["path"])
        expected = f["expected_datapoints"]

        # Count actual datapoints (rows - 1 for header)
        df = pd.read_csv(path)
        actual = len(df)

        if actual == expected:
            matches.append({
                **f,
                "actual_datapoints": actual
            })
        else:
            mismatches.append({
                **f,
                "actual_datapoints": actual,
                "difference": actual - expected
            })

    return matches, mismatches


def find_potential_mappings(not_found: list):
    """Find potential filename mappings for not found files."""
    mappings = []
    still_not_found = []

    # Known mappings
    known_mappings = {
        "shanehsazzadeh2023_trastuzumab_zero_kd.csv": {
            "actual": "shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv",
            "reason": "Different naming convention"
        },
        "kothiwal2025htp_DKK\u00ac\u20201.00_ec50.csv": {
            "actual": "kothiwal2025htp_DKK_1.00_ec50.csv",
            "reason": "Encoding error (¬† replaced with underscore)"
        },
        # Also check for jain2017biophysical_CICRT.csv in different category
        "jain2017biophysical_CICRT.csv": {
            "actual": "jain2017biophysical_CICRT.csv",
            "reason": "File exists in polyreactivity category, not aggregation",
            "actual_category": "polyreactivity"
        }
    }

    # Files that need to be created from raw data
    ginkgo_files = [
        "ginkgo2025gdpa1_tm1_nanodsf_avg.csv",
        "ginkgo2025gdpa1_sec_pctmonomer_avg.csv",
        "ginkgo2025gdpa1_smac_rt_avg.csv",
        "ginkgo2025gdpa1_hic_rt_avg.csv",
        "ginkgo2025gdpa1_acsins_dLmax_ph74_avg.csv",
        "ginkgo2025gdpa1_acsins_dLmax_ph60_avg.csv",
        "ginkgo2025gdpa1_hac_rt_avg.csv",
        "ginkgo2025gdpa1_polyreactivity_prscore_cho_avg.csv"
    ]

    naturalantibody_abdesign_files = [
        f"krawczyk2025naturalantibody_{pdb}_bind.csv"
        for pdb in ["1BJ1", "1MHP", "1N8Z", "1NMB", "1VFB", "1XGR", "1XGT",
                    "1YY9", "3HFM", "5F9W", "5GGS", "5GGV", "6MFP", "6NMS",
                    "6NMU", "7BEJ", "7JMO", "7KF0"]
    ]

    naturalantibody_therapeutics_files = [
        "naturalantibody2025therapeutics_ada_prevalence_all.csv",
        "naturalantibody2025therapeutics_ada_incidence_all.csv",
        "naturalantibody2025therapeutics_ada_baseline_all.csv"
    ]

    for nf in not_found:
        filename = nf["filename"]
        category = nf["category"]

        if filename in known_mappings:
            mapping_info = known_mappings[filename]
            actual_filename = mapping_info["actual"]
            # Check if file is in a different category
            actual_category = mapping_info.get("actual_category", category)
            actual_path = DATA_DIR / actual_category / actual_filename

            if actual_path.exists():
                mappings.append({
                    "flab_paper_name": filename,
                    "actual_name": actual_filename,
                    "flab_paper_category": category,
                    "actual_category": actual_category,
                    "reason": mapping_info["reason"],
                    "actual_path": str(actual_path)
                })
            else:
                still_not_found.append({
                    **nf,
                    "note": f"Known mapping to {actual_filename} but file doesn't exist"
                })
        elif filename in ginkgo_files:
            still_not_found.append({
                **nf,
                "source": "ginkgo",
                "raw_data": "data/additionals/ginkgo/GDPa3_20260106_full.xlsx",
                "note": "Needs to be created from raw Ginkgo data"
            })
        elif filename in naturalantibody_abdesign_files:
            still_not_found.append({
                **nf,
                "source": "naturalantibody_abdesign",
                "raw_data": "data/additionals/naturalantibody/AbDesign-20260110T123750Z-1-001.zip",
                "note": "Needs to be created from AbDesign data"
            })
        elif filename in naturalantibody_therapeutics_files:
            still_not_found.append({
                **nf,
                "source": "naturalantibody_therapeutics",
                "raw_data": "data/additionals/naturalantibody/NA_therapeutics-20260110T123737Z-1-001.zip",
                "note": "Needs to be created from NA Therapeutics data"
            })
        else:
            still_not_found.append({
                **nf,
                "source": "unknown",
                "note": "Unknown source - may need investigation"
            })

    return mappings, still_not_found


def analyze_datapoint_mismatches(mismatches: list):
    """Analyze datapoint mismatches and identify patterns."""
    analysis = []

    for m in mismatches:
        filename = m["filename"]
        expected = m["expected_datapoints"]
        actual = m["actual_datapoints"]
        diff = m["difference"]

        # Check for Hutchinson swap
        if "hutchinson2023enhancement_top200" in filename and expected == 28:
            analysis.append({
                **m,
                "issue_type": "swapped_values",
                "explanation": "Values appear to be swapped with top27 file in flab_paper.csv"
            })
        elif "hutchinson2023enhancement_top27" in filename and expected == 192:
            analysis.append({
                **m,
                "issue_type": "swapped_values",
                "explanation": "Values appear to be swapped with top200 file in flab_paper.csv"
            })
        elif "hie2023efficient" in filename:
            analysis.append({
                **m,
                "issue_type": "count_error",
                "explanation": "Datapoint count in flab_paper.csv doesn't match actual file"
            })
        else:
            analysis.append({
                **m,
                "issue_type": "unknown",
                "explanation": "Datapoint count mismatch - needs investigation"
            })

    return analysis


def generate_report(results: dict):
    """Generate markdown and JSON reports."""
    REPORTS_DIR.mkdir(exist_ok=True)

    # JSON report
    json_report = {
        "summary": {
            "total_expected": results["total_expected"],
            "found": results["found_count"],
            "not_found": results["not_found_count"],
            "datapoint_matches": results["match_count"],
            "datapoint_mismatches": results["mismatch_count"],
            "filename_mappings": len(results["mappings"])
        },
        "datapoint_mismatches": results["mismatch_analysis"],
        "filename_mappings": results["mappings"],
        "not_found_files": results["still_not_found"],
        "found_files_with_correct_counts": len(results["matches"])
    }

    with open(REPORTS_DIR / "flab_correspondence_report.json", "w") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)

    # Markdown report
    md_lines = [
        "# FLAb Data Correspondence Report",
        "",
        "## Summary",
        "",
        f"- **Total expected files**: {results['total_expected']}",
        f"- **Files found**: {results['found_count']}",
        f"- **Files not found**: {results['not_found_count']}",
        f"- **Datapoint matches**: {results['match_count']}",
        f"- **Datapoint mismatches**: {results['mismatch_count']}",
        f"- **Filename mappings identified**: {len(results['mappings'])}",
        "",
        "## Datapoint Mismatches",
        "",
        "| File | Category | Expected | Actual | Difference | Issue |",
        "|------|----------|----------|--------|------------|-------|",
    ]

    for m in results["mismatch_analysis"]:
        md_lines.append(
            f"| `{m['filename']}` | {m['category']} | {m['expected_datapoints']} | "
            f"{m['actual_datapoints']} | {m['difference']:+d} | {m['issue_type']} |"
        )

    md_lines.extend([
        "",
        "## Filename Mappings",
        "",
        "These files exist with different names:",
        "",
        "| flab_paper.csv | Actual file | Reason |",
        "|----------------|-------------|--------|",
    ])

    for m in results["mappings"]:
        category_note = ""
        if m.get("flab_paper_category") != m.get("actual_category"):
            category_note = f" (category: {m.get('flab_paper_category')} -> {m.get('actual_category')})"
        md_lines.append(
            f"| `{m['flab_paper_name']}` | `{m['actual_name']}` | {m['reason']}{category_note} |"
        )

    md_lines.extend([
        "",
        "## Not Found Files",
        "",
        "### Need to be created from raw data",
        "",
    ])

    # Group by source
    by_source = {}
    for nf in results["still_not_found"]:
        source = nf.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(nf)

    for source, files in by_source.items():
        if source == "unknown":
            continue
        md_lines.append(f"#### {source}")
        md_lines.append("")
        if files and "raw_data" in files[0]:
            md_lines.append(f"Source: `{files[0]['raw_data']}`")
            md_lines.append("")
        md_lines.append("| File | Category | Expected Datapoints |")
        md_lines.append("|------|----------|---------------------|")
        for f in files:
            md_lines.append(f"| `{f['filename']}` | {f['category']} | {f['expected_datapoints']} |")
        md_lines.append("")

    if "unknown" in by_source:
        md_lines.append("### Unknown source (need investigation)")
        md_lines.append("")
        md_lines.append("| File | Category | Expected Datapoints |")
        md_lines.append("|------|----------|---------------------|")
        for f in by_source["unknown"]:
            md_lines.append(f"| `{f['filename']}` | {f['category']} | {f['expected_datapoints']} |")
        md_lines.append("")

    with open(REPORTS_DIR / "flab_correspondence_report.md", "w") as f:
        f.write("\n".join(md_lines))

    print(f"Reports generated:")
    print(f"  - {REPORTS_DIR / 'flab_correspondence_report.json'}")
    print(f"  - {REPORTS_DIR / 'flab_correspondence_report.md'}")


def create_filename_mapping_json(mappings: list):
    """Create/update filename mapping JSON for files with different names."""
    mapping_file = BASE_DIR / "data" / "filename_mapping.json"

    # Load existing mapping if exists
    existing = {}
    if mapping_file.exists():
        with open(mapping_file) as f:
            existing = json.load(f)

    # Add new mappings
    new_mappings = {}
    for m in mappings:
        flab_category = m.get("flab_paper_category", m.get("category"))
        actual_category = m.get("actual_category", flab_category)
        key = f"{flab_category}/{m['flab_paper_name']}"
        new_mappings[key] = {
            "flab_paper_name": m["flab_paper_name"],
            "actual_name": m["actual_name"],
            "flab_paper_category": flab_category,
            "actual_category": actual_category,
            "actual_path": f"data/{actual_category}/{m['actual_name']}",
            "reason": m["reason"]
        }

    # Merge with existing
    if "filename_mappings" not in existing:
        existing["filename_mappings"] = {}
    existing["filename_mappings"].update(new_mappings)

    with open(mapping_file, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Filename mapping updated: {mapping_file}")


def main():
    print("=" * 60)
    print("FLAb Data Correspondence Verification")
    print("=" * 60)
    print()

    # Load flab_paper.csv
    flab_paper = load_flab_paper()
    total = len(flab_paper)
    print(f"Total files in flab_paper.csv: {total}")

    # Check file existence
    found, not_found = check_file_existence(flab_paper)
    print(f"Files found: {len(found)}")
    print(f"Files not found: {len(not_found)}")

    # Check datapoint counts
    matches, mismatches = check_datapoint_counts(found)
    print(f"Datapoint matches: {len(matches)}")
    print(f"Datapoint mismatches: {len(mismatches)}")

    # Find potential mappings
    mappings, still_not_found = find_potential_mappings(not_found)
    print(f"Filename mappings found: {len(mappings)}")
    print(f"Still not found: {len(still_not_found)}")

    # Analyze mismatches
    mismatch_analysis = analyze_datapoint_mismatches(mismatches)

    # Generate reports
    results = {
        "total_expected": total,
        "found_count": len(found),
        "not_found_count": len(not_found),
        "match_count": len(matches),
        "mismatch_count": len(mismatches),
        "matches": matches,
        "mismatch_analysis": mismatch_analysis,
        "mappings": mappings,
        "still_not_found": still_not_found
    }

    generate_report(results)

    # Create filename mapping JSON
    if mappings:
        create_filename_mapping_json(mappings)

    print()
    print("=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
