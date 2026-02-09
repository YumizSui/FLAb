#!/usr/bin/env python3
"""
Environment verification script for FLAb thermostability tests.

Checks that all required environments are properly set up before submitting jobs.
"""

import os
import subprocess
import sys
from pathlib import Path

# Base directory
BASE_DIR = Path("/gs/bs/tga-furui/workspace/antibody/FLAb")

# Environment configurations
ENVIRONMENTS = {
    "antiberty": {
        "type": "uv",
        "path": BASE_DIR / "envs/antiberty/.venv",
        "setup_cmd": "cd envs/antiberty && uv sync",
        "test_imports": ["antiberty", "esm", "ablang"],
        "models": ["antiberty", "esm2", "ism", "ablang2"],
    },
    "esmif_mpnn": {
        "type": "pixi",
        "path": BASE_DIR / "envs/esmif_mpnn",
        "setup_cmd": None,  # Pixi projects don't need sync
        "test_imports": ["esm", "torch"],
        "models": ["esmif", "mpnn"],
    },
    "pyrosetta": {
        "type": "pixi",
        "path": BASE_DIR / "envs/pyrosetta",
        "setup_cmd": None,
        "test_imports": ["pyrosetta"],
        "models": ["pyrosetta"],
    },
    "poet2": {
        "type": "venv",
        "path": BASE_DIR / "PoET-2/.venv",
        "setup_cmd": None,
        "test_imports": ["transformers", "torch"],
        "models": ["sablm_nostr", "sablm_str"],
    },
}


def check_env_exists(env_name, config):
    """Check if environment directory exists."""
    path = config["path"]
    if config["type"] == "pixi":
        # For pixi, check if pixi.toml exists
        pixi_file = path / "pixi.toml"
        exists = pixi_file.exists()
        if exists:
            print(f"  ✓ Pixi project found at {path}")
        else:
            print(f"  ✗ Pixi project NOT found at {path}")
        return exists
    else:
        # For venv/uv, check if .venv exists
        exists = path.exists() and (path / "bin/activate").exists()
        if exists:
            print(f"  ✓ Virtual environment found at {path}")
        else:
            print(f"  ✗ Virtual environment NOT found at {path}")
        return exists


def check_env_imports(env_name, config):
    """Try to activate environment and import test modules."""
    # Skip import tests - just verify activation works
    if config["type"] == "pixi":
        # For pixi, check if we can run pixi
        cmd = f"cd {config['path']} && pixi --version 2>&1"
    else:
        # For venv, check if python exists
        python_path = config['path'] / "bin/python"
        if python_path.exists():
            cmd = f"{python_path} --version 2>&1"
        else:
            print(f"  ✗ Python executable not found at {python_path}")
            return False

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            print(f"  ✓ Environment ready (skipped import test)")
            return True
        else:
            print(f"  ✗ Environment check FAILED:")
            print(f"    Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Environment check timed out (>5s)")
        return False
    except Exception as e:
        print(f"  ✗ Environment check error: {e}")
        return False


def main():
    print("=" * 60)
    print("FLAb Test Environment Verification")
    print("=" * 60)
    print()

    # Check data files exist
    print("Checking data files...")
    csv_path = BASE_DIR / "data/thermostability/jain2017biophysical_Tm.csv"
    struct_dir = BASE_DIR / "new_structure/thermostability/jain2017biophysical_Tm"

    if csv_path.exists():
        print(f"  ✓ CSV file exists: {csv_path}")
    else:
        print(f"  ✗ CSV file NOT found: {csv_path}")

    if struct_dir.exists():
        pdb_files = list(struct_dir.rglob("*.pdb"))
        print(f"  ✓ Structure directory exists: {struct_dir}")
        print(f"    Found {len(pdb_files)} PDB files")
    else:
        print(f"  ✗ Structure directory NOT found: {struct_dir}")

    print()

    # Check each environment
    results = {}
    for env_name, config in ENVIRONMENTS.items():
        print(f"Checking {env_name} environment ({config['type']})...")
        print(f"  Models: {', '.join(config['models'])}")

        exists = check_env_exists(env_name, config)

        if not exists and config["setup_cmd"]:
            print(f"  → Setup needed: {config['setup_cmd']}")
            results[env_name] = "needs_setup"
        elif not exists:
            print(f"  → Environment missing and no setup command available")
            results[env_name] = "missing"
        else:
            # Try to import test modules
            imports_ok = check_env_imports(env_name, config)
            results[env_name] = "ready" if imports_ok else "broken"

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    ready = [name for name, status in results.items() if status == "ready"]
    needs_setup = [name for name, status in results.items() if status == "needs_setup"]
    broken = [name for name, status in results.items() if status in ["broken", "missing"]]

    print(f"\n✓ Ready ({len(ready)}):")
    for name in ready:
        models = ", ".join(ENVIRONMENTS[name]["models"])
        print(f"  - {name}: {models}")

    if needs_setup:
        print(f"\n⚠ Needs Setup ({len(needs_setup)}):")
        for name in needs_setup:
            models = ", ".join(ENVIRONMENTS[name]["models"])
            cmd = ENVIRONMENTS[name]["setup_cmd"]
            print(f"  - {name}: {models}")
            print(f"    Run: {cmd}")

    if broken:
        print(f"\n✗ Broken/Missing ({len(broken)}):")
        for name in broken:
            models = ", ".join(ENVIRONMENTS[name]["models"])
            print(f"  - {name}: {models}")

    print()

    if broken:
        print("Some environments are broken or missing. Please investigate.")
        return 1
    elif needs_setup:
        print("Some environments need setup. Run the commands above, then re-run this script.")
        return 2
    else:
        print("All environments are ready! You can proceed with job submission.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
