#!/usr/bin/env python
"""
Generate sbatch scripts for all PoET-2 var2 configs.
"""
from pathlib import Path

# Get all config names
config_dir = Path("/home/kfurui/workspace/AbEval/PoET-2/configs/var2")
configs = sorted([f.stem for f in config_dir.glob("var2_*.yaml")])

print(f"Found {len(configs)} configs")

# Create sbatch directory
sbatch_dir = Path("/home/kfurui/workspace/FLAb/sbatch/poet2_all_configs")
sbatch_dir.mkdir(exist_ok=True, parents=True)

# Generate sbatch files
for config in configs:
    for mode in ['str', 'nostr']:
        job_name = f"{config}_{mode}"
        sbatch_path = sbatch_dir / f"{job_name}.sh"

        sbatch_content = f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name={job_name}
#SBATCH --output=/home/kfurui/workspace/FLAb/poet2_tm_results/{config}/{mode}_%j.out

cd /home/kfurui/workspace/FLAb

# Wait random time to avoid simultaneous starts
sleep $((RANDOM % 30))

source /home/kfurui/workspace/AbEval/PoET-2/.venv/bin/activate

python scripts/test_all_poet2_configs.py \\
    --checkpoint "var2/{config}" \\
    --mode {mode} \\
    --output-dir poet2_tm_results \\
    --device cuda

echo "Job completed"
"""

        with open(sbatch_path, 'w') as f:
            f.write(sbatch_content)

        sbatch_path.chmod(0o755)

print(f"\nGenerated {len(configs) * 2} sbatch scripts in {sbatch_dir}")

# Generate master submission script
submit_script = sbatch_dir / "submit_all.sh"
with open(submit_script, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write("# Submit all PoET-2 config jobs\n\n")
    for config in configs:
        for mode in ['str', 'nostr']:
            job_name = f"{config}_{mode}"
            f.write(f"sbatch sbatch/poet2_all_configs/{job_name}.sh\n")
            f.write("sleep 1\n")

submit_script.chmod(0o755)

print(f"Master submission script: {submit_script}")
print(f"\nTo submit all jobs, run:")
print(f"  bash {submit_script}")
