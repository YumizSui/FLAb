#!/usr/bin/env python
"""
Check which PoET-2 checkpoints are complete.
"""
import json
from pathlib import Path

checkpoint_dir = Path("/home/kfurui/workspace/AbEval/PoET-2/checkpoints/var2")

complete_checkpoints = []
incomplete_checkpoints = []

for ckpt_dir in sorted(checkpoint_dir.iterdir()):
    if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("var2_str-"):
        continue

    best_checkpoint_file = ckpt_dir / "best_checkpoint.json"
    if not best_checkpoint_file.exists():
        incomplete_checkpoints.append((ckpt_dir.name, "no best_checkpoint.json"))
        continue

    with open(best_checkpoint_file) as f:
        best_info = json.load(f)

    model_path = Path("/home/kfurui/workspace/AbEval/PoET-2") / best_info["model_path"]

    if not model_path.exists():
        incomplete_checkpoints.append((ckpt_dir.name, f"missing {model_path.name}"))
        continue

    complete_checkpoints.append(ckpt_dir.name)

print(f"Complete checkpoints: {len(complete_checkpoints)}")
for ckpt in complete_checkpoints:
    print(f"  {ckpt}")

print(f"\nIncomplete checkpoints: {len(incomplete_checkpoints)}")
for ckpt, reason in incomplete_checkpoints:
    print(f"  {ckpt}: {reason}")

# Save complete list
with open("/home/kfurui/workspace/FLAb/poet2_complete_checkpoints.txt", "w") as f:
    for ckpt in complete_checkpoints:
        f.write(f"{ckpt}\n")

print(f"\nComplete checkpoint list saved to: poet2_complete_checkpoints.txt")
