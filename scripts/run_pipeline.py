"""Run the revised experiment serially, with validated caches for restart."""
from pathlib import Path
import subprocess
import sys
import yaml

root=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((root/"configs/revised.yaml").read_text())
def run(*args):
    subprocess.run([sys.executable,*args],cwd=root,check=True)

for seed in cfg["training_seeds"]:
    run("scripts/run_revised.py","train","--seed",str(seed))
for seed in cfg["test_seeds"]:
    run("scripts/run_revised.py","prepare","--test-seed",str(seed))
for train in cfg["training_seeds"]:
    for test in cfg["test_seeds"]:
        run("scripts/run_revised.py","evaluate","--train-seed",str(train),"--test-seed",str(test))
run("scripts/analyze_revised.py")
run("scripts/measure_revised_latency.py")
