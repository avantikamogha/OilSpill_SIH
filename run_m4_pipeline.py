from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
AIS_SRC = ROOT / "AIS" / "src"

files = [
    "cleaning.py",
    "gap_analysis.py",
    "spatial_filter.py",
    "trajectory.py",
    "candidate_gap_analysis.py",
    "features.py",
    "evidence.py"
]

for file in files:

    print("\n" + "=" * 50)
    print(f"Running {file}")
    print("=" * 50)

    result = subprocess.run(
        [
            sys.executable,
            str(AIS_SRC / file)
        ],
        cwd=ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{file} failed."
        )

print("\n" + "=" * 50)
print("M4 PIPELINE COMPLETE")
print("=" * 50)