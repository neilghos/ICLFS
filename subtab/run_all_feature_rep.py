import subprocess
import os
import sys
from pathlib import Path

# Automatically detect and insert the repository root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from subtab.tabular_eval import DEV_DATASETS

def main():
    print(f"Found {len(DEV_DATASETS)} dev datasets to pretrain.")
    
    # Copy existing environment variables to preserve PATH and other active conda variables
    env = os.environ.copy()
    env["PYTHONPATH"] = repo_root
    
    script_dir = Path(__file__).resolve().parent
    train_script = str(script_dir / "train_feature_rep.py")
    
    for idx, dataset in enumerate(DEV_DATASETS, 1):
        print(f"\n[{idx}/{len(DEV_DATASETS)}] Pretraining feature representations for: {dataset}")
        cmd = [
            sys.executable, train_script,
            "--dataset", dataset,
            "--chunk-size", "64"
        ]
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"Successfully finished: {dataset}")
        except subprocess.CalledProcessError as e:
            print(f"Error pretraining {dataset}: {e}")

if __name__ == "__main__":
    main()
