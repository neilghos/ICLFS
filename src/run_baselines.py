import subprocess
import pandas as pd
from pathlib import Path

DATASETS = ["gisette", "pixraw10p"]
METHODS = ["LS", "MCFS", "NDFS", "CAE", "LSCAE"] # Full baseline stack

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "all_baselines_summary.csv"

def run_baseline(dataset):
    print(f"\n>>> RUNNING BASELINES FOR: {dataset}")
    # Path-aware resolution for baseline.py
    base_script = Path(__file__).parent / "baseline.py"
    cmd = [
        "/data/conda_envs/torch/bin/python", str(base_script),
        "--dataset", dataset,
        "--methods", *METHODS,
        "--kmeans-runs", "20"
    ]
    try:
        subprocess.run(cmd, check=True)
        summary_path = RESULTS_DIR / f"baseline_{dataset}_summary.csv"
        if summary_path.exists():
            return pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error running {dataset}: {e}")
    return None

def main():
    all_summaries = []
    for ds in DATASETS:
        df = run_baseline(ds)
        if df is not None:
            all_summaries.append(df)
            # Save progress
            pd.concat(all_summaries).to_csv(SUMMARY_FILE, index=False)

    print(f"\n✅ All Baselines Complete. Results saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
