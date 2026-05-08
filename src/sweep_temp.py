import subprocess
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
# 0.01 to 0.20 with 0.02 interval + 0.1 and 0.21 explicitly
TEMPS = sorted(list(set(np.arange(0.01, 0.21, 0.02).tolist() + [0.1, 0.21])))
DATASETS = {
    "coil20": {"heads": 1},
    "yale": {"heads": 5},
    "tox171": {"heads": 1},
    "prostate": {"heads": 1},
}

COMMON_DEFAULTS = {
    "latent_dim": 512,
    "epochs": 100,
    "diversity_weight": 0.005,
}

RESULTS_DIR = Path("src/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "temp_sweep_summary.csv"

def run_experiment(dataset, heads, temp):
    cmd = [
        "/data/conda_envs/torch/bin/python", "src/icl_eval.py",
        "--dataset", dataset,
        "--n-heads", str(heads),
        "--latent-dim", str(COMMON_DEFAULTS["latent_dim"]),
        "--epochs", str(COMMON_DEFAULTS["epochs"]),
        "--diversity-weight", str(COMMON_DEFAULTS["diversity_weight"]),
        "--temperature", f"{temp:.4f}",
        # No --temperature-end for static sweep
    ]
    
    print(f"\n>>> RUNNING: {dataset} | Temp: {temp:.4f} | Heads: {heads}")
    try:
        subprocess.run(cmd, check=True)
        result_path = RESULTS_DIR / f"icl_{dataset}_summary.csv"
        if result_path.exists():
            df = pd.read_csv(result_path)
            last_row = df.iloc[-1].to_dict()
            last_row["temperature"] = temp
            return last_row
    except Exception as e:
        print(f"Error running experiment: {e}")
    return None

def main():
    all_results = []
    
    for t in TEMPS:
        for ds_name, config in DATASETS.items():
            res = run_experiment(ds_name, config["heads"], t)
            if res:
                all_results.append(res)
                # Save intermediate results
                pd.DataFrame(all_results).to_csv(SUMMARY_FILE, index=False)

    print(f"\n✅ Temperature Sweep Complete. Results saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
