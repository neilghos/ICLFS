import subprocess
import pandas as pd
import os
from pathlib import Path

# --- Configuration ---
LATENT_DIMS = [512, 128, 256, 64, 1024, 2048]
DATASETS = {
    "coil20": {"heads": 1},
    "yale": {"heads": 5},
    "tox171": {"heads": 1},
    "prostate": {"heads": 1},
}

COMMON_DEFAULTS = {
    "epochs": 100,
    "diversity_weight": 0.005,
    "temp_start": 0.1,
    "temp_end": 0.01,
}

RESULTS_DIR = Path("src/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_FILE = RESULTS_DIR / "latent_sweep_summary.csv"

def run_experiment(dataset, heads, latent_dim):
    cmd = [
        "/data/conda_envs/torch/bin/python", "src/icl_eval.py",
        "--dataset", dataset,
        "--n-heads", str(heads),
        "--latent-dim", str(latent_dim),
        "--epochs", str(COMMON_DEFAULTS["epochs"]),
        "--diversity-weight", str(COMMON_DEFAULTS["diversity_weight"]),
        "--temperature", str(COMMON_DEFAULTS["temp_start"]),
        "--temperature-end", str(COMMON_DEFAULTS["temp_end"]),
    ]
    
    print(f"\n>>> RUNNING: {dataset} | Dim: {latent_dim} | Heads: {heads}")
    try:
        subprocess.run(cmd, check=True)
        # icl_eval saves results to results/icl_<dataset>_summary.csv
        # We read that and grab the last row
        result_path = RESULTS_DIR / f"icl_{dataset}_summary.csv"
        if result_path.exists():
            df = pd.read_csv(result_path)
            last_row = df.iloc[-1].to_dict()
            last_row["latent_dim"] = latent_dim
            return last_row
    except Exception as e:
        print(f"Error running experiment: {e}")
    return None

def main():
    all_results = []
    
    for dim in LATENT_DIMS:
        for ds_name, config in DATASETS.items():
            res = run_experiment(ds_name, config["heads"], dim)
            if res:
                all_results.append(res)
                # Save intermediate results
                pd.DataFrame(all_results).to_csv(SUMMARY_FILE, index=False)

    print(f"\n✅ Sweep Complete. Results saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
