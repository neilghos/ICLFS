from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from data import get_dataset_bundle
from models import InvertedFeatureExpert
from extractor import get_feature_scores
import data_loaders


def load_topk_indices(csv_path):
    indices = []
    selected_scores = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["feature_index"]))
            selected_scores.append(float(row["score"]))
    return indices, selected_scores


def _anchor_view(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


dataset_name = "madelon"

bundle = get_dataset_bundle(dataset_name)
ckpt = torch.load(f"checkpoints/{dataset_name}_last.pt", map_location="cpu")

model = InvertedFeatureExpert(
    n_patients=bundle.num_train_samples,
    latent_dim=ckpt.get("latent_dim", 512),
    n_heads=ckpt.get("n_heads", 5),
)

state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

plot_dir = Path(f"Plots/{dataset_name}")
plot_dir.mkdir(parents=True, exist_ok=True)

scores = get_feature_scores(model, bundle.train_loader)
hist_path = plot_dir / "feature_score_histogram.png"

topk_csv = Path(f"checkpoints/{dataset_name}_topk_features.csv")
topk_indices, topk_scores = load_topk_indices(topk_csv)

plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, edgecolor="black", alpha=0.8, label="All Features")
plt.scatter(
    topk_scores,
    [0.5] * len(topk_scores),
    color="red",
    s=50,
    zorder=3,
    label="Top 20 Features",
)
for score in topk_scores:
    plt.axvline(score, color="red", alpha=0.15, linewidth=1)
plt.title("Distribution of Feature Identity Scores")
plt.xlabel("L2 Norm")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(hist_path)
plt.close()
print(f"Saved to {hist_path.resolve()}")

device = next(model.parameters()).device
batch = next(iter(bundle.train_loader))
anchor = _anchor_view(batch)
h, _ = model(anchor.to(device))
h_2d = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(h.detach().cpu().numpy())

manifold_path = plot_dir / "feature_manifold_top20_overlay.png"
plt.figure(figsize=(12, 10))
plt.scatter(h_2d[:, 0], h_2d[:, 1], alpha=0.7, c="blue", edgecolors="k", label="All Features")
plt.scatter(
    h_2d[topk_indices, 0],
    h_2d[topk_indices, 1],
    alpha=0.95,
    c="red",
    edgecolors="k",
    s=80,
    zorder=3,
    label="Top 20 Features",
)

for feature_idx in topk_indices:
    plt.annotate(
        str(feature_idx),
        (h_2d[feature_idx, 0], h_2d[feature_idx, 1]),
        fontsize=8,
        color="darkred",
        xytext=(4, 4),
        textcoords="offset points",
    )

plt.title("ICL Feature Manifold with Top-20 Overlay")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(manifold_path)
plt.close()
print(f"Saved to {manifold_path.resolve()}")
