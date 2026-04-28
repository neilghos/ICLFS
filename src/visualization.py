import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from sklearn.manifold import TSNE
from extractor import get_feature_scores, get_topk_feature_indices

PLOTS_DIR = Path("/home/utsab/Desktop/ICLFE/ICLFE/Plots")


def _resolve_plot_path(filename, dataset_name=None):
    plot_path = Path(filename)
    if not plot_path.is_absolute():
        base_dir = PLOTS_DIR / dataset_name if dataset_name else PLOTS_DIR
        plot_path = base_dir / plot_path
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    return plot_path


def _anchor_view(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


@torch.no_grad()
def visualize_feature_clusters(
    model,
    loader,
    feature_names=None,
    filename="feature_manifold.png",
    dataset_name=None,
    top_k=None,
    annotate_top_k=True,
):
    """
    Project feature embeddings to 2D for visualization.
    """
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(loader))
    anchor = _anchor_view(batch).to(device)
    h, _ = model(anchor)
    
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    h_2d = tsne.fit_transform(h.cpu().numpy())
    
    plt.figure(figsize=(12, 10))
    plt.scatter(
        h_2d[:, 0],
        h_2d[:, 1],
        alpha=0.7,
        c="blue",
        edgecolors="k",
        label="All Features",
    )

    if top_k is not None and top_k > 0:
        feature_scores = get_feature_scores(model, loader)
        top_k_indices = get_topk_feature_indices(feature_scores, top_k)
        plt.scatter(
            h_2d[top_k_indices, 0],
            h_2d[top_k_indices, 1],
            alpha=0.95,
            c="red",
            edgecolors="k",
            s=80,
            label=f"Top {top_k} Features",
            zorder=3,
        )

        if annotate_top_k:
            for feature_idx in top_k_indices:
                label = (
                    feature_names[feature_idx]
                    if feature_names is not None and feature_idx < len(feature_names)
                    else str(int(feature_idx))
                )
                plt.annotate(
                    label,
                    (h_2d[feature_idx, 0], h_2d[feature_idx, 1]),
                    fontsize=8,
                    color="darkred",
                    alpha=0.9,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
    
    if feature_names is not None and not (top_k is not None and top_k > 0):
        for i, name in enumerate(feature_names):
            plt.annotate(name, (h_2d[i, 0], h_2d[i, 1]), fontsize=9, alpha=0.8)
    else:
        # For high-D data, just label a few or leave blank
        pass
        
    plt.title("ICL Feature Manifold: Identifying Redundant Signaling")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True, alpha=0.3)
    if top_k is not None and top_k > 0:
        plt.legend()
    
    plot_path = _resolve_plot_path(filename, dataset_name=dataset_name)
    plt.savefig(plot_path)
    print(f"Saved feature manifold visualization to {plot_path}")
    plt.close()


@torch.no_grad()
def visualize_attention_heatmap(
    model,
    loader,
    title="Feature Attention Map",
    filename="attention_heatmap_log.png",
    dataset_name=None,
):
    """
    Visualize attention weights when the model forward path returns them.
    """
    model.eval()
    device = next(model.parameters()).device

    batch = next(iter(loader))
    anchor = _anchor_view(batch).to(device)
    outputs = model(anchor)
    if not isinstance(outputs, tuple) or len(outputs) < 3:
        raise ValueError("Model does not return attention weights needed for heatmap visualization.")

    attn_weights = outputs[2]
    attn_matrix = attn_weights.mean(dim=0).cpu().numpy()

    eps = 1e-8
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attn_matrix + eps,
        norm=colors.LogNorm(vmin=attn_matrix.min() + eps, vmax=attn_matrix.max()),
        cmap="magma",
        cbar=True,
    )
    plt.title(f"{title} (Log Scale - Multi-Head Avg)")
    plt.xlabel("Key Features")
    plt.ylabel("Query Features")

    plot_path = _resolve_plot_path(filename, dataset_name=dataset_name)
    plt.savefig(plot_path)
    print(f"Attention Heatmap saved to {plot_path}")
    plt.close()
