import csv
from pathlib import Path

import numpy as np
import torch

from data import get_dataset_bundle
from extractor import get_feature_scores, get_topk_feature_indices
from loss import contrastive_loss
from models import InvertedFeatureExpert
from visualization import visualize_feature_clusters
import data_loaders  # Registers dataset adapters.


def _save_checkpoint(model, checkpoint_path, dataset_name, epochs):
    return


def _epoch_path(path_template, epoch):
    path = Path(path_template)
    return str(path.with_name(f"{path.stem}_epoch_{epoch:03d}{path.suffix}"))


def _default_checkpoint_path(dataset_name):
    return f"checkpoints/{dataset_name}_last.pt"


def _default_feature_list_path(dataset_name):
    return f"checkpoints/{dataset_name}_topk_features.csv"


def _default_plot_filename(dataset_name):
    return f"{dataset_name}_feature_manifold.png"


def _get_feature_names(dataset_name):
    if dataset_name == "sonar":
        from data_loaders.sonar_data import get_sonar_feature_names

        return get_sonar_feature_names()
    return None


def train_icl(
    model,
    train_loader,
    epochs=1,
    checkpoint_path=None,
    top_k=100,
    feature_list_path=None,
    plot_filename=None,
    dataset_name="madelon",
    seed=42,
    visualize_each_epoch=False,
    save_epoch_checkpoints=False,
    save_first_epoch_checkpoint=False,
):
    """
    Train the Inverted Contrastive Learning model on a dataset.
    """
    import random

    checkpoint_path = checkpoint_path or _default_checkpoint_path(dataset_name)
    feature_list_path = feature_list_path or _default_feature_list_path(dataset_name)
    plot_filename = plot_filename or _default_plot_filename(dataset_name)
    feature_names = _get_feature_names(dataset_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()

    print(f"Starting {dataset_name.capitalize()} Training on {device} ({epochs} Epochs)...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            views = [view.to(device) for view in batch]
            anchor = views[0]

            h_anchor, z_anchor = model(anchor)
            losses = []
            for positive_view in views[1:]:
                _, z_positive = model(positive_view)
                losses.append(contrastive_loss(z_anchor, z_positive))

            loss = torch.stack(losses).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(train_loader):.4f}")
        if visualize_each_epoch:
            visualize_feature_clusters(
                model,
                train_loader,
                feature_names=feature_names,
                filename=_epoch_path(plot_filename, epoch + 1),
                dataset_name=dataset_name,
                top_k=top_k,
            )

    print(f"{dataset_name.capitalize()} Training Complete.")

    visualize_feature_clusters(
        model,
        train_loader,
        feature_names=feature_names,
        filename=plot_filename,
        dataset_name=dataset_name,
        top_k=top_k,
    )

    feature_list_file = Path(feature_list_path)
    feature_list_file.parent.mkdir(parents=True, exist_ok=True)
    feature_scores = get_feature_scores(model, train_loader)
    top_k_indices = get_topk_feature_indices(feature_scores, top_k)
    with feature_list_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature_index", "score"])
        for rank, feature_idx in enumerate(top_k_indices, start=1):
            writer.writerow([rank, int(feature_idx), float(feature_scores[feature_idx])])
    print(f"Saved top-{top_k} feature list to {feature_list_file}")


if __name__ == "__main__":
    dataset_name = "madelon"
    bundle = get_dataset_bundle(dataset_name)
    model = InvertedFeatureExpert(n_patients=bundle.num_samples)
    train_icl(
        model,
        bundle.train_loader,
        epochs=1,
        checkpoint_path=f"checkpoints/{dataset_name}_last.pt",
        top_k=20,
        feature_list_path=f"checkpoints/{dataset_name}_topk_features.csv",
        dataset_name=dataset_name,
    )
