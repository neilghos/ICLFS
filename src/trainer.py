import numpy as np
import torch

from data import get_dataset_bundle
from loss import contrastive_loss
from models import InvertedFeatureExpert
from visualization import visualize_feature_clusters
import data_loaders  


def train_icl(
    model,
    train_loader,
    epochs=1,
    top_k=100,
    plot_filename=None,
    dataset_name="madelon",
    seed=42,
    visualize_each_epoch=False
):
    """
    Train the Inverted Contrastive Learning model on a dataset.
    """
    import random
    feature_names = None

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
        if visualize_each_epoch and plot_filename is not None:
            visualize_feature_clusters(
                model,
                train_loader,
                feature_names=feature_names,
                filename=plot_filename,
                dataset_name=dataset_name,
                top_k=top_k,
            )

    print(f"{dataset_name.capitalize()} Training Complete.")

    if plot_filename is not None:
        visualize_feature_clusters(
            model,
            train_loader,
            feature_names=feature_names,
            filename=plot_filename,
            dataset_name=dataset_name,
            top_k=top_k,
        )


if __name__ == "__main__":
    dataset_name = "madelon"
    bundle = get_dataset_bundle(dataset_name)
    model = InvertedFeatureExpert(n_patients=bundle.num_samples)
    train_icl(
        model,
        bundle.train_loader,
        epochs=1,
        top_k=20,
        dataset_name=dataset_name,
    )
