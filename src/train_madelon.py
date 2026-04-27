import torch
from pathlib import Path
import csv
from data import get_dataset_bundle
from models import InvertedFeatureExpert
from loss import contrastive_loss
from extractor import get_feature_scores, get_topk_feature_indices
from visualization import visualize_feature_clusters
import numpy as np
import data_loaders  # Registers dataset adapters.

def train_icl(
    model,
    train_loader,
    val_loader,
    test_loader,
    labels,
    epochs=100,
    checkpoint_path="checkpoints/madelon_last.pt",
    top_k=20,
    feature_list_path="checkpoints/madelon_topk_features.csv",
):
    """
    Train the Inverted Contrastive Learning model on Madelon.
    """
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Starting Madelon Training on {device} (100 Epochs)...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for v1, v2 in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            
            h, z = model(v1)
            h2, z2 = model(v2)
            
            loss = contrastive_loss(z, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(train_loader):.4f}")
    print("Madelon Training Complete.")

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset": "madelon",
            "epochs": epochs,
            "latent_dim": getattr(model.encoder[-1], "out_features", None),
            "n_heads": model.attention.num_heads,
        },
        checkpoint_file,
    )
    print(f"Saved checkpoint to {checkpoint_file}")

    visualize_feature_clusters(
        model,
        train_loader,
        filename="feature_manifold.png",
        dataset_name="madelon",
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
    bundle = get_dataset_bundle("madelon")
    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader
    labels = bundle.labels
    input_dim = bundle.num_train_samples
    
    model = InvertedFeatureExpert(n_patients=input_dim)
    train_icl(model, train_loader, val_loader, test_loader, labels, epochs=100)
