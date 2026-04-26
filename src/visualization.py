import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_breast_cancer

@torch.no_grad()
def visualize_feature_clusters(model, loader, feature_names=None, filename="feature_manifold.png"):
    """
    Project feature embeddings to 2D for visualization.
    """
    model.eval()
    device = next(model.parameters()).device
    v1, _ = next(iter(loader))
    v1 = v1.to(device)
    h, _ = model(v1)
    
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    h_2d = tsne.fit_transform(h.cpu().numpy())
    
    plt.figure(figsize=(12, 10))
    plt.scatter(h_2d[:, 0], h_2d[:, 1], alpha=0.7, c='blue', edgecolors='k')
    
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            plt.annotate(name, (h_2d[i, 0], h_2d[i, 1]), fontsize=9, alpha=0.8)
    else:
        # For high-D data, just label a few or leave blank
        pass
        
    plt.title("ICL Feature Manifold: Identifying Redundant Signaling")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig("feature_manifold.png")
    print("Saved feature manifold visualization to feature_manifold.png")
    plt.close()

@torch.no_grad()
def transform_patients(model, x_standard, train_loader):
    """
    Project standard patient vectors onto the learned Inverted Manifold.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Get the feature embeddings (the new basis)
    # The train_loader (inverted) contains the features as samples
    v_train, _ = next(iter(train_loader))
    E_f, _ = model(v_train.to(device)) 
    
    # 2. Project patients [N, 30] @ [30, 64] -> [N, 64]
    z_manifold = torch.mm(x_standard.to(device), E_f)
    
    return z_manifold.cpu().numpy()

@torch.no_grad()
def visualize_patient_clusters(model, data_loader, train_loader, y_labels, title="Patient Manifold", filename="patient_manifold.png"):
    """
    Visualize how patients cluster in the ICL-derived manifold.
    """
    # 1. Get ALL data to match label size
    x_list = []
    for x in data_loader:
        x_list.append(x)
    x_full = torch.cat(x_list, dim=0)
    
    # 2. Transform to ICL manifold
    z_full = transform_patients(model, x_full, train_loader)
    
    # 3. TSNE for visualization
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    z_2d = tsne.fit_transform(z_full)
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_labels, cmap='coolwarm', alpha=0.8)
    plt.colorbar(scatter, label='Diagnosis (0: Malignant, 1: Benign)')
    plt.title(title)
    plt.xlabel("ICL-TSNE 1")
    plt.ylabel("ICL-TSNE 2")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(filename)
    print(f"Saved {title} to {filename}")
    plt.close()
