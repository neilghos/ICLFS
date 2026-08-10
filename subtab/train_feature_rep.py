import argparse
import gc
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
from pathlib import Path

# Automatically detect and insert the repository root to sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from subtab.tabular_eval import load_dataset_bundle, TABM_DATASET_REGISTRY, TABM_DATA_ROOT, RESULTS_DIR
from src.feature_rep import HierarchicalFeatureRep

SEMANTIC_FEATURE_NAMES = {
    "california": [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
        "Population", "AveOccup", "Latitude", "Longitude"
    ],
    "churn": [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", 
        "HasCrCard", "IsActiveMember", "Gender_Female", "Gender_Male",
        "Geo_France", "Geo_Germany", "Geo_Spain"
    ],
    "house": [
        "Elevation", "Latitude", "Longitude", "Slope", "Aspect", "DistCoast",
        "DistHighway", "DistRail", "Income", "HouseAge", "Rooms", "Bedrooms",
        "Population", "Occupancy", "Feature14", "Feature15"
    ],
    "diamond": (
        ["carat", "depth", "table", "x", "y", "z"] + 
        [f"cut_{i}" for i in range(5)] + 
        [f"color_{i}" for i in range(7)] + 
        [f"clarity_{i}" for i in range(8)]
    ),
    "adult": (
        ["Age", "fnlwgt", "EducationNum", "CapitalGain", "CapitalLoss", "HoursPerWeek", "Sex"] +
        [f"workclass_{i}" for i in range(9)] + 
        [f"education_{i}" for i in range(16)] + 
        [f"marital_{i}" for i in range(7)] + 
        [f"occupation_{i}" for i in range(15)] + 
        [f"relationship_{i}" for i in range(6)] + 
        [f"race_{i}" for i in range(5)] + 
        [f"country_{i}" for i in range(42)]
    ),
    "higgs": [
        "lepton_pT", "lepton_eta", "lepton_phi", "missing_energy_mag", "missing_energy_phi",
        "MET_rel", "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",
        "jet1_pT", "jet1_eta", "jet1_phi", "jet1_btag", "jet2_pT", "jet2_eta", "jet2_phi", "jet2_btag",
        "jet3_pT", "jet3_eta", "jet3_phi", "jet3_btag", "jet4_pT", "jet4_eta", "jet4_phi"
    ],
    "otto": [f"feat_{i}" for i in range(1, 94)]
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Pretrain Hierarchical Feature Representations with Local Correlations")
    parser.add_argument(
        "--dataset",
        default="california",
        choices=sorted(TABM_DATASET_REGISTRY.keys()),
        help="Dataset name to pretrain feature representations for."
    )
    parser.add_argument("--tabm-root", type=Path, default=TABM_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chunk-size", type=int, default=64, help="Size of contiguous sample blocks.")
    parser.add_argument("--d-token", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=100, help="Optimization steps per epoch.")
    parser.add_argument("--batch-blocks", type=int, default=32, help="Number of sample blocks to sample per batch.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset: {args.dataset}")
    bundle = load_dataset_bundle(args.tabm_root, args.dataset, seed=42)
    x_train = bundle.x_train # [N, D]
    num_samples, num_features = x_train.shape
    chunk_size = args.chunk_size

    # Calculate number of blocks K
    num_blocks = max(2, num_samples // chunk_size)
    print(f"Dataset has N={num_samples} samples. Partitioning into K={num_blocks} blocks of size ~{num_samples/num_blocks:.1f}...")

    # 1. Precompute local correlation matrices R[m] of shape [num_features, num_features] for each block m
    print("Precomputing local correlation matrices...")
    R_list = []
    block_indices_split = np.array_split(np.arange(num_samples), num_blocks)
    
    for m in range(num_blocks):
        block_idx = block_indices_split[m]
        block_data = x_train[block_idx]
        df = pd.DataFrame(block_data)
        corr_matrix = df.corr(method="pearson").abs().fillna(0.0).values
        R_list.append(torch.from_numpy(corr_matrix).float().to(device))
    
    # Pack local correlations to a single tensor: [K, D, D]
    R = torch.stack(R_list, dim=0)

    # Initialize model with K chunks
    model = HierarchicalFeatureRep(num_features, num_blocks, args.d_token).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training Loop
    print("Starting optimization of chunk embeddings...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for step in range(args.steps_per_epoch):
            optimizer.zero_grad()
            
            # Randomly sample features and blocks to construct a dense correlation batch
            batch_features = list(range(num_features)) # Always align all features for small datasets
            if len(batch_features) > 16:
                batch_features = random.sample(batch_features, 16)
                
            sampled_blocks = random.sample(range(num_blocks), min(num_blocks, args.batch_blocks))
            
            # Get normalized embeddings: [D, K, d_token]
            E_norm = model.get_normalized_embeddings()
            
            # Slice embeddings for our batch: [len(batch_features), len(sampled_blocks), d_token]
            E_batch = E_norm[batch_features][:, sampled_blocks]
            
            # Flatten to [len(batch_features) * len(sampled_blocks), d_token]
            BF = len(batch_features)
            SB = len(sampled_blocks)
            E_flat = E_batch.view(BF * SB, args.d_token)
            
            # Compute cosine similarities for the batch: [BF * SB, BF * SB]
            S = torch.matmul(E_flat, E_flat.T)
            
            # Construct batch target similarities T using vectorized PyTorch operations
            feat_tensor = torch.tensor(batch_features, device=device)
            block_tensor = torch.tensor(sampled_blocks, device=device)
            
            # Create grid indices of shape [BF * SB]
            feat_ids = feat_tensor.unsqueeze(1).repeat(1, SB).view(-1)
            block_ids = block_tensor.repeat(BF)
            
            # Grids of shape [BF * SB, BF * SB]
            feat_grid_j = feat_ids.unsqueeze(1).repeat(1, BF * SB)
            feat_grid_k = feat_ids.unsqueeze(0).repeat(BF * SB, 1)
            
            block_grid_c = block_ids.unsqueeze(1).repeat(1, BF * SB)
            block_grid_d = block_ids.unsqueeze(0).repeat(BF * SB, 1)
            
            same_feat = (feat_grid_j == feat_grid_k)
            same_block = (block_grid_c == block_grid_d)
            
            T = torch.zeros(BF * SB, BF * SB, device=device)
            
            # Rule A: Same feature, different blocks (compact similarity = 0.90)
            T[same_feat] = 0.90
            
            # Rule B: Different features, same block (align with local correlation)
            R_vals = R[block_grid_c, feat_grid_j, feat_grid_k]
            T[same_block & ~same_feat] = R_vals[same_block & ~same_feat]
            
            # Diagonal is 1.0 (Same feature, same block)
            T.fill_diagonal_(1.0)
                                    
            loss = nn.MSELoss()(S, T)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / args.steps_per_epoch
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Loss: {avg_loss:.6f}")

    # Save learned representations
    save_path = args.out_dir / f"{args.dataset}_feature_rep.pt"
    model.save_embeddings(str(save_path))
    print(f"Saved chunk embeddings to: {save_path}")

    # 3. Visualization Plotter (Dual 3D PCA Subplots)
    print("Projecting embeddings for dual 3D visualization...")
    model.eval()
    with torch.no_grad():
        E_norm = model.get_normalized_embeddings().cpu().numpy() # [D, K, d_token]
        
    # Get feature names from semantic lookup or metadata fallback
    feature_names = SEMANTIC_FEATURE_NAMES.get(args.dataset.lower(), None)
    if feature_names is None or len(feature_names) != num_features:
        feature_names = bundle.metadata.get("output_feature_names", [f"Feature {i}" for i in range(num_features)])
        if len(feature_names) != num_features:
            feature_names = [f"Feature {i}" for i in range(num_features)]

    # Compute pooled feature embeddings: [D, d_token]
    E_feat = E_norm.mean(axis=1)
    
    # Project pooled features to 3D
    pca_feat = PCA(n_components=3)
    coords_feat = pca_feat.fit_transform(E_feat) # [D, 3]
    
    # Flatten and project all chunks to 3D
    E_flat = E_norm.reshape(num_features * num_blocks, args.d_token)
    pca_chunks = PCA(n_components=3)
    coords_chunks = pca_chunks.fit_transform(E_flat) # [D * K, 3]
    
    fig = plt.figure(figsize=(22, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_features))
    
    # Subplot 1: Feature Cones (Averaged Chunk Embeddings)
    ax1 = fig.add_subplot(121, projection='3d')
    for j in range(num_features):
        color = colors[j]
        # Plot pooled feature center
        ax1.scatter(coords_feat[j, 0], coords_feat[j, 1], coords_feat[j, 2], color=color, s=150, edgecolors='black')
        # Draw dashed line from origin to show the direction cone
        ax1.plot([0, coords_feat[j, 0]], [0, coords_feat[j, 1]], [0, coords_feat[j, 2]], color=color, linestyle="--", alpha=0.6)
        # Annotate with actual feature name
        ax1.text(
            coords_feat[j, 0], coords_feat[j, 1], coords_feat[j, 2],
            f"  {feature_names[j]}",
            fontsize=9,
            color="black",
            weight='bold'
        )
    ax1.set_title("Global Feature Cones (Averaged Embeddings)", fontsize=14, pad=15)
    ax1.set_xlabel("PCA 1", fontsize=10)
    ax1.set_ylabel("PCA 2", fontsize=10)
    ax1.set_zlabel("PCA 3", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # Subplot 2: Chunk Cloud (Grouped by Feature ID)
    ax2 = fig.add_subplot(122, projection='3d')
    for j in range(num_features):
        color = colors[j]
        start_idx = j * num_blocks
        end_idx = start_idx + num_blocks
        f_coords = coords_chunks[start_idx:end_idx]
        
        # Plot chunk cloud points (smaller, no text to prevent clutter)
        ax2.scatter(f_coords[:, 0], f_coords[:, 1], f_coords[:, 2], color=color, label=feature_names[j], s=25, alpha=0.7)
        # Draw bounding loop connecting chunks
        loop_coords = np.vstack([f_coords, f_coords[0]])
        ax2.plot(loop_coords[:, 0], loop_coords[:, 1], loop_coords[:, 2], color=color, linestyle="-", alpha=0.3)
        
    ax2.set_title("Local Chunk Clouds on Hypersphere", fontsize=14, pad=15)
    ax2.set_xlabel("PCA 1", fontsize=10)
    ax2.set_ylabel("PCA 2", fontsize=10)
    ax2.set_zlabel("PCA 3", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Features")
    
    plt.suptitle(f"Hierarchical Feature Manifold Alignment - {args.dataset.capitalize()}", fontsize=16, y=0.98)
    plt.tight_layout()
    
    plot_path = args.out_dir / f"{args.dataset}_feature_rep_alignment.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dual 3D visualization plot to: {plot_path}")

if __name__ == "__main__":
    main()
