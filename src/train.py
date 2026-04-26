import torch
from data import get_dataloaders
from models import InvertedFeatureExpert
from loss import contrastive_loss
from visualization import visualize_feature_clusters, visualize_patient_clusters
from downstream_task import run_slam_dunk_battle

def train_icl(model, train_loader, val_loader, test_loader, labels, epochs=300):
    """
    Train the Inverted Contrastive Learning model.
    """
    y_train, y_val, y_test = labels
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Starting Training on {device}...")
    
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
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch:03d} | Loss: {epoch_loss/len(train_loader):.4f}")
            
    print("Training Complete.")
    
    # 1. Feature Manifold Visualization
    print("Generating feature manifold visualization...")
    visualize_feature_clusters(model, train_loader)
    
    # 2. Validation Manifold Visualization
    print("Generating validation manifold visualization...")
    visualize_patient_clusters(model, val_loader, train_loader, y_val, 
                               title="Validation Manifold", filename="val_manifold.png")
    
    # 3. Test Manifold Visualization
    print("Generating test manifold visualization...")
    visualize_patient_clusters(model, test_loader, train_loader, y_test, 
                               title="Test Manifold", filename="test_manifold.png")
    
    # 4. Downstream Evaluation Battle
    run_slam_dunk_battle(model, train_loader, val_loader, test_loader, labels)

if __name__ == "__main__":
    # 1. Setup Data
    train_loader, val_loader, test_loader, labels = get_dataloaders()
    
    # 2. Setup Model
    sample_v1, _ = next(iter(train_loader))
    input_dim = sample_v1.shape[1]
    
    model = InvertedFeatureExpert(n_patients=input_dim)
    
    # 3. Start Training
    train_icl(model, train_loader, val_loader, test_loader, labels, epochs=300)
