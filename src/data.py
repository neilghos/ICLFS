import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class BreastCancerDataset(Dataset):
    """
    Inverted Dataset: Samples are Features, not Patients.
    Each item returns two noisy views of a feature's profile across patients.
    """
    def __init__(self, x, inverted=True):
        if inverted:
            # Transpose: [N_patients, D_features] -> [D_features, N_patients]
            self.x = torch.from_numpy(x.T).float()
        else:
            self.x = torch.from_numpy(x).float()
            
        self.inverted = inverted
        self.n_patients = self.x.shape[1] if inverted else None
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        if not self.inverted:
            return self.x[idx] # Standard patient-wise return for evaluation
            
        feat_vector = self.x[idx]
        
        # MULTIVIEW LOGIC: Apply masking/dropout to patient responses
        # This forces the ICL to recognize the feature signature even with missing data
        mask1 = (torch.rand(self.n_patients) > 0.1).float()
        mask2 = (torch.rand(self.n_patients) > 0.1).float()
        
        v1 = feat_vector * mask1
        v2 = feat_vector * mask2
        
        return v1, v2

def get_dataloaders(batch_size=30, random_state=42):
    """
    Train Loader: Feature-wise (Inverted)
    Val/Test Loader: Patient-wise (Standard) for downstream PCA/LDA comparison.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 1. Standard Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # 2. Scaling (Fit only on Train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 3. Create Datasets
    # Training is INVERTED for the ICL Expert
    train_ds = BreastCancerDataset(X_train, inverted=True)
    # Val/Test are STANDARD for evaluation against PCA/LDA
    val_ds = BreastCancerDataset(X_val, inverted=False)
    test_ds = BreastCancerDataset(X_test, inverted=False)
    
    # 4. Create Loaders
    # batch_size=30 ensures the Attention Layer sees ALL features at once
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, (y_train, y_val, y_test)

if __name__ == "__main__":
    # Verification
    train_l, val_l, test_l, labels = get_dataloaders()
    
    v1, v2 = next(iter(train_l))
    print(f"--- INVERTED TRAINING (ICL EXPERT) ---")
    print(f"Feature-Batch shape: {v1.shape}") # Should be [30, N_train_patients]
    
    x_val = next(iter(val_l))
    print(f"\n--- STANDARD EVALUATION ---")
    print(f"Patient-Batch shape: {x_val.shape}") # Should be [batch_size, 30]
