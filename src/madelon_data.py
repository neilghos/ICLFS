import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import pandas as pd
import numpy as np

class MadelonDataset(Dataset):
    """
    Inverted Madelon: [500 features, N_patients].
    """
    def __init__(self, x, inverted=True):
        if inverted:
            self.x = torch.from_numpy(x.T).float()
        else:
            self.x = torch.from_numpy(x).float()
            
        self.inverted = inverted
        self.n_patients = self.x.shape[1] if inverted else None
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        if not self.inverted:
            return self.x[idx]
            
        feat_vector = self.x[idx]
        mask1 = (torch.rand(self.n_patients) > 0.15).float()
        mask2 = (torch.rand(self.n_patients) > 0.15).float()
        
        return feat_vector * mask1, feat_vector * mask2

def get_madelon_loaders(batch_size=128, random_state=42):
    """
    Loads Madelon dataset from the local ARFF file.
    No simulation, no network fallbacks.
    """
    # Load from the official local ARFF file provided by the user
    path = "/home/utsab/Desktop/ICL/phpfLuQE4.arff"
    print(f"Loading Madelon from {path}...")
    
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    
    # Last column is the target
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    
    # Convert labels to [0, 1] if they are binary strings or other types
    if isinstance(y[0], bytes):
        y = y.astype(str).astype(int)
    y = np.where(y <= 0, 0, 1) if np.min(y) < 0 else y
    if np.min(y) == 1: y -= 1
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Datasets
    train_ds = MadelonDataset(X_train, inverted=True)
    val_ds = MadelonDataset(X_val, inverted=False)
    test_ds = MadelonDataset(X_test, inverted=False)
    
    # Expert sees all 500 features at once
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, (y_train, y_val, y_test)

if __name__ == "__main__":
    train_l, val_l, test_l, labels = get_madelon_loaders()
    v1, v2 = next(iter(train_l))
    print(f"Madelon Inverted Feature Batch: {v1.shape}")
    
    x_val = next(iter(val_l))
    print(f"Madelon Standard Patient Batch: {x_val.shape}")
