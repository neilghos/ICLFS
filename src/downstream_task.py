import torch
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

@torch.no_grad()
def get_expert_feature_scores(model, train_loader):
    """
    Computes confidence scores for all features based on 64-dim embedding magnitude.
    """
    model.eval()
    device = next(model.parameters()).device
    v1, _ = next(iter(train_loader)) 
    h, _ = model(v1.to(device))
    
    # Calculate L2 norm for each feature in the 64-dim latent space
    scores = torch.norm(h, p=2, dim=1) 
    return scores.cpu().numpy()

@torch.no_grad()
def run_slam_dunk_battle(model, train_loader, val_loader, test_loader, labels):
    """
    Competitive evaluation: PCA (20) vs ICL-Expert Selection (20) using SVM-RBF.
    Judging both models on the exact same 20-dim budget.
    """
    y_train, y_val, y_test = labels
    model.eval()
    device = next(model.parameters()).device
    
    # --- 1. PREPARE DATA ---
    X_train_raw = train_loader.dataset.x.T.cpu().numpy() # [N, 500]
    X_test_raw = test_loader.dataset.x.cpu().numpy()     # [N, 500]

    # --- 2. THE PCA BASELINE (Direct k=20) ---
    pca_baseline = PCA(n_components=20)
    X_train_pca = pca_baseline.fit_transform(X_train_raw)
    X_test_pca = pca_baseline.transform(X_test_raw)

    # --- 3. THE ICL HYBRID MANIFOLD ---
    # Step 1: Extract Expert Scores from the 64-dim bottleneck
    expert_scores = get_expert_feature_scores(model, train_loader)
    
    # Step 2: Select Top 50 "Trusted" Features
    top_k_indices = np.argsort(expert_scores)[-50:]
    X_train_filtered = X_train_raw[:, top_k_indices]
    X_test_filtered = X_test_raw[:, top_k_indices]
    
    # Step 3: Scree-Reduction to the final 20-dim budget
    refiner = PCA(n_components=20)
    X_train_icl_hybrid = refiner.fit_transform(X_train_filtered)
    X_test_icl_hybrid = refiner.transform(X_test_filtered)

    # --- 4. THE DOWNSTREAM BATTLE (SVM-RBF on k=20) ---
    results = []
    for name, (X_tr, X_te) in [("PCA (Baseline)", (X_train_pca, X_test_pca)), 
                                ("ICL Hybrid (Ours)", (X_train_icl_hybrid, X_test_icl_hybrid))]:
        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        
        results.append({
            "Method": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "Final Dims": 20
        })

    # --- 5. REPORT ---
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("      THE CIKM SLAM DUNK BATTLE (20 Dims)")
    print("      (Hybrid Selection via 64-dim Expert)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    return df
