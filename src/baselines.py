import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from sklearn.decomposition import FactorAnalysis, FastICA, KernelPCA, PCA, SparsePCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


class DeepVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def _vae_loss(recon, x, mu, logvar, beta=0.1):
    recon_loss = F.mse_loss(recon, x)
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + beta * kld


def _make_loader(x, batch_size, shuffle):
    tensor_x = torch.FloatTensor(x)
    return DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=shuffle)


def _evaluate_ae(model, x, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).to(device)
        recon, _ = model(x_tensor)
        return F.mse_loss(recon, x_tensor).item()


def _evaluate_vae(model, x, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).to(device)
        recon, mu, logvar = model(x_tensor)
        return _vae_loss(recon, x_tensor, mu, logvar).item()


def train_autoencoder(
    model,
    x_train,
    x_val,
    *,
    model_type,
    lr=1e-3,
    weight_decay=1e-5,
    batch_size=64,
    max_epochs=400,
    patience=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    train_loader = _make_loader(x_train, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    stale_epochs = 0

    for _ in range(max_epochs):
        model.train()
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()

            if model_type == "ae":
                recon, _ = model(batch_x)
                loss = F.mse_loss(recon, batch_x)
            else:
                recon, mu, logvar = model(batch_x)
                loss = _vae_loss(recon, batch_x, mu, logvar)

            loss.backward()
            optimizer.step()

        if model_type == "ae":
            val_loss = _evaluate_ae(model, x_val, device)
        else:
            val_loss = _evaluate_vae(model, x_val, device)

        scheduler.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model.cpu().eval(), best_val


def get_latent_dims(dataset_name):
    if dataset_name == "madelon":
        return [20]
    return [10, 20, 30]


def train_deep_rivals(x_train, x_val, latent_dims=(20,), seed=42):
    set_seed(seed)
    input_dim = x_train.shape[1]

    best_ae = {"model": None, "val_loss": float("inf"), "latent_dim": None}
    best_vae = {"model": None, "val_loss": float("inf"), "latent_dim": None}

    for latent_dim in latent_dims:
        ae, ae_val = train_autoencoder(
            DeepAE(input_dim, latent_dim=latent_dim),
            x_train,
            x_val,
            model_type="ae",
        )
        if ae_val < best_ae["val_loss"]:
            best_ae = {"model": ae, "val_loss": ae_val, "latent_dim": latent_dim}

        vae, vae_val = train_autoencoder(
            DeepVAE(input_dim, latent_dim=latent_dim),
            x_train,
            x_val,
            model_type="vae",
        )
        if vae_val < best_vae["val_loss"]:
            best_vae = {"model": vae, "val_loss": vae_val, "latent_dim": latent_dim}

    return best_ae, best_vae


def get_judge_grid():
    return {
        "XGBoost": [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        ],
        "SVM-RBF": [
            {"C": 0.1, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 1.0, "gamma": "auto"},
        ],
    }


def build_judge(judge_name, params, seed=42):
    if judge_name == "XGBoost":
        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            **params,
        )
    return SVC(kernel="rbf", random_state=seed, **params)


def tune_and_score_judge(judge_name, x_train, y_train, x_val, y_val, x_test, y_test, seed=42):
    best_params = None
    best_val_acc = -1.0

    for params in get_judge_grid()[judge_name]:
        judge = build_judge(judge_name, params, seed=seed)
        judge.fit(x_train, y_train)
        val_acc = accuracy_score(y_val, judge.predict(x_val))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    final_judge = build_judge(judge_name, best_params, seed=seed)
    final_judge.fit(x_train_full, y_train_full)
    test_acc = accuracy_score(y_test, final_judge.predict(x_test))
    return test_acc, best_val_acc, best_params


def make_classical_reducers(dataset_name="madelon", latent_dims=None):
    reducers = []
    if latent_dims is None:
        latent_dims = get_latent_dims(dataset_name)

    for n_components in latent_dims:
        reducers.extend(
            [
                (
                    f"PCA (Linear) [d={n_components}]",
                    PCA(n_components=n_components),
                ),
                (
                    f"Sparse PCA [d={n_components}]",
                    SparsePCA(n_components=n_components, random_state=42),
                ),
                (
                    f"Factor Analysis [d={n_components}]",
                    FactorAnalysis(n_components=n_components),
                ),
                (
                    f"ICA (FastICA) [d={n_components}]",
                    FastICA(n_components=n_components, random_state=42, max_iter=1000),
                ),
                (
                    f"KernelPCA (RBF) [d={n_components}]",
                    KernelPCA(n_components=n_components, kernel="rbf", gamma=None),
                ),
            ]
        )

    return reducers


def make_umap_reducers(n_components=20):
    reducers = []
    for n_neighbors in [15, 30, 50]:
        for min_dist in [0.0, 0.1, 0.5]:
            for metric in ["euclidean", "cosine"]:
                for seed in [42, 52]:
                    name = (
                        f"UMAP [n={n_neighbors}, min_dist={min_dist}, "
                        f"metric={metric}, seed={seed}]"
                    )
                    reducers.append(
                        (
                            name,
                            umap.UMAP(
                                n_components=n_components,
                                n_neighbors=n_neighbors,
                                min_dist=min_dist,
                                metric=metric,
                                random_state=seed,
                            ),
                        )
                    )
    return reducers


def encode_autoencoder(model, x):
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x)
        _, z = model(x_tensor)
    return z.numpy()


def encode_vae(model, x):
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x)
        _, mu, _ = model(x_tensor)
    return mu.numpy()


def evaluate_reducer(name, reducer, x_train, y_train, x_val, y_val, x_test, y_test):
    x_tr = reducer.fit_transform(x_train)
    x_va = reducer.transform(x_val)
    x_te = reducer.transform(x_test)

    results = []
    for judge_name in get_judge_grid():
        test_acc, val_acc, best_params = tune_and_score_judge(
            judge_name,
            x_tr,
            y_train,
            x_va,
            y_val,
            x_te,
            y_test,
        )
        results.append(
            {
                "Method": name,
                "Judge": judge_name,
                "ValAccuracy": val_acc,
                "Accuracy": test_acc,
                "BestJudgeParams": str(best_params),
            }
        )
    return results


def run_baseline_sweep(x_train, y_train, x_val, y_val, x_test, y_test, dataset_name="madelon", seed=42):
    """
    Runs baseline reducers with validation-tuned downstream judges and caches results.
    """
    print(f"\n--- INITIATING BASELINE SWEEP: {dataset_name.upper()} ---")
    set_seed(seed)
    results = []

    for name, reducer in make_classical_reducers(dataset_name):
        results.extend(evaluate_reducer(name, reducer, x_train, y_train, x_val, y_val, x_test, y_test))

    for name, reducer in make_umap_reducers():
        results.extend(evaluate_reducer(name, reducer, x_train, y_train, x_val, y_val, x_test, y_test))

    best_ae, best_vae = train_deep_rivals(
        x_train,
        x_val,
        latent_dims=tuple(get_latent_dims(dataset_name)),
        seed=seed,
    )
    deep_models = [
        (f"Deep AE [d={best_ae['latent_dim']}]", best_ae["model"], encode_autoencoder),
        (f"Deep VAE [d={best_vae['latent_dim']}]", best_vae["model"], encode_vae),
    ]

    for name, model, encoder in deep_models:
        x_tr = encoder(model, x_train)
        x_va = encoder(model, x_val)
        x_te = encoder(model, x_test)
        for judge_name in get_judge_grid():
            test_acc, val_acc, best_params = tune_and_score_judge(
                judge_name,
                x_tr,
                y_train,
                x_va,
                y_val,
                x_te,
                y_test,
                seed=seed,
            )
            results.append(
                {
                    "Method": name,
                    "Judge": judge_name,
                    "ValAccuracy": val_acc,
                    "Accuracy": test_acc,
                    "BestJudgeParams": str(best_params),
                }
            )

    df = pd.DataFrame(results).sort_values(["Judge", "Accuracy"], ascending=[True, False])
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/baselines_{dataset_name}.csv", index=False)
    print(f"Baseline Sweep Complete. Saved to results/baselines_{dataset_name}.csv")
    return df


if __name__ == "__main__":
    from data import get_dataset_bundle

    bundle = get_dataset_bundle("madelon")
    run_baseline_sweep(
        bundle.x_train,
        bundle.y_train,
        bundle.x_val,
        bundle.y_val,
        bundle.x_test,
        bundle.y_test,
        dataset_name="madelon",
    )
