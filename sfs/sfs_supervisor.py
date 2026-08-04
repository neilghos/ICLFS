from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.augmentor import build_augmentor
from src.loss import contrastive_loss, decorrelation_loss
from src.models import InvertedFeatureExpert
from src.runtime_config import get_augmentor_config
from evaluation_model import ExtraTree_Model, KNN_Model, SVM_Model


class InvertedFeatureViewDataset(Dataset):
    """Feature-major multi-view dataset for the shared ICL backbone."""

    def __init__(self, x: np.ndarray, augmentor_config: dict | None = None):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32).T).float()
        self.n_patients = self.x.shape[1]
        self.augmentor = build_augmentor(
            n_patients=self.n_patients,
            strategy="four_view_mask",
            config=augmentor_config,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.augmentor(self.x[idx])


class FeatureScoreHead(nn.Module):
    """Maps per-feature embeddings to scalar saliency logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SampleClassifier(nn.Module):
    """Small classifier operating on gated sample-major inputs."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupervisedGateICLModel(nn.Module):
    """
    Wraps the shared ICL backbone with a learnable gate head and sample classifier.

    The gating path is:
    z_j -> a_j -> g_j = sigmoid(a_j / tau) -> x_i * g -> classifier
    """

    def __init__(
        self,
        *,
        n_patients: int,
        num_features: int,
        num_classes: int,
        latent_dim: int = 512,
        n_heads: int = 1,
        encoder_hidden_dim: int = 1024,
        projector_hidden_dim: int = 256,
        projector_out_dim: int = 128,
        gate_hidden_dim: int = 64,
        classifier_hidden_dim: int = 128,
        gate_temperature: float = 1.0,
    ):
        super().__init__()
        self.backbone = InvertedFeatureExpert(
            n_patients=n_patients,
            latent_dim=latent_dim,
            n_heads=n_heads,
            encoder_hidden_dim=encoder_hidden_dim,
            projector_hidden_dim=projector_hidden_dim,
            projector_out_dim=projector_out_dim,
        )
        self.score_head = FeatureScoreHead(projector_out_dim, hidden_dim=gate_hidden_dim)
        self.classifier = SampleClassifier(
            input_dim=num_features,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
        )
        self.gate_temperature = gate_temperature

    def encode_views(self, x: torch.Tensor):
        return self.backbone(x)

    def compute_gate_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.score_head(z)

    def compute_gates(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.compute_gate_logits(z)
        gates = torch.sigmoid(gate_logits / self.gate_temperature)
        return gate_logits, gates

    def classify_with_gates(
        self,
        anchor_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, z_anchor = self.backbone(anchor_features)
        gate_logits, gates = self.compute_gates(z_anchor)
        sample_major = anchor_features.transpose(0, 1)
        gated_samples = sample_major * gates.unsqueeze(0)
        logits = self.classifier(gated_samples)
        cls_loss = F.cross_entropy(logits, labels)
        return cls_loss, logits, gate_logits, gates

    def rank_features(self, anchor_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, z_anchor = self.backbone(anchor_features)
        gate_logits, gates = self.compute_gates(z_anchor)
        return gate_logits, gates


@dataclass
class SupervisedTrainingArtifacts:
    model: SupervisedGateICLModel
    feature_scores: np.ndarray
    feature_gates: np.ndarray
    best_validation_accuracy: float
    best_validation_k: int
    best_epoch: int


def build_inverted_view_loader(
    x: np.ndarray,
    *,
    config_path: str | None = None,
) -> DataLoader:
    train_ds = InvertedFeatureViewDataset(
        x,
        augmentor_config=get_augmentor_config(config_path),
    )
    return DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)


def _as_class_index(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


def _build_evaluation_model(
    model_name: str,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
):
    if model_name == "svm":
        return SVM_Model(x_train, y_train, x_eval, y_eval)
    if model_name == "knn":
        return KNN_Model(x_train, y_train, x_eval, y_eval)
    if model_name == "extratree":
        return ExtraTree_Model(x_train, y_train, x_eval, y_eval)
    raise ValueError(f"Unknown evaluation model '{model_name}'")


def _evaluate_ranking(
    ranking: np.ndarray,
    *,
    k_list: list[int],
    evaluation_model,
) -> tuple[float, int]:
    best_accuracy = float("-inf")
    best_k = -1
    for k_selected in k_list:
        selected_idx = np.asarray(ranking[:k_selected], dtype=int)
        accuracy = float(evaluation_model.train_and_test(selected_idx))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = int(k_selected)
    return best_accuracy, best_k


def train_supervised_gate_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    k_list: list[int],
    evaluation_model: str = "svm",
    epochs: int = 100,
    seed: int = 42,
    latent_dim: int = 512,
    n_heads: int = 1,
    encoder_hidden_dim: int = 1024,
    projector_hidden_dim: int = 256,
    projector_output_dim: int = 128,
    gate_hidden_dim: int = 64,
    classifier_hidden_dim: int = 128,
    gate_temperature: float = 1.0,
    temperature: float = 0.05,
    decorrelation_weight: float = 0.4,
    cls_weight: float = 1.0,
    config_path: str | None = None,
) -> SupervisedTrainingArtifacts:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = np.asarray(x, dtype=np.float32)
    x_valid = np.asarray(x_valid, dtype=np.float32)
    y_index = _as_class_index(y)
    y_valid_index = _as_class_index(y_valid)
    num_classes = int(np.unique(y_index).shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_inverted_view_loader(x, config_path=config_path)
    labels = torch.from_numpy(y_index).long().to(device)

    model = SupervisedGateICLModel(
        n_patients=x.shape[0],
        num_features=x.shape[1],
        num_classes=num_classes,
        latent_dim=latent_dim,
        n_heads=n_heads,
        encoder_hidden_dim=encoder_hidden_dim,
        projector_hidden_dim=projector_hidden_dim,
        projector_out_dim=projector_output_dim,
        gate_hidden_dim=gate_hidden_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        gate_temperature=gate_temperature,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    validation_evaluator = _build_evaluation_model(
        evaluation_model,
        x_train=x,
        y_train=y_index,
        x_eval=x_valid,
        y_eval=y_valid_index,
    )
    best_validation_accuracy = float("-inf")
    best_validation_k = -1
    best_epoch = -1
    best_state_dict = None

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            views = [view.to(device) for view in batch]
            anchor = views[0]
            pos_views = views[1:5]
            neg_view = views[5]

            optimizer.zero_grad()
            num_pos = len(pos_views)
            for v in pos_views:
                _, z_anchor = model.encode_views(anchor)
                _, z_neg = model.encode_views(neg_view)
                _, z_v = model.encode_views(v)

                l_con = contrastive_loss(z_anchor, z_v, temperature=temperature, z_neg=z_neg)
                l_div = decorrelation_loss(z_anchor)
                l_cls, _, _, _ = model.classify_with_gates(anchor, labels)
                total_loss = (l_con / num_pos) + (decorrelation_weight * l_div) + (cls_weight * l_cls)
                total_loss.backward()
                epoch_loss += total_loss.item()
            optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"SFE Supervisor Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f}")
            model.eval()
            with torch.no_grad():
                batch = next(iter(loader))
                anchor = batch[0].to(device)
                gate_logits, _ = model.rank_features(anchor)
            ranking = np.argsort(gate_logits.detach().cpu().numpy())[::-1]
            validation_accuracy, validation_k = _evaluate_ranking(
                ranking,
                k_list=k_list,
                evaluation_model=validation_evaluator,
            )
            print(
                f"Validation checkpoint | epoch={epoch:03d} "
                f"best_k={validation_k} accuracy={validation_accuracy:.6f}"
            )
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                best_validation_k = validation_k
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
            model.train()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        anchor = batch[0].to(device)
        gate_logits, gates = model.rank_features(anchor)
    feature_scores = gate_logits.detach().cpu().numpy()
    feature_gates = gates.detach().cpu().numpy()

    return SupervisedTrainingArtifacts(
        model=model,
        feature_scores=feature_scores,
        feature_gates=feature_gates,
        best_validation_accuracy=best_validation_accuracy,
        best_validation_k=best_validation_k,
        best_epoch=best_epoch,
    )


def rank_features_from_scores(feature_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(feature_scores, dtype=np.float64)
    return np.argsort(scores)[::-1]
