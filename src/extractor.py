import numpy as np
import torch


def _anchor_view(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


@torch.no_grad()
def get_feature_embeddings(model, train_loader):
    """
    Return the learned embedding for each original feature using the anchor view.
    """
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(train_loader))
    anchor = _anchor_view(batch)
    embeddings, _ = model(anchor.to(device))
    return embeddings.cpu().numpy()


@torch.no_grad()
def get_feature_scores(model, train_loader):
    """
    Score each original feature by the norm of its learned embedding.
    """
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(train_loader))
    anchor = _anchor_view(batch)
    embeddings, _ = model(anchor.to(device))
    return torch.norm(embeddings, p=2, dim=1).cpu().numpy()


def get_topk_feature_indices(feature_scores, top_k):
    """
    Return top-k feature indices sorted from highest score to lowest.
    """
    return np.argsort(feature_scores)[-top_k:][::-1]


@torch.no_grad()
def extract_topk_features(model, train_loader, x_train, x_val, x_test, top_k):
    """
    Rank features with the model and slice the train/val/test matrices accordingly.
    """
    feature_scores = get_feature_scores(model, train_loader)
    top_k_indices = get_topk_feature_indices(feature_scores, top_k)
    return (
        x_train[:, top_k_indices],
        x_val[:, top_k_indices],
        x_test[:, top_k_indices],
        feature_scores,
        top_k_indices,
    )
