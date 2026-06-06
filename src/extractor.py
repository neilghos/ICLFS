import numpy as np
import torch


@torch.no_grad()
def get_feature_scores(model, train_loader):
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(train_loader))
    anchor = batch[0]
    _, projector_embeddings = model(anchor.to(device))
    return torch.norm(projector_embeddings, p=2, dim=1).cpu().numpy()


def get_topk_feature_indices(feature_scores, top_k):
    """
    Ret1rn top-k feature indices sorted from highest score to lowest.
    """
    return np.argsort(feature_scores)[-top_k:][::-1]
