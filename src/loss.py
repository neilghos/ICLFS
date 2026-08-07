import torch
import torch.nn.functional as F

def decorrelation_loss(z):
    # Normalize feature embeddings before measuring inter-feature similarity.
    z = F.normalize(z, dim=1)
    batch_size = z.size(0)
    corr = torch.mm(z, z.t())
    identity = torch.eye(batch_size, device=z.device)
    loss = (corr - identity).pow(2).mean()
    return loss


def contrastive_loss(z1, z2, temperature=0.05, z_neg=None):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    logits = torch.mm(z1, z2.t()) / temperature  # Shape: [Batch, Batch]
    
    if z_neg is not None:
        z_neg = F.normalize(z_neg, dim=1)
        neg_sim = torch.sum(z1 * z_neg, dim=1, keepdim=True) / temperature # Shape: [Batch, 1]
        logits = torch.cat([logits, neg_sim], dim=1) # Shape: [Batch, Batch + 1]
        
    targets = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, targets)


def feature_wise_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """
    z1, z2: [B, D, d_proj] - Projected feature tokens for two augmented views of B samples.
    Computes per-feature InfoNCE loss over batch dimension [B, B] for each of the D features:
    - Positive pair: (z1[b, j], z2[b, j]) - same sample b for feature j.
    - Negatives: (z1[b, j], z2[m, j]) - different samples m != b for feature j.
    This avoids [B*D, B*D] memory explosion while preserving feature-wise contrastive alignment.
    """
    B, D, d = z1.shape
    z1_norm = F.normalize(z1, dim=-1)  # [B, D, d]
    z2_norm = F.normalize(z2, dim=-1)  # [B, D, d]

    # Batch matrix multiplication across D features: [D, B, d] x [D, d, B] -> [D, B, B]
    z1_perm = z1_norm.permute(1, 0, 2)  # [D, B, d]
    z2_perm = z2_norm.permute(1, 2, 0)  # [D, d, B]
    sim = torch.bmm(z1_perm, z2_perm) / temperature  # [D, B, B]

    targets = torch.arange(B, device=z1.device).unsqueeze(0).expand(D, -1)  # [D, B]
    loss = F.cross_entropy(sim.reshape(D * B, B), targets.reshape(D * B))
    return loss


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon):
    Pulls embeddings z of chunks belonging to the SAME class label together,
    and pushes embeddings of chunks belonging to DIFFERENT class labels apart.
    z: [B, d_proj]
    labels: [B] (integer class labels)
    """
    device = z.device
    z_norm = F.normalize(z, dim=1)
    batch_size = z.size(0)

    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # [B, B] positive mask

    # Similarity matrix
    anchor_dot_contrast = torch.div(torch.matmul(z_norm, z_norm.T), temperature)

    # For numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # Mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size, device=device).view(-1, 1),
        0,
    )
    mask = mask * logits_mask

    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    # Loss is negative mean log-likelihood
    loss = -mean_log_prob_pos.mean()
    return loss
# kbs
