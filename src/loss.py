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
