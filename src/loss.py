import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.07):
    """
    Numerically stable InfoNCE loss using CrossEntropy.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    
    # Similarity matrix
    logits = torch.mm(z, z.t()) / temperature
    
    # Create mask to remove self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    logits = logits.masked_fill(mask, -1e9)
    
    # For every sample i, the positive pair is (i + batch_size) or (i - batch_size)
    targets = torch.arange(2 * batch_size, device=z.device)
    targets = (targets + batch_size) % (2 * batch_size)
    
    return F.cross_entropy(logits, targets)
