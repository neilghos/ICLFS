import torch
import torch.nn.functional as F

def contrastive_loss(z1, z2, temperature=0.1, z_neg=None):
    """
    Enhanced InfoNCE loss with optional hard negative support.
    If z_neg is provided, it is treated as a dedicated negative for its corresponding anchor.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Similarity between anchors (z1) and positives (z2)
    logits = torch.mm(z1, z2.t()) / temperature  # Shape: [Batch, Batch]
    
    if z_neg is not None:
        z_neg = F.normalize(z_neg, dim=1)
        # Similarity between each anchor and its specific negative view
        # We only care about the diagonal (anchor i vs negative i)
        neg_sim = torch.sum(z1 * z_neg, dim=1, keepdim=True) / temperature # Shape: [Batch, 1]
        
        # Add the specific negative similarity as an extra column in the logits
        # Now each row has [Batch] positive candidates and [1] hard negative candidate
        logits = torch.cat([logits, neg_sim], dim=1) # Shape: [Batch, Batch + 1]
        
    targets = torch.arange(z1.shape[0], device=z1.device)
    return F.cross_entropy(logits, targets)
