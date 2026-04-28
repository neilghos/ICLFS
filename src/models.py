import torch
import torch.nn as nn

class InvertedFeatureExpert(nn.Module):
    def __init__(self, n_patients, latent_dim=512, n_heads=5):
        super().__init__()

        # 1. Relational Attention Layer
        # Input: (D_features, N_patients)
        self.attention = nn.MultiheadAttention(embed_dim=n_patients, num_heads=n_heads)
        
        # 2. Encoder Backbone (Distills the 'Identity')
        self.encoder = nn.Sequential(
            nn.Linear(n_patients, 1024),
            nn.BatchNorm1d(1024,momentum=0.01, eps=1e-5),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim)
        )
        
        # 3. Projection Head (For the Contrastive Task)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-5),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        # x shape: [D_features, N_patients]
        
        # Self-Attention across features to find relational dependencies
        # Needs shape [Seq_Len, Batch, Dim] -> [D, 1, N]
        x_attn = x.unsqueeze(1) 
        attn_out, _ = self.attention(x_attn, x_attn, x_attn)
        x = attn_out.squeeze(1)
        
        # Extract Latent Basis
        h = self.encoder(x)
        
        # Map to Hypersphere
        z = self.projector(h)
        return h, z
