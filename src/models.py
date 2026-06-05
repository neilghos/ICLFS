import torch
import torch.nn as nn

class ResidualProjector(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, out_dim=128):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-5),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-5),
            nn.ReLU()
        )
        self.l3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_hidden=False):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h_proj = h1 + h2
        z = self.l3(h_proj)
        if return_hidden:
            return h_proj, z
        return z


class InvertedFeatureExpert(nn.Module):
    def __init__(
        self,
        n_patients,
        latent_dim=512,
        n_heads=5,
        encoder_hidden_dim=1024,
        projector_hidden_dim=256,
        projector_out_dim=128,
    ):
        super().__init__()

        # 1. Relational Attention Layer
        # Input: (D_features, N_patients)
        self.attention = nn.MultiheadAttention(embed_dim=n_patients, num_heads=n_heads)
        
        # 2. Encoder Backbone (Distills the 'Identity')
        self.encoder = nn.Sequential(
            nn.Linear(n_patients, encoder_hidden_dim),
            nn.BatchNorm1d(encoder_hidden_dim, momentum=0.01, eps=1e-5),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_dim, latent_dim)
        )
        
        # 3. Residual Projection Head (Ablation #1)
        self.projector = ResidualProjector(
            latent_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )

    def forward(self, x, return_attn=False, return_projector_hidden=False):
        # x shape: [D_features, N_patients]
        
        # Self-Attention across features to find relational dependencies
        x_attn = x.unsqueeze(1) 
        attn_out, attn_weights = self.attention(x_attn, x_attn, x_attn)
        x_mixed = attn_out.squeeze(1)
        
        # Encoder Backbone
        h = self.encoder(x_mixed)
        
        # Residual Projector
        if return_projector_hidden:
            h_proj, z = self.projector(h, return_hidden=True)
        else:
            z = self.projector(h)
        
        if return_attn:
            if return_projector_hidden:
                return h, z, attn_weights, h_proj
            return h, z, attn_weights
        if return_projector_hidden:
            return h, z, h_proj
        return h, z
