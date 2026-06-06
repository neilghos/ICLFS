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

    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h_proj = h1 + h2
        return self.l3(h_proj)


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

        # Attention mixes feature tokens before encoding.
        self.attention = nn.MultiheadAttention(embed_dim=n_patients, num_heads=n_heads)
        
        # Encode each feature profile into the latent space.
        self.encoder = nn.Sequential(
            nn.Linear(n_patients, encoder_hidden_dim),
            nn.BatchNorm1d(encoder_hidden_dim, momentum=0.01, eps=1e-5),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_dim, latent_dim)
        )
        
        self.projector = ResidualProjector(
            latent_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )

    def forward(self, x, return_attn=False):
        x_attn = x.unsqueeze(1) 
        attn_out, attn_weights = self.attention(x_attn, x_attn, x_attn)
        x_mixed = attn_out.squeeze(1)

        h = self.encoder(x_mixed)
        z = self.projector(h)

        if return_attn:
            return h, z, attn_weights
        return h, z
