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


class GLUFeatureMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GLU(dim=1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class InvertedFeatureExpert(nn.Module):
    def __init__(
        self,
        n_patients,
        latent_dim=512,
        n_heads=5,
        encoder_hidden_dim=1024,
        projector_hidden_dim=256,
        projector_out_dim=128,
        mixer_type="attention",
    ):
        super().__init__()

        self.mixer_type = mixer_type
        if mixer_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=n_patients,
                num_heads=n_heads,
                dropout=0.1,
            )
        elif mixer_type == "glu":
            mixer_hidden_dim = min(encoder_hidden_dim, 256)
            self.feature_mixer = GLUFeatureMixer(
                input_dim=n_patients,
                hidden_dim=mixer_hidden_dim,
            )
        else:
            raise ValueError(f"Unsupported mixer_type: {mixer_type}")

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
        attn_weights = None
        if self.mixer_type == "attention":
            x_attn = x.unsqueeze(1)
            attn_out, attn_weights = self.attention(
                x_attn,
                x_attn,
                x_attn,
                need_weights=return_attn,
            )
            x_mixed = attn_out.squeeze(1)
        else:
            x_mixed = self.feature_mixer(x)

        h = self.encoder(x_mixed)
        z = self.projector(h)

        if return_attn:
            return h, z, attn_weights
        return h, z
