import math
import numpy as np
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


class MLPFeatureMixer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class InvertedFeatureExpert(nn.Module):
    def __init__(
        self,
        n_patients,
        latent_dim=512,
        encoder_hidden_dim=1024,
        projector_hidden_dim=256,
        projector_out_dim=128,
    ):
        super().__init__()

        mixer_hidden_dim = min(encoder_hidden_dim, 256)
        self.feature_mixer = MLPFeatureMixer(
            input_dim=n_patients,
            hidden_dim=mixer_hidden_dim,
        )

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
        x_mixed = self.feature_mixer(x)

        h = self.encoder(x_mixed)
        z = self.projector(h)

        if return_attn:
            return h, z, None
        return h, z


class FeatureChunkExpert(nn.Module):
    """
    Model for Class-Conditioned Supervised Chunk Learning.
    Inputs:
        x: Feature chunk vector [B, chunk_len] (a sub-profile of a feature over L samples).
    Outputs:
        h: Latent representation of the chunk [B, latent_dim]
        z: Projected feature representation [B, projector_out_dim]
        logits: Downstream target logits over C classes [B, num_classes]
    """

    def __init__(
        self,
        chunk_len: int,
        num_classes: int = 2,
        latent_dim: int = 512,
        encoder_hidden_dim: int = 512,
        projector_hidden_dim: int = 256,
        projector_out_dim: int = 128,
    ):
        super().__init__()
        self.chunk_len = chunk_len
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(chunk_len, encoder_hidden_dim),
            nn.BatchNorm1d(encoder_hidden_dim, momentum=0.01, eps=1e-5),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(encoder_hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01, eps=1e-5),
            nn.LeakyReLU(0.2),
        )

        self.projector = ResidualProjector(
            latent_dim,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
        )

        # Classifier head mapping chunk representation -> Downstream target class logits
        self.classifier = nn.Sequential(
            nn.Linear(projector_out_dim, projector_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projector_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.projector(h)
        logits = self.classifier(z)
        return h, z, logits


class NonLinearFeatureValueEncoder(nn.Module):
    """
    TabM-style Non-Linear Feature Value Encoder.
    Passes scalar feature value x_{i, j} through a non-linear feature activation layer
    before modulating the chunk-learned feature embedding W_j:
    u_{i, j} = ReLU(x_{i, j} * W_{j, 1} + b_{j, 1}) * W_{j, 2} * W_j
    """

    def __init__(self, num_features: int, out_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.num_features = num_features
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        # Per-feature non-linear transformation parameters
        self.w1 = nn.Parameter(torch.randn(num_features, hidden_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_features, hidden_dim))
        self.act = nn.ReLU()
        self.w2 = nn.Parameter(torch.randn(num_features, hidden_dim, out_dim) * 0.02)

    def forward(self, x: torch.Tensor, feature_embeddings: torch.Tensor) -> torch.Tensor:
        """
        x: [B, K] - scalar values for selected K features
        feature_embeddings: [K, d_out] - chunk-learned feature vectors W_j
        Returns:
            z_sample: [B, d_out] - non-linearly encoded sample representation
        """
        B, K = x.shape
        x_unsq = x.unsqueeze(-1)  # [B, K, 1]

        # First non-linear expansion: [B, K, hidden_dim]
        h1 = self.act(x_unsq * self.w1[:K].unsqueeze(0) + self.b1[:K].unsqueeze(0))

        # Second linear mapping to out_dim: [B, K, out_dim]
        u = torch.einsum("bkh, kho -> bko", h1, self.w2[:K])

        # Combine with chunk-learned feature embeddings W_j: [B, K, out_dim]
        u_combined = u * feature_embeddings.unsqueeze(0)

        # Non-linear sample aggregation: [B, out_dim]
        scale = math.sqrt(max(1, K))
        return u_combined.sum(dim=1) / scale


class BatchEnsembleMultiHeadProjector(nn.Module):
    """
    BatchEnsemble Multi-Head Sample Projector with Non-Linear Value Encoding.
    Generates M ensemble head sample representations using rank-1 input/output scaling vectors:
    Z^{(m)} = ( (LeakyReLU(X_sel * W_{j,1}) * R_m) @ (W_sel * S_m) ) / sqrt(K)
    Returns concatenated ensemble representation: [N, M * d_out]
    """

    def __init__(self, num_features: int, d_out: int = 256, num_heads: int = 16):
        super().__init__()
        self.num_features = num_features
        self.d_out = d_out
        self.num_heads = num_heads

        # Non-linear feature value encoding weights
        self.val_w1 = nn.Parameter(torch.randn(num_features, 16) * 0.02 + 1.0)
        self.val_act = nn.LeakyReLU(0.2)
        self.val_w2 = nn.Parameter(torch.randn(num_features, 16) * 0.02 + 1.0)

        # Rank-1 input scaling vectors R: [M, num_features]
        r_init = torch.bernoulli(torch.full((num_heads, num_features), 0.5)) * 2.0 - 1.0
        self.r_weights = nn.Parameter(r_init * 0.1 + 1.0)

        # Rank-1 output scaling vectors S: [M, d_out]
        s_init = torch.bernoulli(torch.full((num_heads, d_out), 0.5)) * 2.0 - 1.0
        self.s_weights = nn.Parameter(s_init * 0.1 + 1.0)

    def forward(self, x: torch.Tensor, feature_embeddings: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D] - raw tabular samples
        feature_embeddings: [D, d_out] - ICL feature embeddings
        selected_idx: [K] - indices of selected features
        Returns: [N, num_heads * d_out]
        """
        selected_x = x[:, selected_idx]  # [N, K]
        selected_w = feature_embeddings[selected_idx]  # [K, d_out]
        K = selected_idx.shape[0]
        scale = math.sqrt(max(1, K))

        # Apply non-linear feature value transformation per selected feature
        w1_sel = self.val_w1[selected_idx]  # [K, 16]
        w2_sel = self.val_w2[selected_idx]  # [K, 16]
        x_unsq = selected_x.unsqueeze(-1)  # [N, K, 1]
        h1 = self.val_act(x_unsq * w1_sel.unsqueeze(0))  # [N, K, 16]
        x_nl = (h1 * w2_sel.unsqueeze(0)).mean(dim=-1)  # [N, K]

        head_reps = []
        for m in range(self.num_heads):
            r_m = self.r_weights[m, selected_idx]  # [K]
            s_m = self.s_weights[m]  # [d_out]

            x_scaled = x_nl * r_m.unsqueeze(0)  # [N, K]
            w_scaled = selected_w * s_m.unsqueeze(0)  # [K, d_out]

            z_m = (x_scaled @ w_scaled) / scale  # [N, d_out]
            head_reps.append(z_m)

        return torch.cat(head_reps, dim=-1)  # [N, M * d_out]


class LearnableFeatureValueTokenizer(nn.Module):
    """
    Learnable Feature Value Tokenizer for Tabular Data.
    Maps scalar feature values x_{i, j} to d_token dimensional feature vectors:
    e(x_{i, j}) = LeakyReLU(x_{i, j} * W_{j, 1} + b_{j, 1}) * W_{j, 2}
    Preserves clean 0/1 binary feature identities while learning non-linear continuous curves.
    """

    def __init__(self, num_features: int, d_token: int = 16):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        self.w1 = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_features, d_token))
        self.act = nn.LeakyReLU(0.2)
        self.w2 = nn.Parameter(torch.randn(num_features, d_token, d_token) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] - batch of scalar tabular features
        Returns: [B, D, d_token]
        """
        batch_size, num_features = x.shape
        x_unsq = x.unsqueeze(-1)  # [B, D, 1]

        h1 = self.act(x_unsq * self.w1[:num_features].unsqueeze(0) + self.b1[:num_features].unsqueeze(0))
        # Batch transformation across D features: [B, D, d_token]
        tokens = torch.einsum("bdh, dht -> bdt", h1, self.w2[:num_features])
        return tokens


class TabularFeatureTokenizer(nn.Module):
    """
    Maps each feature scalar value x_{i, j} for sample i to a d-dimensional feature token:
    e_{i, j} = x_{i, j} * W_j + b_j + Feature_ID_Embedding(j)
    """

    def __init__(self, num_features: int, d_token: int = 64):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        # Per-feature weight W_j and bias b_j: shape [num_features, d_token]
        self.weight = nn.Parameter(torch.randn(num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, d_token))

        # Feature ID embedding to distinguish different features
        self.feature_id_embed = nn.Embedding(num_features, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] - mini-batch of tabular samples
        Returns: [B, D, d_token]
        """
        batch_size, num_features = x.shape
        x_unsq = x.unsqueeze(-1)  # [B, D, 1]

        # Element-wise scaling per feature: [B, D, d_token]
        value_embed = x_unsq * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        feature_ids = torch.arange(num_features, device=x.device)
        id_embed = self.feature_id_embed(feature_ids).unsqueeze(0)  # [1, D, d_token]

        return value_embed + id_embed


class CrossFeatureTransformerBlock(nn.Module):
    def __init__(self, d_token: int = 64, n_heads: int = 4, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, d_token]
        """
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FeatureWiseSubTabModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        projector_out_dim: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        self.tokenizer = TabularFeatureTokenizer(num_features, d_token=d_token)
        self.layers = nn.ModuleList([
            CrossFeatureTransformerBlock(d_token=d_token, n_heads=n_heads, d_ff=2 * d_token)
            for _ in range(n_layers)
        ])

        # Projector for contrastive loss over feature tokens: maps [B, D, d_token] -> [B, D, projector_out_dim]
        self.feature_projector = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.ReLU(),
            nn.Linear(d_token, projector_out_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [B, D]
        Returns:
            h_sample: [B, D * d_token] (contextualized sample representation)
            z_tokens: [B, D, projector_out_dim] (projected feature tokens for InfoNCE)
        """
        tokens = self.tokenizer(x)  # [B, D, d_token]
        for layer in self.layers:
            tokens = layer(tokens)  # [B, D, d_token]

        z_tokens = self.feature_projector(tokens)  # [B, D, projector_out_dim]

        # Mean + Max pooling over feature tokens to create a compact, expressive sample representation [B, 2 * d_token]
        mean_pool = tokens.mean(dim=1)  # [B, d_token]
        max_pool = tokens.max(dim=1).values  # [B, d_token]
        h_sample = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2 * d_token]
        return h_sample, z_tokens


class PeriodicFeatureValueTokenizer(nn.Module):
    """
    Periodic / Piecewise Linear Frequency Tokenizer for Tabular Data.
    Expands scalar feature values x_{i, j} with periodic sinusoids:
        P_{i, j} = [x_{i, j}, sin(x_{i, j} * W_f + b_f), cos(x_{i, j} * W_f + b_f)]  in R^{1 + 2K}
    Passes P_{i, j} through per-feature transformation into d_token dimensional feature vectors:
        e_{i, j} = LeakyReLU(P_{i, j} @ W_1 + b_1) @ W_2 + FeatureID(j)
    """

    def __init__(self, num_features: int, d_token: int = 128, n_frequencies: int = 16):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token
        self.n_frequencies = n_frequencies

        # Periodic frequency parameters W_f and b_f: [num_features, n_frequencies]
        self.w_freq = nn.Parameter(torch.randn(num_features, n_frequencies) * 0.5)
        self.b_freq = nn.Parameter(torch.zeros(num_features, n_frequencies))

        # Input dimension after sinusoidal expansion: 1 (raw) + 2 * n_frequencies
        in_dim = 1 + 2 * n_frequencies

        self.w1 = nn.Parameter(torch.randn(num_features, in_dim, d_token) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_features, d_token))
        self.act = nn.LeakyReLU(0.2)
        self.w2 = nn.Parameter(torch.randn(num_features, d_token, d_token) * 0.02)
        self.feature_id_embed = nn.Embedding(num_features, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D] - batch of scalar tabular features
        Returns: [B, D, d_token]
        """
        batch_size, num_features = x.shape
        x_unsq = x.unsqueeze(-1)  # [B, D, 1]

        # Sinusoidal periodic expansion across D features: [B, D, n_frequencies]
        freq_arg = x_unsq * self.w_freq[:num_features].unsqueeze(0) + self.b_freq[:num_features].unsqueeze(0)
        sin_feat = torch.sin(freq_arg)  # [B, D, n_frequencies]
        cos_feat = torch.cos(freq_arg)  # [B, D, n_frequencies]

        # Concatenate raw value + sin + cos: [B, D, 1 + 2 * n_frequencies]
        p_feat = torch.cat([x_unsq, sin_feat, cos_feat], dim=-1)

        # Per-feature linear projection: [B, D, d_token]
        h1 = self.act(torch.einsum("bdi, dit -> bdt", p_feat, self.w1[:num_features]) + self.b1[:num_features].unsqueeze(0))
        tokens = torch.einsum("bdt, dth -> bdh", h1, self.w2[:num_features])

        # Add feature ID positional embedding
        feature_ids = torch.arange(num_features, device=x.device)
        id_embed = self.feature_id_embed(feature_ids).unsqueeze(0)  # [1, D, d_token]

        return tokens + id_embed


class PLEPeriodicFeatureValueTokenizer(nn.Module):
    """
    Piecewise Linear Encoding (PLE) + Fourier Frequency Tokenizer for Tabular Data.
    Combines:
        1. Quantile Bin Encodings (Piecewise Linear Interpolation across N_bins).
        2. Sinusoidal Periodic Fourier Frequency Expansions [sin(W x + b), cos(W x + b)].
    Yields sharp tree-like step decisions + smooth non-linear wave modeling.
    Complexity: O(1) vectorized GPU tensor slicing. Ultra-fast!
    """

    def __init__(self, num_features: int, in_channels: int = 1, d_token: int = 128, n_frequencies: int = 16, n_bins: int = 32):
        super().__init__()
        self.num_features = num_features
        self.in_channels = in_channels
        self.d_token = d_token
        self.n_frequencies = n_frequencies
        self.n_bins = n_bins

        # Periodic frequency parameters W_f and b_f: [num_features, in_channels, n_frequencies]
        self.w_freq = nn.Parameter(torch.randn(num_features, in_channels, n_frequencies) * 0.5)
        self.b_freq = nn.Parameter(torch.zeros(num_features, n_frequencies))

        # PLE linear bin transformation weights
        self.w_ple = nn.Parameter(torch.randn(num_features, n_bins, d_token) * 0.02)

        # Periodic input dimension: in_channels (raw) + 2 * n_frequencies
        in_dim_periodic = in_channels + 2 * n_frequencies
        self.w_periodic = nn.Parameter(torch.randn(num_features, in_dim_periodic, d_token) * 0.02)
        self.b_periodic = nn.Parameter(torch.zeros(num_features, d_token))

        self.act = nn.LeakyReLU(0.2)
        self.w_out = nn.Parameter(torch.randn(num_features, d_token, d_token) * 0.02)
        self.feature_id_embed = nn.Embedding(num_features, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D, in_channels] - batch of tabular features with in_channels
        Returns: [B, D, d_token]
        """
        batch_size, num_features, in_channels = x.shape

        # 1. Periodic Fourier Expansion: [B, D, in_channels + 2 * n_frequencies]
        freq_arg = torch.einsum("bdc, dcf -> bdf", x, self.w_freq[:num_features]) + self.b_freq[:num_features].unsqueeze(0)
        sin_feat = torch.sin(freq_arg)
        cos_feat = torch.cos(freq_arg)
        p_feat = torch.cat([x, sin_feat, cos_feat], dim=-1)

        # Periodic Projection: [B, D, d_token]
        h_periodic = self.act(torch.einsum("bdi, dit -> bdt", p_feat, self.w_periodic[:num_features]) + self.b_periodic[:num_features].unsqueeze(0))

        # 2. Piecewise Linear Encoding (PLE) Bin Activation: [B, D, n_bins]
        # Run PLE on the standard-scaled channel (the last channel)
        ple_x = x[:, :, -1]
        bin_edges = torch.linspace(-3.0, 3.0, self.n_bins, device=x.device).unsqueeze(0).unsqueeze(0)
        ple_ramps = torch.clamp((ple_x.unsqueeze(-1) - bin_edges) * 2.0 + 0.5, 0.0, 1.0)
        h_ple = torch.einsum("bdi, dit -> bdt", ple_ramps, self.w_ple[:num_features])

        # Combined Representation: [B, D, d_token]
        h_combined = h_periodic + h_ple
        tokens = torch.einsum("bdt, dth -> bdh", h_combined, self.w_out[:num_features])

        # Feature ID positional embedding
        feature_ids = torch.arange(num_features, device=x.device)
        id_embed = self.feature_id_embed(feature_ids).unsqueeze(0)

        return tokens + id_embed


class SupervisedICLModel(nn.Module):
    """
    End-to-End Supervised Inverted Contextualized Learning with PLE (SupICL-PLE).
    Features:
        1. Piecewise Linear Encoding (PLE) + Fourier Frequency Tokenizer.
        2. Cross-feature interaction attention blocks over feature tokens.
        3. Mean + Max token pooling into compact sample representation [B, 2 * d_token].
        4. Heavy residual MLP classification / regression head.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int = 2,
        in_channels: int = 1,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.d_token = d_token

        self.tokenizer = PLEPeriodicFeatureValueTokenizer(num_features, in_channels=in_channels, d_token=d_token, n_frequencies=16, n_bins=32)
        self.layers = nn.ModuleList([
            CrossFeatureTransformerBlock(d_token=d_token, n_heads=n_heads, d_ff=2 * d_token, dropout=dropout)
            for _ in range(n_layers)
        ])

        sample_dim = 2 * d_token  # Mean pool + Max pool

        self.head = nn.Sequential(
            nn.Linear(sample_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

        # Feature Reconstruction Decoder Head: per-feature decoder [B, D, d_token] -> [B, D]
        self.recon_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Linear(d_token // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: [B, D] - batch of raw tabular samples
        mask: optional binary mask [B, D] where 1 indicates masked feature to reconstruct
        Returns:
            logits: [B, num_outputs]
            recon_x: [B, D] (reconstructed raw feature values, if training)
        """
        if mask is not None:
            x_input = x * (1.0 - mask.unsqueeze(-1))
        else:
            x_input = x

        tokens = self.tokenizer(x_input)  # [B, D, d_token]
        for layer in self.layers:
            tokens = layer(tokens)  # [B, D, d_token]

        mean_pool = tokens.mean(dim=1)  # [B, d_token]
        max_pool = tokens.max(dim=1).values  # [B, d_token]
        z_sample = torch.cat([mean_pool, max_pool], dim=1)  # [B, 2 * d_token]

        logits = self.head(z_sample)  # [B, num_outputs]

        if self.training:
            recon_x = self.recon_head(tokens).squeeze(-1)  # [B, D]
            return logits, recon_x
        else:
            return logits, None


class StarGraphValueEmbedding(nn.Module):
    """
    Non-Linear Feature Value Embedding Layer for FeatureGraphTab.
    Computes 3 parallel sub-representations per feature node:
      1. Continuous multi-channel linear value projection (Quantile Norm + Quantile Uni + Raw)
      2. Periodic Fourier frequency spectrum
      3. Piecewise Linear Encoding (PLE) bin ramps
    Fuses them into a single node tensor e_j^{(0)} in R^{d_token}.
    """

    def __init__(
        self,
        num_features: int,
        in_channels: int = 3,
        d_token: int = 128,
        n_frequencies: int = 16,
        n_bins: int = 32,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_token = d_token

        # 1. Multi-channel linear value projection
        self.w_linear = nn.Linear(in_channels, d_token)

        # 2. Periodic Fourier frequency spectrum
        frequencies = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        self.w_fourier = nn.Linear(2 * n_frequencies, d_token)

        # 3. Piecewise Linear Encoding (PLE) bin ramps
        bin_edges = torch.linspace(-3.0, 3.0, n_bins + 1, dtype=torch.float32)
        self.register_buffer("bin_edges", bin_edges)
        self.w_ple = nn.Linear(n_bins, d_token)

        # Feature ID embedding
        self.feature_id_embed = nn.Embedding(num_features, d_token)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # x: [B, D, in_channels]
        batch_size, num_features, in_channels = x.shape

        # 1. Continuous Linear Value Representation
        v_linear = self.w_linear(x)  # [B, D, d_token]

        # Use primary standardized raw value channel (channel -1)
        raw_val = x[:, :, -1]  # [B, D]

        # 2. Periodic Fourier Frequency Spectrum
        freq_args = raw_val.unsqueeze(-1) * self.frequencies.unsqueeze(0).unsqueeze(0)
        fourier_feats = torch.cat([torch.cos(2.0 * math.pi * freq_args), torch.sin(2.0 * math.pi * freq_args)], dim=-1)
        v_fourier = self.w_fourier(fourier_feats)  # [B, D, d_token]

        # 3. Piecewise Linear Encoding (PLE) Ramp
        val_clamped = torch.clamp(raw_val, min=float(self.bin_edges[0]), max=float(self.bin_edges[-1]))
        val_expanded = val_clamped.unsqueeze(-1)  # [B, D, 1]
        b_left = self.bin_edges[:-1].unsqueeze(0).unsqueeze(0)
        b_right = self.bin_edges[1:].unsqueeze(0).unsqueeze(0)
        b_width = b_right - b_left
        ple_ramps = torch.clamp((val_expanded - b_left) / b_width, min=0.0, max=1.0)
        v_ple = self.w_ple(ple_ramps)  # [B, D, d_token]

        # Feature ID embedding
        feature_ids = torch.arange(num_features, device=x.device)
        id_embed = self.feature_id_embed(feature_ids).unsqueeze(0)  # [1, D, d_token]

        # Fused Feature Node Embedding e_j^{(0)}
        e_features = v_linear + v_fourier + v_ple + id_embed  # [B, D, d_token]

        return e_features, None


class MultiHeadLightGCNPropagation(nn.Module):
    """
    Multi-Head LightGCN Message Propagation over H=8 Parallel Subspace Feature Graphs.
    Projects node embeddings into H=8 heads, computes sample-adaptive & correlation-attributed 
    adjacency matrices A^{(h)} for each head, and performs 100% linear LightGCN graph convolutions.
    """

    def __init__(self, d_token: int = 128, n_heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.n_layers = n_layers
        self.scale = 1.0 / (self.d_head ** 0.5)

        self.w_q = nn.Linear(d_token, d_token)
        self.w_k = nn.Linear(d_token, d_token)
        self.w_v = nn.Linear(d_token, d_token)
        self.w_out = nn.Linear(d_token, d_token)

        self.layer_weights = [1.0 / (n_layers + 1)] * (n_layers + 1)
        self.register_buffer("A_norm", None)

    def set_adj_matrix(self, A_norm: torch.Tensor):
        self.register_buffer("A_norm", A_norm)

    def forward(self, e_features: torch.Tensor) -> torch.Tensor:
        # e_features: [B, D, d_token]
        batch_size, num_features, _ = e_features.shape

        # Multi-Head Query, Key, Value projections: [B, H, D, d_head]
        q = self.w_q(e_features).view(batch_size, num_features, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(e_features).view(batch_size, num_features, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(e_features).view(batch_size, num_features, self.n_heads, self.d_head).transpose(1, 2)

        # Dynamic sample-adaptive similarity matrix for each head: [B, H, D, D]
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, D, D]

        # Blend with static correlation matrix A_norm if available
        if self.A_norm is not None:
            A_static = self.A_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, D, D]
            sim = sim + A_static

        # Softmax normalize adjacency matrix per head
        A_head = torch.softmax(sim, dim=-1)  # [B, H, D, D]

        head_states = [v]
        curr_v = v

        for _ in range(self.n_layers):
            # Pure linear LightGCN graph convolution across all H heads simultaneously:
            # A_head @ curr_v -> [B, H, D, d_head]
            next_v = torch.matmul(A_head, curr_v)
            curr_v = next_v
            head_states.append(curr_v)

        # Multi-layer LightGCN linear combination
        v_final = torch.zeros_like(v)
        for layer_idx, state in enumerate(head_states):
            v_final = v_final + self.layer_weights[layer_idx] * state

        # Reshape back to [B, D, d_token] and project
        v_concat = v_final.transpose(1, 2).contiguous().view(batch_size, num_features, self.d_token)
        return self.w_out(v_concat)


class StarGraphTabModel(nn.Module):
    """
    MultiHeadGraphTab Architecture.
    Pipeline:
      1. Non-Linear Value Embedding Layer (Raw + Fourier + PLE)
      2. Multi-Head LightGCN Message Propagation (H=8 parallel graph heads)
      3. Per-Feature Token Residual FFN Layer
      4. Learned Attentive Readout Pool & Non-Linear Prediction Head
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int = 1,
        in_channels: int = 3,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = StarGraphValueEmbedding(
            num_features=num_features,
            in_channels=in_channels,
            d_token=d_token,
        )
        self.gnn = MultiHeadLightGCNPropagation(d_token=d_token, n_heads=n_heads, n_layers=n_layers)

        # Per-Feature Token Residual FFN Layer
        self.feature_ffn = nn.Sequential(
            nn.Linear(d_token, 2 * d_token),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_token, d_token),
        )

        # Learned Attentive Feature Readout Pool
        self.attn_pool_weights = nn.Linear(d_token, 1)

        # Non-Linear Prediction Head
        sample_dim = 2 * d_token
        self.head = nn.Sequential(
            nn.Linear(sample_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_outputs),
        )

        # Feature Reconstruction Decoder Head: per-feature decoder [B, D, d_token] -> [B, D]
        self.recon_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Linear(d_token // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None:
            x_input = x.clone()
            x_input[:, :, -1] = x[:, :, -1] * (1.0 - mask)
        else:
            x_input = x

        e_features, _ = self.embedding(x_input)
        e_features_final = self.gnn(e_features)  # [B, D, d_token]

        # 1. Per-Feature Token Residual FFN Pass
        e_features_processed = e_features_final + self.feature_ffn(e_features_final)  # [B, D, d_token]

        # 2. Learned Attentive Readout Pool
        attn_scores = torch.softmax(self.attn_pool_weights(e_features_processed), dim=1)  # [B, D, 1]
        attn_pool = (attn_scores * e_features_processed).sum(dim=1)  # [B, d_token]
        max_pool = e_features_processed.max(dim=1).values  # [B, d_token]
        z_sample = torch.cat([attn_pool, max_pool], dim=1)  # [B, 2 * d_token]

        logits = self.head(z_sample)

        if self.training:
            recon_x = self.recon_head(e_features_processed).squeeze(-1)  # [B, D]
            return logits, recon_x
        else:
            return logits, None
