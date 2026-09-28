from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class GeneTransformerVAEV2(nn.Module):
    """
    Permutation-equivariant patch-level gene VAE.

    Key design properties:
    - no patch sequence positional encoding
    - same per-patch input projection + patch-context Transformer family as legacy GeneVAE
    - no LayerNorm applied directly to latent mean
    - log-variance clamped to a configurable stable range
    - no extra ad-hoc latent noise beyond standard VAE reparameterization
    - deterministic validation/inference is available via sample=False
    """

    def __init__(
        self,
        n_genes: int,
        model_dim: int = 512,
        latent_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
    ) -> None:
        super().__init__()

        if model_dim % n_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} must be divisible by n_heads={n_heads}"
            )
        if logvar_min >= logvar_max:
            raise ValueError("logvar_min must be smaller than logvar_max")

        self.n_genes = int(n_genes)
        self.model_dim = int(model_dim)
        self.latent_dim = int(latent_dim)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.gene_input_proj = nn.Linear(self.n_genes, self.model_dim)
        self.input_norm = LayerNorm(self.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.n_heads,
            batch_first=True,
            dropout=self.dropout,
            activation=F.gelu,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        self.latent_head = nn.Linear(self.model_dim, 2 * self.latent_dim)

        # Preserve the legacy decoder family so the major change is the latent encoder
        # semantics, not an unrelated decoder redesign.
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.model_dim),
            LayerNorm(self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim * 2),
            LayerNorm(self.model_dim * 2),
            nn.GELU(),
            nn.Linear(self.model_dim * 2, self.n_genes),
            nn.GELU(),
        )

    @staticmethod
    def _patch_valid(mask: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor:
        if mask is None:
            b, p = shape
            return torch.ones((b, p), dtype=torch.bool)
        return mask.sum(dim=-1) > 0

    def encode(
        self,
        gene_expr: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gene_expr.ndim != 3:
            raise ValueError(f"Expected gene_expr [B,P,G], got {tuple(gene_expr.shape)}")
        if gene_expr.shape[-1] != self.n_genes:
            raise ValueError(
                f"Expected {self.n_genes} genes, got {gene_expr.shape[-1]}"
            )
        if mask is not None and mask.shape != gene_expr.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != gene_expr shape {tuple(gene_expr.shape)}"
            )

        if mask is not None:
            gene_expr = gene_expr * mask.to(dtype=gene_expr.dtype)

        patch_emb = self.input_norm(self.gene_input_proj(gene_expr))

        patch_valid = self._patch_valid(mask, gene_expr.shape[:2]).to(gene_expr.device)
        if not torch.all(patch_valid.any(dim=1)):
            raise ValueError("At least one sample has no valid patches")
        padding_mask = ~patch_valid

        # No positional encoding: self-attention is permutation-equivariant over patches.
        enc_out = self.transformer_encoder(
            patch_emb,
            src_key_padding_mask=padding_mask,
        )

        latent_params = self.latent_head(enc_out)
        z_mean, z_log_var = torch.chunk(latent_params, 2, dim=-1)
        z_log_var = torch.clamp(
            z_log_var,
            min=self.logvar_min,
            max=self.logvar_max,
        )

        # Zero padded patches so they cannot leak arbitrary values into diagnostics.
        valid_f = patch_valid.unsqueeze(-1).to(z_mean.dtype)
        z_mean = z_mean * valid_f
        z_log_var = z_log_var * valid_f

        return z_mean, z_log_var

    def reparameterize(
        self,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
    ) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon

    def decode(
        self,
        latent_z: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if latent_z.ndim != 3 or latent_z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent [B,P,{self.latent_dim}], got {tuple(latent_z.shape)}"
            )
        recon = self.decoder(latent_z)
        if mask is not None:
            recon = recon * mask.to(dtype=recon.dtype)
        return recon

    def forward(
        self,
        gene_expr: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encode(gene_expr, mask)
        latent_z = self.reparameterize(z_mean, z_log_var) if sample else z_mean
        recon = self.decode(latent_z, mask)
        return recon, z_mean, z_log_var
