"""Per-patch MLP baseline for gene-expression prediction."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SimpleMLP(nn.Module):
    """Three-layer MLP that predicts a gene-expression vector for each patch independently."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_features_1: int,
        hidden_features_2: int,
        output_features: int,
    ) -> None:
        """Build the MLP.

        Args:
            embedding_dim: Dimension of each input patch embedding.
            hidden_features_1: Width of the first hidden layer.
            hidden_features_2: Width of the second hidden layer.
            output_features: Number of predicted genes.
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_features_1)
        self.ln1 = nn.LayerNorm(hidden_features_1)
        self.fc2 = nn.Linear(hidden_features_1, hidden_features_2)
        self.ln2 = nn.LayerNorm(hidden_features_2)
        self.fc3 = nn.Linear(hidden_features_2, output_features)
        self.loss_fn = nn.MSELoss()

    def _calculate_gene_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """MSE over non-NaN targets; a graph-connected zero if no target is valid."""
        mask = ~torch.isnan(targets)
        if not mask.any():
            return predictions.sum() * 0.0
        return F.mse_loss(predictions[mask], targets[mask])

    def _get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expression per patch.

        Args:
            x: Patch embeddings, shape [1, n_patches, embedding_dim].

        Returns:
            Predictions of shape [n_patches, output_features].
        """
        x = x.squeeze(0)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the masked MSE loss for one slide.

        Args:
            x: Patch embeddings, shape [1, n_patches, embedding_dim].
            y: Measured expression, shape [1, n_patches, output_features] (NaN = unmeasured).

        Returns:
            The loss tensor and a logging dict ``{"gene_loss": float}``.
        """
        gene_loss = self._calculate_gene_loss(self._get_predictions(x), y.squeeze(0))
        return gene_loss, {"gene_loss": gene_loss.item()}
