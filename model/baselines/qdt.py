"""
QDT: Decision Transformer with Q-value Recalibration

Extends Decision Transformer with Q-value based RTG recalibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dt import DecisionTransformer


class QDT(nn.Module):
    """
    Decision Transformer with Q-value recalibration.

    Additional Q-value head predicts optimal return, which is used to
    recalibrate the RTG target during training.
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 16,
        K: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dt = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            K=K,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )

        # Q-value head for RTG recalibration
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass.

        Returns:
            action_logits: (B, T, K, M)
            q_values: (B, T) predicted Q-values
            recalibrated_rtg: (B, T) recalibrated RTG targets
        """
        # Get action logits from DT
        action_logits = self.dt(states, actions, rtg, mask)

        # Extract state representations from DT
        # Use the state token output
        B, T, _, _ = action_logits.shape
        state_emb = self.dt.state_embed(states)  # (B, T, d_model)

        # Q-value prediction
        q_values = self.q_head(state_emb).squeeze(-1)  # (B, T)

        # Recalibrate RTG: original + Q-value adjustment
        recalibrated_rtg = rtg + q_values.detach()

        return action_logits, q_values, recalibrated_rtg

    @torch.no_grad()
    def act(self, state: torch.Tensor, rtg: float, deterministic: bool = True) -> torch.Tensor:
        """Action selection."""
        return self.dt.act(state, rtg, deterministic)