"""
QT: Q-value Regularization Transformer

Transformer with Q-value regularization loss term.
Similar architecture to Decision Transformer but trained with
additional Q-value consistency loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dt import DecisionTransformer


class QTransformer(nn.Module):
    """
    Transformer with Q-value regularization.

    Combines action prediction with Q-value estimation for consistency.
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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K
        self.d_model = d_model

        # Main DT component
        self.dt = DecisionTransformer(
            state_dim=state_dim,
            action_dim=action_dim,
            K=K,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )

        # Separate Q-value heads (one per action dimension)
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(0.01),
                nn.Linear(d_model, action_dim)
            )
            for _ in range(K)
        ])

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
            q_values: (B, T, K, M) Q-values per action dimension
        """
        # Get action logits
        action_logits = self.dt(states, actions, rtg, mask)

        # Get Q-values
        B, T, _, _ = action_logits.shape
        state_emb = self.dt.state_embed(states)  # (B, T, d_model)

        q_values = torch.zeros(B, T, self.K, self.action_dim, device=states.device)
        for i in range(self.K):
            q_values[:, :, i, :] = self.q_heads[i](state_emb)

        return action_logits, q_values

    def q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value consistency loss.

        For each action dimension:
        - Q(s, a) should predict actual return
        - Q(s, a) should be consistent across dimensions
        """
        B, T, K = actions.shape

        # Current Q-values
        _, q_values = self.forward(states, actions, rewards.new_full((B, T), rewards.mean()))

        with torch.no_grad():
            # Next state Q-values (bootstrap)
            _, q_next = self.forward(next_states, torch.zeros_like(actions), rewards.new_full((B, T), rewards.mean()))
            q_next_max = q_next[:, :, 0, :].max(-1).values  # (B, T)

            # Target: reward + gamma * max Q_next
            target = rewards + 0.99 * q_next_max * (1 - dones)

        # Q-loss per dimension
        q_loss = torch.tensor(0.0, device=states.device)
        for i in range(K):
            q_selected = q_values[:, :, i, :].gather(2, actions[:, :, i:i+1].expand_as(q_values[:, :, i, :])).squeeze(-1)
            q_loss = q_loss + F.mse_loss(q_selected, target)

        return q_loss / K

    @torch.no_grad()
    def act(self, state: torch.Tensor, rtg: float, deterministic: bool = True) -> torch.Tensor:
        """Action selection."""
        return self.dt.act(state, rtg, deterministic)