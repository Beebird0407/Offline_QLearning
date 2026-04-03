"""
DeMa: Mamba-based Decision Transformer

Uses Mamba instead of Transformer for sequence modeling.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False
    Mamba = None


class DeMaBlock(nn.Module):
    """Mamba block for DeMa."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self._use_mamba = True
        else:
            self.ssm = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            self._use_mamba = False
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self._use_mamba:
            return self.norm(self.ssm(x) + x)
        else:
            out, _ = self.ssm(x)
            return self.norm(out + x)


class DeMaTransformer(nn.Module):
    """
    Mamba-based Decision Transformer.

    Same interface as DecisionTransformer but uses Mamba layers.
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 16,
        K: int = 3,
        d_model: int = 128,
        n_layers: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.K = K
        self.d_model = d_model

        # Embeddings
        self.rtg_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            DeMaBlock(d_model) for _ in range(n_layers)
        ])

        # Action heads
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
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
    ) -> torch.Tensor:
        """Forward pass similar to DecisionTransformer."""
        B, T, _ = states.shape

        # Embed
        rtg_emb = self.rtg_embed(rtg.unsqueeze(-1))
        state_emb = self.state_embed(states)

        action_embs = [self.action_embed(actions[..., i]) for i in range(self.K)]

        # Concatenate all tokens
        tokens = torch.stack([rtg_emb, state_emb] + action_embs, dim=2)  # (B, T, K+2, d_model)
        x = tokens.reshape(B, T * (self.K + 2), self.d_model)

        # Mamba processing
        seq_len = T * (self.K + 2)
        for i in range(0, seq_len, T):
            chunk = x[:, i:i+T, :]
            for mamba in self.mamba_layers:
                chunk = mamba(chunk)
            x[:, i:i+T, :] = chunk

        # Reshape and get action logits
        x = x.reshape(B, T, self.K + 2, self.d_model)

        logits = torch.zeros(B, T, self.K, self.action_dim, device=x.device)
        for i in range(self.K):
            logits[:, :, i, :] = self.action_heads[i](x[:, :, i + 2, :])

        return logits

    @torch.no_grad()
    def act(self, state: torch.Tensor, rtg: float, deterministic: bool = True) -> torch.Tensor:
        """Action selection."""
        B = state.shape[0]

        state_emb = self.state_embed(state).unsqueeze(1)
        rtg_tensor = torch.full((B, 1), rtg, device=state.device)
        rtg_emb = self.rtg_embed(rtg_tensor.unsqueeze(-1))

        action_tokens = torch.zeros(B, 0, dtype=torch.long, device=state.device)

        for i in range(self.K):
            tokens = [rtg_emb, state_emb]
            for j in range(i):
                tokens.append(self.action_embed(action_tokens[:, j:j+1]))

            for _ in range(self.K - i - 1):
                tokens.append(torch.zeros(B, 1, self.d_model, device=state.device))

            x = torch.cat(tokens, dim=1)
            for mamba in self.mamba_layers:
                x = mamba(x)

            logits = self.action_heads[i](x[:, -1, :])

            if deterministic:
                action = logits.argmax(-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

            action_tokens = torch.cat([action_tokens, action.unsqueeze(-1)], dim=1)

        return action_tokens