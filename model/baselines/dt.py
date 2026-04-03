"""
Decision Transformer (DT)

Transformer-based offline RL that predicts Return-to-Go (RTG).
Architecture: RTG + state -> action prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 16,  # M bins
        K: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim  # M bins
        self.K = K
        self.d_model = d_model

        # Embedding dimensions
        self.rtg_embed = nn.Linear(1, d_model)
        self.state_embed = nn.Linear(state_dim, d_model)
        self.action_embed = nn.Embedding(action_dim, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 3 * K + 1, d_model))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads (one per action dimension)
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
        states: torch.Tensor,      # (B, T, state_dim)
        actions: torch.Tensor,      # (B, T, K) - action tokens
        rtg: torch.Tensor,          # (B, T) - return-to-go targets
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, _ = states.shape

        # Embed inputs
        rtg_emb = self.rtg_embed(rtg.unsqueeze(-1))  # (B, T, d_model)
        state_emb = self.state_embed(states)  # (B, T, d_model)

        # Action embeddings for each dimension
        action_embs = []
        for i in range(self.K):
            a_emb = self.action_embed(actions[..., i])  # (B, T, d_model)
            action_embs.append(a_emb)

        # Interleave: [rtg, state, a1, a2, ..., aK]
        tokens = [rtg_emb, state_emb] + action_embs
        x = torch.stack(tokens, dim=1)  # (B, 3+K, T, d_model)
        x = x.permute(0, 2, 1, 3).reshape(B, T * (self.K + 2), self.d_model)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)

        # Reshape: (B, T, K+2, d_model) -> (B, T, K, d_model)
        x = x.reshape(B, T, self.K + 2, self.d_model)
        x = x[:, :, 2:, :]  # Skip rtg and state

        # Action logits for each dimension
        logits = torch.zeros(B, T, self.K, self.action_dim, device=x.device)
        for i in range(self.K):
            logits[:, :, i, :] = self.action_heads[i](x[:, :, i, :])

        return logits

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        rtg: float,
        deterministic: bool = True
    ) -> torch.Tensor:
        B = state.shape[0]
        T = 1

        # Embed state
        state_emb = self.state_embed(state).unsqueeze(1)  # (B, 1, d_model)
        rtg_tensor = torch.full((B, 1), rtg, device=state.device)
        rtg_emb = self.rtg_embed(rtg_tensor.unsqueeze(-1))  # (B, 1, d_model)

        # Start with empty actions
        action_tokens = torch.zeros(B, 0, dtype=torch.long, device=state.device)

        for i in range(self.K):
            # Build input
            tokens = [rtg_emb, state_emb]
            for j in range(i):
                a_emb = self.action_embed(action_tokens[:, j:j+1])  # (B, 1, d_model)
                tokens.append(a_emb)

            # Pad with zeros for remaining action positions
            for _ in range(self.K - i - 1):
                tokens.append(torch.zeros(B, 1, self.d_model, device=state.device))

            x = torch.cat(tokens, dim=1)  # (B, i+2, d_model)
            x = x + self.pos_embed[:, :x.shape[1], :]

            # Transformer
            x_out = self.transformer(x)

            # Get logits for this action dimension
            logits = self.action_heads[i](x_out[:, -1, :])  # (B, M)

            if deterministic:
                action = logits.argmax(-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

            action_tokens = torch.cat([action_tokens, action.unsqueeze(-1)], dim=1)

        return action_tokens