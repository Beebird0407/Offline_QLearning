"""
Q-Transformer: Transformer-based Q-Learning

Similar to Q-Mamba but uses Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class QTransformerModel(nn.Module):
    """
    Transformer-based Q-Learner.

    Same as QMamba but with Transformer instead of Mamba.
    """

    def __init__(
        self,
        state_dim: int = 9,
        K: int = 3,
        M: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.M = M
        self.d_model = d_model
        self.token_dim = 5

        # Token embedding
        self.token_embed = nn.Embedding(M + 1, self.token_dim)

        # Input projection
        inp_dim = state_dim + self.token_dim
        self.input_proj = nn.Linear(inp_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Q-value heads
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + inp_dim, d_model),
                nn.LeakyReLU(0.01),
                nn.Linear(d_model, d_model // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(d_model // 2, M)
            )
            for _ in range(K)
        ])

        # Start token
        self.start_token_idx = M

    def _get_start_token(self, batch_size: int, device) -> torch.Tensor:
        return torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=device)

    def _forward_one_step(self, s_t, prev_token, h=None):
        B = s_t.shape[0]
        prev_tok_emb = self.token_embed(prev_token)
        inp = torch.cat([s_t, prev_tok_emb], dim=-1)
        x = self.input_proj(inp).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return x, inp

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_all_q: bool = False
    ) -> torch.Tensor:
        B, T, _ = states.shape
        Q_all = torch.zeros(B, T, self.K, self.M, device=states.device)

        for t in range(T):
            s_t = states[:, t]
            prev_token = self._get_start_token(B, states.device)

            for i in range(self.K):
                x, inp = self._forward_one_step(s_t, prev_token)
                q_i = self.q_heads[i](torch.cat([x, inp], dim=-1))

                q_min = q_i.min(-1, keepdim=True).values
                q_max = q_i.max(-1, keepdim=True).values
                q_i = (q_i - q_min) / (q_max - q_min + 1e-8)

                Q_all[:, t, i] = q_i

                if i < self.K - 1:
                    prev_token = actions[:, t, i] if actions is not None else q_i.argmax(-1)

        return Q_all

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = True, h=None) -> tuple:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]

        acts = torch.zeros(B, self.K, dtype=torch.long, device=s.device)
        prev_token = self._get_start_token(B, s.device)

        for i in range(self.K):
            x, inp = self._forward_one_step(s, prev_token)
            q_i = self.q_heads[i](torch.cat([x, inp], dim=-1))

            q_min = q_i.min(-1, keepdim=True).values
            q_max = q_i.max(-1, keepdim=True).values
            q_i_norm = (q_i - q_min) / (q_max - q_min + 1e-8)

            if deterministic:
                acts[:, i] = q_i_norm.argmax(-1)
            else:
                probs = F.softmax(q_i_norm, dim=-1)
                acts[:, i] = torch.multinomial(probs, 1).squeeze(-1)

            prev_token = acts[:, i]

        return acts, None, None