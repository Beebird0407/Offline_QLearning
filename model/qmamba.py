"""
Q-Mamba: Mamba-based Q-Learner for Black-Box Optimization Configuration

Architecture:
- Input: Current state s^t concatenated with previous action token (a_{i-1}^t)
- Embedding layer: Binary token -> embedding
- Mamba block: Sequence modeling
- Q-value heads: One per action dimension, outputs M Q-values each
- Autoregressive inference: Q1 -> Q2 -> ... -> QK
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False
    Mamba = None


class RunningNorm(nn.Module):
    """Online state normalization with running statistics."""

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(dim, dtype=torch.float32))
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float dtype for computation
        x = x.float()

        if not self._initialized:
            x_flat = x.reshape(-1, x.shape[-1])
            if x_flat.shape[0] > 1:
                self.running_mean = x_flat.mean(0).float()
                self.running_var = x_flat.var(0).float() + self.eps
            else:
                self.running_mean = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
                self.running_var = torch.ones(x.shape[-1], device=x.device, dtype=torch.float32) + self.eps
            self._initialized = True
            return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)

        if self.training:
            with torch.no_grad():
                x_flat = x.reshape(-1, x.shape[-1])
                if x_flat.shape[0] > 1:
                    mean = x_flat.mean(0).float()
                    var = x_flat.var(0).float() + self.eps
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        return (x - self.running_mean) / (self.running_var.sqrt() + self.eps)


class MambaBlock(nn.Module):
    """Mamba SSM block with residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if _MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            self._use_mamba = True
        else:
            # Fallback to GRU
            self.ssm = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True
            )
            self._use_mamba = False

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_mamba:
            out = self.ssm(x)
            out = self.norm(out + x)
            return out, None
        else:
            out, h_new = self.ssm(x, h)
            out = self.norm(out + x)
            return out, h_new


class QMamba(nn.Module):
    """
    Q-Mamba: Q-learning agent with Mamba backbone.

    Learns to select optimal hyperparameter configurations for evolutionary algorithms.

    Args:
        state_dim: Dimension of state representation (9 default)
        K: Number of action parameters (3 for Alg0: F1, F2, Cr)
        M: Number of bins per action parameter (16 default)
        d_model: Hidden dimension for Mamba
        d_state: Mamba state dimension
        n_layers: Number of Mamba layers
    """

    def __init__(
        self,
        state_dim: int = 9,
        K: int = 3,
        M: int = 16,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.M = M
        self.d_model = d_model
        self.token_dim = 5  # 5-bit encoding for each action bin

        # Total input dimension: state + token
        inp_dim = state_dim + self.token_dim

        # State normalization
        self.state_norm = RunningNorm(state_dim)

        # Token embedding
        self.token_embed = nn.Embedding(M + 1, self.token_dim)  # M bins + 1 start token

        # Input projection
        self.input_proj = nn.Linear(inp_dim, d_model)

        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])

        # Q-value heads (one per action dimension)
        # Each head outputs M Q-values
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

        # Start token (all ones = 11111 = M in decimal)
        self.start_token_idx = M  # Use M as start token index

    def _get_start_token(self, batch_size: int, device) -> torch.Tensor:
        """Get start token tensor."""
        return torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=device)

    def _forward_one_step(
        self,
        s_t: torch.Tensor,
        prev_token: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single step forward pass for one action dimension.

        Args:
            s_t: (B, state_dim) current state
            prev_token: (B,) previous action token indices
            h: hidden state

        Returns:
            q_values: (B, M) Q-values for this dimension
            curr_token: (B,) current action token (argmax)
            h: updated hidden state
        """
        B = s_t.shape[0]

        # Embed previous token
        prev_tok_emb = self.token_embed(prev_token)  # (B, token_dim)

        # Concatenate state and token
        inp = torch.cat([s_t, prev_tok_emb], dim=-1)  # (B, state_dim + token_dim)

        # Project to d_model
        x = self.input_proj(inp).unsqueeze(1)  # (B, 1, d_model)

        # Pass through Mamba layers
        for mamba_layer in self.mamba_layers:
            x, h = mamba_layer(x, h)

        x = x.squeeze(1)  # (B, d_model)

        # Concatenate with original input for Q-head
        q_inp = torch.cat([x, inp], dim=-1)  # (B, d_model + inp_dim)

        # Get Q-values for all K dimensions (share intermediate)
        # For autoregressive, we process each dimension separately
        return q_inp, h

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_all_q: bool = False
    ) -> torch.Tensor:
        """
        Forward pass over a batch of sequences.

        Args:
            states: (B, T, state_dim)
            actions: (B, T, K) - action bin indices
            return_all_q: If True, return all Q-values

        Returns:
            Q_all: (B, T, K, M) Q-values for each action at each timestep
        """
        B, T, _ = states.shape

        # Normalize states
        s_flat = self.state_norm(states.reshape(B * T, -1))
        states = s_flat.reshape(B, T, -1)

        Q_all = torch.zeros(B, T, self.K, self.M, device=states.device)
        h = None

        for t in range(T):
            s_t = states[:, t]  # (B, state_dim)
            prev_token = self._get_start_token(B, states.device)

            for i in range(self.K):
                q_inp, h = self._forward_one_step(s_t, prev_token, h)

                # Compute Q-values for this dimension
                q_i = self.q_heads[i](q_inp)  # (B, M)

                # Min-max normalization
                q_min = q_i.min(-1, keepdim=True).values
                q_max = q_i.max(-1, keepdim=True).values
                q_i = (q_i - q_min) / (q_max - q_min + 1e-8)

                Q_all[:, t, i] = q_i

                # Get action token for next dimension
                if i < self.K - 1:
                    # Use actual action if available (training), else argmax
                    if actions is not None:
                        prev_token = actions[:, t, i]
                    else:
                        prev_token = q_i.argmax(-1)

            # Detach h between timesteps for memory efficiency
            if h is not None and isinstance(h, torch.Tensor):
                h = h.detach()

        return Q_all

    @torch.no_grad()
    def act(
        self,
        s: torch.Tensor,
        deterministic: bool = True,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Greedy action selection at inference.

        Args:
            s: (B, state_dim) or (state_dim,) current state
            deterministic: If True, use argmax; else sample
            h: hidden state

        Returns:
            acts: (B, K) selected action bins
            q_values: (B, K, M) Q-values for all actions
            h: updated hidden state
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]

        s_norm = self.state_norm(s)
        prev_token = self._get_start_token(B, s.device)

        acts = torch.zeros(B, self.K, dtype=torch.long, device=s.device)
        q_values = torch.zeros(B, self.K, self.M, device=s.device)
        h_out = None

        for i in range(self.K):
            q_inp, h_out = self._forward_one_step(s_norm, prev_token, h_out)

            q_i = self.q_heads[i](q_inp)  # (B, M)

            # Min-max normalization
            q_min = q_i.min(-1, keepdim=True).values
            q_max = q_i.max(-1, keepdim=True).values
            q_i_norm = (q_i - q_min) / (q_max - q_min + 1e-8)

            q_values[:, i] = q_i_norm

            if deterministic:
                acts[:, i] = q_i_norm.argmax(-1)
            else:
                probs = F.softmax(q_i_norm, dim=-1)
                acts[:, i] = torch.multinomial(probs, 1).squeeze(-1)

            prev_token = acts[:, i]

        return acts, q_values, h_out

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'state_dim': self.state_dim,
            'K': self.K,
            'M': self.M,
            'd_model': self.d_model,
        }

    @property
    def uses_mamba(self) -> bool:
        """Check if using Mamba SSM (vs GRU fallback)."""
        if len(self.mamba_layers) > 0:
            return getattr(self.mamba_layers[0], '_use_mamba', False)
        return False

    @property
    def num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)