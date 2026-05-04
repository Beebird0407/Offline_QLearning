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

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, force_gru: bool = False):
        super().__init__()
        self._force_gru = force_gru

        if _MAMBA_AVAILABLE and not force_gru:
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
        if self._use_mamba and not self._force_gru:
            out = self.ssm(x)
            out = self.norm(out + x)
            return out, None
        else:
            out, h_new = self.ssm(x, h)
            out = self.norm(out + x)
            return out, h_new


class QMamba(nn.Module):
    def __init__(
        self,
        state_dim: int = 9,
        K: int = 3,
        M: int = 16,
        d_model: int = 14,
        d_state: int = 32,
        n_layers: int = 1,
        num_hidden_mlp: int = 32,
        force_cpu: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.K = K
        self.M = M
        self.d_model = d_model
        self.d_state = d_state
        self.force_cpu = force_cpu
        self.token_dim = 5
        self.num_hidden_mlp = num_hidden_mlp

        inp_dim = state_dim + self.token_dim
        self.inp_dim = inp_dim
        self.state_norm = RunningNorm(state_dim)
        self.token_embed = nn.Embedding(M + 1, self.token_dim)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, force_gru=force_cpu) for _ in range(n_layers)
        ])
        # Q-head: match Q-Mamba-main DAC_block structure
        # Linear(14, 32) → LeakyReLU → Linear(32, 16)
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + inp_dim, num_hidden_mlp),
                nn.LeakyReLU(0.01),
                nn.Linear(num_hidden_mlp, M)
            )
            for _ in range(K)
        ])
        self.start_token_idx = M

    def _get_start_token(self, batch_size: int, device) -> torch.Tensor:
        """Get start token tensor."""
        return torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=device)

    def _forward_one_step(
        self,
        s_t: torch.Tensor,
        prev_token: torch.Tensor,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Single step forward pass for one action dimension."""
        B = s_t.shape[0]
        prev_tok_emb = self.token_embed(prev_token)
        inp = torch.cat([s_t, prev_tok_emb], dim=-1)
        x = inp.unsqueeze(1)  # (B, 1, d_model) where d_model = inp_dim
        for mamba_layer in self.mamba_layers:
            x, h = mamba_layer(x, h)
        x = x.squeeze(1)
        q_inp = torch.cat([x, inp], dim=-1)
        return q_inp, h

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        return_all_q: bool = False
    ) -> torch.Tensor:

        B, T, _ = states.shape
        s_flat = self.state_norm(states.reshape(B * T, -1))
        states = s_flat.reshape(B, T, -1)  # (B, T, state_dim)

        Q_all = torch.zeros(B, T, self.K, self.M, device=states.device)

        # Vectorized: process all T timesteps at once per action dimension
        # Autoregressive dependency is across K (actions), not T (timesteps)
        prev_tokens = self._get_start_token(B, states.device).unsqueeze(1).expand(B, T)  # (B, T)

        for i in range(self.K):
            # Build full-sequence input: (B, T, state_dim + token_dim)
            tok_emb = self.token_embed(prev_tokens)  # (B, T, token_dim)
            inp = torch.cat([states, tok_emb], dim=-1)  # (B, T, inp_dim)

            # Single Mamba/GRU pass over full T-length sequence
            x = inp  # (B, T, d_model)
            for mamba_layer in self.mamba_layers:
                x, _ = mamba_layer(x)
            # Residual already inside MambaBlock

            # Q-head for action dimension i
            q_inp = torch.cat([x, inp], dim=-1)  # (B, T, d_model + inp_dim)
            q_i = self.q_heads[i](q_inp)  # (B, T, M)

            # Min-max normalize per (b, t)
            q_min = q_i.min(-1, keepdim=True).values
            q_max = q_i.max(-1, keepdim=True).values
            q_i = (q_i - q_min) / (q_max - q_min + 1e-8)

            Q_all[:, :, i] = q_i

            # Prepare tokens for next action dimension
            if i < self.K - 1:
                if actions is not None:
                    prev_tokens = actions[:, :, i]  # (B, T)
                else:
                    prev_tokens = q_i.argmax(-1)  # (B, T)

        return Q_all

    @torch.no_grad()
    def act(
        self,
        s: torch.Tensor,
        deterministic: bool = True,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Greedy action selection at inference."""
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
            q_i = self.q_heads[i](q_inp)

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

    def forward_cumulative(
        self,
        cumulative_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with cumulative (state, action_history) input.

        Args:
            cumulative_input: (B, T, state_dim + K) tensor with cumulative history

        Returns:
            q_values: (B, T, K, M) Q-values for each action step
        """
        B, T, feat_dim = cumulative_input.shape
        s_flat = self.state_norm(cumulative_input.reshape(B * T, -1)[:, :self.state_dim])
        cumulative_input = cumulative_input.reshape(B * T, -1)
        cumulative_input[:, :self.state_dim] = s_flat
        cumulative_input = cumulative_input.reshape(B, T, -1)

        states = cumulative_input[:, :, :self.state_dim]  # (B, T, state_dim)
        action_hist = cumulative_input[:, :, self.state_dim:].long()  # (B, T, K)

        Q_all = torch.zeros(B, T, self.K, self.M, device=cumulative_input.device)

        # Vectorized: process all T timesteps at once per action dimension
        prev_tokens = self._get_start_token(B, cumulative_input.device).unsqueeze(1).expand(B, T)

        for i in range(self.K):
            tok_emb = self.token_embed(prev_tokens)  # (B, T, token_dim)
            inp = torch.cat([states, tok_emb], dim=-1)  # (B, T, inp_dim)

            x = inp
            for mamba_layer in self.mamba_layers:
                x, _ = mamba_layer(x)

            q_inp = torch.cat([x, inp], dim=-1)
            q_i = self.q_heads[i](q_inp)

            q_min = q_i.min(-1, keepdim=True).values
            q_max = q_i.max(-1, keepdim=True).values
            q_i_norm = (q_i - q_min) / (q_max - q_min + 1e-8)

            Q_all[:, :, i] = q_i_norm

            if i < self.K - 1:
                prev_tokens = action_hist[:, :, i]  # (B, T)

        return Q_all

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            'state_dim': self.state_dim,
            'K': self.K,
            'M': self.M,
            'd_model': self.d_model,
            'd_state': self.d_state,
            'num_hidden_mlp': self.num_hidden_mlp,
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