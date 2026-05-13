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
    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(dim, dtype=torch.float32))
        self.register_buffer('running_var', torch.ones(dim, dtype=torch.float32))
        self._initialized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, force_gru: bool = False):
        super().__init__()
        self._force_gru = force_gru
        if _MAMBA_AVAILABLE and not force_gru:
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self._use_mamba = True
        else:
            self.ssm = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
            self._use_mamba = False
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_mamba and not self._force_gru:
            out = self.ssm(x)
            out = self.norm(out + x)
            return out, None
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
        self.ff1 = nn.Linear(d_model, num_hidden_mlp)
        self.ff2 = nn.Linear(num_hidden_mlp, M)
        self.start_token_idx = M

    def _get_start_token(self, batch_size: int, device) -> torch.Tensor:
        return torch.full((batch_size,), self.start_token_idx, dtype=torch.long, device=device)

    def _int_to_binary(self, int_tensor: torch.Tensor) -> torch.Tensor:
        """Bit-decomposition on GPU — avoids Python string conversion bottleneck."""
        shifts = torch.arange(self.token_dim - 1, -1, -1, device=int_tensor.device)
        return ((int_tensor.unsqueeze(-1) >> shifts) & 1).float()

    def parse_batch_to_input(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = states.shape
        K = actions.shape[2]
        s_norm = self.state_norm(states.reshape(B * T, -1)).reshape(B, T, -1)
        # Interleave state with each action token: [s, a0, s, a1, ...]
        s_aug = s_norm.unsqueeze(2).expand(B, T, K, self.state_dim).reshape(B, T * K, self.state_dim)
        # start token for each of the K action slots per timestep
        start_token = torch.ones(B, T, 1, self.token_dim, device=states.device)
        # binary-decompose all actions at once (fully on GPU)
        act_bin = self._int_to_binary(actions)[:, :, :-1]  # (B, T, K-1, token_dim)
        act_seq = torch.cat([start_token, act_bin], dim=2).reshape(B, T * K, self.token_dim)
        return torch.cat([s_aug, act_seq], dim=-1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, K = actions.shape[0], states.shape[1], actions.shape[2]
        x = self.parse_batch_to_input(states, actions)
        for mamba_layer in self.mamba_layers:
            x, _ = mamba_layer(x)
        x = F.leaky_relu(self.ff1(x))
        x = self.ff2(x)
        x = (x - x.min(-1, keepdim=True).values) / (x.max(-1, keepdim=True).values - x.min(-1, keepdim=True).values + 1e-20)
        return x.reshape(B, T, K, self.M)

    @torch.no_grad()
    def act(
        self,
        s: torch.Tensor,
        deterministic: bool = True,
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]
        s_norm = self.state_norm(s)
        start_token = self.token_embed(self._get_start_token(B, s.device))
        inp_seq = torch.cat([s_norm, start_token], dim=-1).unsqueeze(1)
        acts = torch.zeros(B, self.K, dtype=torch.long, device=s.device)
        q_values = torch.zeros(B, self.K, self.M, device=s.device)
        h_out = h
        for i in range(self.K):
            for mamba_layer in self.mamba_layers:
                inp_seq, h_out = mamba_layer(inp_seq, h_out)
            q_i = F.leaky_relu(self.ff1(inp_seq[:, -1:]))
            q_i = self.ff2(q_i).squeeze(1)
            q_min, q_max = q_i.min(-1, keepdim=True).values, q_i.max(-1, keepdim=True).values
            q_i_norm = (q_i - q_min) / (q_max - q_min + 1e-8)
            q_values[:, i] = q_i_norm
            if deterministic:
                a_i = q_i_norm.argmax(-1)
            else:
                a_i = torch.multinomial(F.softmax(q_i_norm, dim=-1), 1).squeeze(-1)
            acts[:, i] = a_i
            next_tok = self.token_embed(a_i)
            inp_seq = torch.cat([inp_seq[:, -1:, :], torch.cat([s_norm, next_tok], dim=-1).unsqueeze(1)], dim=1)
        return acts, q_values, h_out

    @torch.no_grad()
    def act_batch(self, states: torch.Tensor) -> torch.Tensor:
        B, T = states.shape[0], states.shape[1]
        s_norm = self.state_norm(states.reshape(B * T, -1)).reshape(B, T, -1)
        start_token = self.token_embed(self._get_start_token(B, states.device))
        inp = torch.cat([s_norm[:, 0:1], start_token.unsqueeze(1)], dim=-1)
        acts = torch.zeros(B, T, self.K, dtype=torch.long, device=states.device)
        q_vals = torch.zeros(B, T, self.K, self.M, device=states.device)
        for t in range(T):
            s_t = s_norm[:, t]
            for i in range(self.K):
                for mamba_layer in self.mamba_layers:
                    inp, _ = mamba_layer(inp)
                q_i = F.leaky_relu(self.ff1(inp[:, -1:]))
                q_i = self.ff2(q_i).squeeze(1)
                q_min, q_max = q_i.min(-1, keepdim=True).values, q_i.max(-1, keepdim=True).values
                q_i = (q_i - q_min) / (q_max - q_min + 1e-8)
                q_vals[:, t, i] = q_i
                a_i = q_i.argmax(-1)
                acts[:, t, i] = a_i
                next_tok = self.token_embed(a_i)
                inp = torch.cat([inp[:, -1:, :], torch.cat([s_t, next_tok], dim=-1).unsqueeze(1)], dim=1)
        return acts, q_vals

    def get_config(self) -> dict:
        return {
            'state_dim': self.state_dim, 'K': self.K, 'M': self.M,
            'd_model': self.d_model, 'd_state': self.d_state,
            'num_hidden_mlp': self.num_hidden_mlp,
        }

    @property
    def uses_mamba(self) -> bool:
        if len(self.mamba_layers) > 0:
            return getattr(self.mamba_layers[0], '_use_mamba', False)
        return False

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QEnsemble(nn.Module):
    """Ensemble of K independent QMamba networks with diversity regularization.

    Each member is initialised with a different random seed, producing distinct
    inductive biases.  The ensemble variance across members replaces MC dropout
    as the uncertainty signal for adaptive CQL.
    """

    def __init__(
        self,
        n_members: int = 5,
        state_dim: int = 9,
        K: int = 3,
        M: int = 16,
        d_model: int = 14,
        d_state: int = 32,
        n_layers: int = 1,
        num_hidden_mlp: int = 32,
        force_cpu: bool = False,
        base_seed: int = 42,
    ):
        super().__init__()
        self.n_members = n_members
        self.K = K
        self.M = M
        self.state_dim = state_dim
        self.d_model = d_model

        for i in range(n_members):
            member = QMamba(
                state_dim=state_dim,
                K=K,
                M=M,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                num_hidden_mlp=num_hidden_mlp,
                force_cpu=force_cpu,
            )
            # Re-initialise with a deterministic per-member seed so that
            # members start from genuinely different points.
            torch.manual_seed(base_seed + i * 100 + 1)
            for p in member.parameters():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                elif p.dim() == 1:
                    nn.init.zeros_(p)
            self.add_module(f'qnet_{i}', member)

    @property
    def members(self) -> list:
        return [m for m in self.children() if isinstance(m, QMamba)]

    @property
    def uses_mamba(self) -> bool:
        if self.n_members > 0:
            return self.members[0].uses_mamba
        return False

    @property
    def num_parameters(self) -> int:
        return sum(m.num_parameters for m in self.members)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return Q-values from all members.

        Returns:
            Tensor of shape (n_members, B, T, K_act, M)
        """
        return torch.stack([m(states, actions) for m in self.members], dim=0)

    def ensemble_variance(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Per-sample ensemble variance — replaces MC dropout for uncertainty.

        Returns:
            Tensor of shape (B, T) — mean variance across action dims and bins.
        """
        Q_all = self.forward(states, actions)                     # (n_members, B, T, K, M)
        var = Q_all.var(dim=0)                                     # (B, T, K, M)
        return var.mean(dim=[-1, -2])                              # (B, T)

    @torch.no_grad()
    def act(
        self,
        s: torch.Tensor,
        mode: str = 'pessimistic',
        deterministic: bool = True,
    ):
        """Ensemble action selection.

        Args:
            s: state tensor (B, state_dim) or (state_dim,)
            mode: 'pessimistic' (min-Q), 'mean' (avg-Q), or 'thompson'
                  (random member).
            deterministic: passed through to each member's act (only
                           relevant for 'thompson' where we use the
                           sampled member's own argmax).
        Returns:
            actions: (B, K) long tensor
            q_values: (B, K, M) — aggregated Q-values
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]

        acts_all = []
        qvals_all = []
        for member in self.members:
            a, qv, _ = member.act(s, deterministic=deterministic)
            acts_all.append(a)        # each (B, K)
            qvals_all.append(qv)      # each (B, K, M)

        acts_stack = torch.stack(acts_all, dim=0)    # (n_members, B, K)
        qvals_stack = torch.stack(qvals_all, dim=0)  # (n_members, B, K, M)

        if mode == 'pessimistic':
            # For each action bin, take the minimum Q across members, then argmax
            Q_agg = qvals_stack.min(dim=0).values      # (B, K, M)
        elif mode == 'mean':
            Q_agg = qvals_stack.mean(dim=0)            # (B, K, M)
        elif mode == 'thompson':
            idx = torch.randint(0, self.n_members, (1,)).item()
            Q_agg = qvals_stack[idx]                   # (B, K, M)
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")

        actions = Q_agg.argmax(-1)                      # (B, K)
        return actions, Q_agg

    @torch.no_grad()
    def act_batch(
        self,
        states: torch.Tensor,
        mode: str = 'pessimistic',
    ):
        """Batch ensemble inference over a full trajectory.

        Returns:
            actions: (B, T, K)
            q_values: (B, T, K, M) — aggregated Q-values
        """
        B, T = states.shape[0], states.shape[1]

        all_qvals = []
        for member in self.members:
            _, qv = member.act_batch(states)            # each (B, T, K, M)
            all_qvals.append(qv)

        qvals_stack = torch.stack(all_qvals, dim=0)     # (n_members, B, T, K, M)

        if mode == 'pessimistic':
            Q_agg = qvals_stack.min(dim=0).values
        elif mode == 'mean':
            Q_agg = qvals_stack.mean(dim=0)
        elif mode == 'thompson':
            idx = torch.randint(0, self.n_members, (1,)).item()
            Q_agg = qvals_stack[idx]
        else:
            raise ValueError(f"Unknown ensemble mode: {mode}")

        actions = Q_agg.argmax(-1)                       # (B, T, K)
        return actions, Q_agg

    def get_config(self) -> dict:
        return {
            'n_members': self.n_members,
            'state_dim': self.state_dim,
            'K': self.K,
            'M': self.M,
            'd_model': self.d_model,
        }
