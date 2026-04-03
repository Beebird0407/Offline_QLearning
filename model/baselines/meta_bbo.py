"""
MetaBBO: Online Meta-Learning Black-Box Optimization Baselines

Implements:
- RLPSO: PSO with MLP-learned parameters
- LDE: DE with LSTM-learned parameters
- GLEET: Global-Local Evolution with Transformer
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import pickle


class RLPSO(nn.Module):
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # w, c1, c2
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict PSO parameters from state."""
        params = self.net(state)
        # Clamp to reasonable ranges
        w = torch.sigmoid(params[:, 0]) * 0.4 + 0.1  # [0.1, 0.5]
        c1 = torch.sigmoid(params[:, 1]) * 1.0 + 0.5  # [0.5, 1.5]
        c2 = torch.sigmoid(params[:, 2]) * 1.0 + 0.5  # [0.5, 1.5]
        return torch.stack([w, c1, c2], dim=-1)

    @torch.no_grad()
    def predict(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Predict PSO parameters from numpy state."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        params = self.forward(s).squeeze(0).numpy()
        return float(params[0]), float(params[1]), float(params[2])

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, state_dim: int = 9) -> 'RLPSO':
        model = cls(state_dim=state_dim)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model


class LDE(nn.Module):
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # F, Cr
        )

    def forward(self, state_seq: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        lstm_out, hidden = self.lstm(state_seq, hidden)
        # Use last output
        last_out = lstm_out[:, -1, :]
        params = self.fc(last_out)

        # Clamp to reasonable ranges
        F = torch.sigmoid(params[:, 0]) * 0.8 + 0.2  # [0.2, 1.0]
        Cr = torch.sigmoid(params[:, 1])  # [0.0, 1.0]

        return torch.stack([F, Cr], dim=-1), hidden

    @torch.no_grad()
    def predict(self, state: np.ndarray, hidden: Optional[Tuple] = None) -> Tuple[Tuple[float, float], Optional[Tuple]]:
        """Predict DE parameters from numpy state."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        params, hidden = self.forward(s, hidden)
        params = params.squeeze(0).numpy()
        return (float(params[0]), float(params[1])), hidden

    def save(self, path: str):
        torch.save({'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, state_dim: int = 9) -> 'LDE':
        model = cls(state_dim=state_dim)
        ckpt = torch.load(path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        return model


class GLEET(nn.Module):
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # State embedding
        self.embed = nn.Linear(state_dim, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Global and local context pooling
        self.global_pool = nn.Linear(hidden_dim, hidden_dim)
        self.local_pool = nn.Linear(hidden_dim, hidden_dim)

        # Parameter prediction heads
        self.fc_global = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # global F, Cr
        )

        self.fc_local = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # local F, Cr
        )

        # Combination weight
        self.alpha = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor, pop_history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]

        # Embed current state
        state_emb = self.embed(state)  # (B, hidden_dim)

        if pop_history is not None and pop_history.shape[1] > 0:
            # Encode population history
            pop_emb = self.embed(pop_history)  # (B, T, hidden_dim)

            # Transformer processing
            trans_out = self.transformer(pop_emb)  # (B, T, hidden_dim)

            # Global context (mean pooling)
            global_ctx = trans_out.mean(1)  # (B, hidden_dim)
            global_ctx = torch.relu(self.global_pool(global_ctx))

            # Local context (last state)
            local_ctx = trans_out[:, -1, :]  # (B, hidden_dim)
            local_ctx = torch.relu(self.local_pool(local_ctx))

            # Combine contexts
            combined = torch.cat([global_ctx, local_ctx], dim=-1)

            # Predict global and local parameters
            global_params = torch.sigmoid(self.fc_global(global_ctx))
            local_params = torch.sigmoid(self.fc_local(combined))

            # Learn combination weight
            alpha = self.alpha(combined)  # (B, 1)

            # Combine
            F = alpha * local_params[:, 0] + (1 - alpha) * global_params[:, 0]
            Cr = alpha * local_params[:, 1] + (1 - alpha) * global_params[:, 1]
        else:
            # No history, use only current state
            global_params = torch.sigmoid(self.fc_global(state_emb))
            F = global_params[:, 0]
            Cr = global_params[:, 1]

        return F, Cr

    @torch.no_grad()
    def predict(self, state: np.ndarray, pop_history: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Predict DE parameters from numpy arrays."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        pop_h = torch.tensor(pop_history, dtype=torch.float32).unsqueeze(0) if pop_history is not None else None
        F, Cr = self.forward(s, pop_h)
        return float(F.squeeze()), float(Cr.squeeze())

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, state_dim: int = 9) -> 'GLEET':
        model = cls(state_dim=state_dim)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model


class MetaBBOManager:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.models: Dict[str, nn.Module] = {}
        self._states: Dict[str, any] = {}  # Hidden states for LSTM

    def load_baseline(self, name: str, path: str, state_dim: int = 9) -> bool:
        try:
            if name.lower() == 'rlpso':
                self.models[name] = RLPSO.load(path, state_dim)
            elif name.lower() == 'lde':
                self.models[name] = LDE.load(path, state_dim)
            elif name.lower() == 'gleet':
                self.models[name] = GLEET.load(path, state_dim)
            else:
                return False

            self.models[name].to(self.device)
            self.models[name].eval()
            return True
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return False

    def predict(self, name: str, state: np.ndarray) -> Tuple[int, ...]:
        if name not in self.models:
            raise ValueError(f"Model {name} not loaded")

        model = self.models[name]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if name.lower() == 'rlpso':
                params = model(state_tensor).squeeze(0).cpu().numpy()
                w, c1, c2 = params
                # Convert to bins (assuming M=16)
                w_bin = int(np.clip(w / 0.04, 0, 15))
                c1_bin = int(np.clip((c1 - 0.5) / 0.07, 0, 15))
                c2_bin = int(np.clip((c2 - 0.5) / 0.07, 0, 15))
                return (w_bin, c1_bin, c2_bin)

            elif name.lower() == 'lde':
                params, _ = model(state_tensor.unsqueeze(1))
                params = params.squeeze(0).cpu().numpy()
                F, Cr = params
                F_bin = int(np.clip(F / 0.067, 0, 15))
                Cr_bin = int(np.clip(Cr / 0.067, 0, 15))
                return (F_bin, Cr_bin)

            elif name.lower() == 'gleet':
                F, Cr = model(state_tensor)
                F, Cr = F.squeeze().item(), Cr.squeeze().item()
                F_bin = int(np.clip(F / 0.067, 0, 15))
                Cr_bin = int(np.clip(Cr / 0.067, 0, 15))
                return (F_bin, Cr_bin)

        return (8, 8)  # Default middle values

    def reset(self):
        """Reset hidden states."""
        self._states = {}

    @property
    def available_models(self) -> list:
        return list(self.models.keys())


def create_random_baseline(K: int = 3, M: int = 16) -> callable:
    def predict(state: np.ndarray, rng: Optional[np.random.RandomState] = None) -> Tuple[int, ...]:
        if rng is None:
            rng = np.random.RandomState()
        return tuple(rng.randint(0, M) for _ in range(K))
    return predict


def create_exploit_baseline(K: int = 3, M: int = 16) -> callable:
    def predict(state: np.ndarray, rng: Optional[np.random.RandomState] = None, t: int = 0, T: int = 500) -> Tuple[int, ...]:
        if rng is None:
            rng = np.random.RandomState()

        # Progress-based exploitation
        progress = t / T if T > 0 else 0

        # F decreases over time (less mutation as we converge)
        F_base = int(12 - progress * 4)
        F_bin = int(np.clip(F_base + rng.randint(-1, 2), 0, M - 1))

        # Cr increases over time (more exploitation)
        Cr_base = int(8 + progress * 6)
        Cr_bin = int(np.clip(Cr_base + rng.randint(-1, 2), 0, M - 1))

        # Additional params for Alg1/Alg2
        bins = [F_bin, Cr_bin]
        while len(bins) < K:
            bins.append(rng.randint(0, M))

        return tuple(bins)

    return predict
