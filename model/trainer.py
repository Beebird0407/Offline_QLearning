"""
Q-Mamba Trainer with TD Learning and CQL Regularization

Implements the Q-loss function from the paper:
- TD error for i < K: (Q_{i,a_i^t} - max_j Q_{i+1,j}^t)^2 / 2
- TD error for i = K: β * (Q_{K,a_K^t} - (r^t + γ * max_j Q_{1,j}^{t+1}))^2 / 2
- Conservative regularization: (λ/2) * (Q_{i,j}^t)^2 for j ≠ a_i^t
"""

import os
import json
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .qmamba import QMamba


@dataclass
class TrainingConfig:
    """Training configuration."""
    lr: float = 5e-3
    gamma: float = 0.99
    beta: float = 10.0
    lam: float = 1.0
    batch_size: int = 64
    n_epochs: int = 300
    grad_clip: float = 0.5
    weight_decay: float = 1e-4
    device: str = 'cpu'
    save_dir: str = './checkpoints'
    eval_interval: int = 10
    checkpoint_interval: int = 50
    scheduler: str = 'none'  # 'none', 'cosine', or 'step'


class QMTrainer:
    """
    Trainer for Q-Mamba with:
    - TD learning with组合 Q-loss
    - CQL regularization
    - Gradient clipping
    - Checkpointing and resume
    """

    def __init__(
        self,
        model: QMamba,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            model: QMamba model to train
            config: Training configuration
            device: Device to use (auto-detect if None)
        """
        self.model = model

        if config is None:
            config = TrainingConfig()
        self.config = config

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model.to(device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None
        if config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )
        elif config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )

        # Training history
        self.history = {
            'total_loss': [],
            'td_loss': [],
            'cql_loss': [],
            'lr': []
        }

        # Checkpoint management
        self.best_loss = float('inf')
        self.epoch = 0
        self.global_step = 0

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute combined Q-loss (vectorized).

        Args:
            states: (B, T, state_dim)
            actions: (B, T, K) action bin indices
            rewards: (B, T) rewards
            next_states: (B, T, state_dim)
            dones: (B, T) done flags
            mask: (B, T) valid transition mask

        Returns:
            total_loss, td_loss, cql_loss
        """
        B, T, K = actions.shape
        M = self.model.M
        gamma = self.config.gamma
        beta = self.config.beta
        lam = self.config.lam

        # Forward pass
        Q = self.model(states, actions)  # (B, T, K, M)

        with torch.no_grad():
            # Next state Q-values
            dummy_actions = torch.zeros_like(actions)
            Q_next = self.model(next_states, dummy_actions)  # (B, T, K, M)

            # Bootstrap: max Q-value for next state's first dimension
            Q1_next_max = Q_next[:, :, 0, :].max(-1).values  # (B, T)

            # Compute returns: r^t + γ * max_j Q_{1,j}^{t+1}
            target_q = rewards + gamma * Q1_next_max * (1 - dones)
            target_q = torch.clamp(target_q, 0.0, 1.0)

        # Expand mask for broadcasting: (B, T, 1, 1)
        mask_4d = mask.unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1)

        # === TD Loss (vectorized) ===
        td_total = torch.tensor(0.0, device=self.device)

        # For i < K-1: chain TD
        for i in range(K - 1):
            # Q-selected: Q[:, :, i, :] gathered at actions[:, :, i]
            Q_sel = torch.gather(Q[:, :, i, :], dim=-1, index=actions[:, :, i].unsqueeze(-1)).squeeze(-1)  # (B, T)
            # Q-target: max of Q[:, :, i+1, :]
            Q_next_i1_max = Q[:, :, i + 1, :].max(-1).values  # (B, T)
            # TD error
            td_i = 0.5 * (Q_sel - Q_next_i1_max.detach()) ** 2
            td_total = td_total + (td_i * mask).sum()

        # For i = K-1 (final): reward TD
        Q_sel_last = torch.gather(Q[:, :, K - 1, :], dim=-1, index=actions[:, :, K - 1].unsqueeze(-1)).squeeze(-1)  # (B, T)
        td_last = beta * 0.5 * (Q_sel_last - target_q.detach()) ** 2
        td_total = td_total + (td_last * mask).sum()

        # === CQL Loss (vectorized) ===
        # Sum of Q-values for j != selected action, for all valid transitions
        # Q_i_expanded: (B, T, K, M), actions_expanded: (B, T, K, 1)
        actions_exp = actions.unsqueeze(-1).expand(B, T, K, 1)  # (B, T, K, 1)
        Q_sel_all = torch.gather(Q, dim=-1, index=actions_exp).squeeze(-1)  # (B, T, K)

        # Create mask for unselected actions: all Q values except selected ones
        # For each (b, t, i), we want sum over j != actions[b, t, i] of Q[b, t, i, j]^2
        # This equals: sum(all Q^2) - Q[selected]^2

        Q_sq = Q ** 2  # (B, T, K, M)
        Q_sq_sum_all = Q_sq.sum(dim=-1)  # (B, T, K) - sum over all M bins
        Q_sq_sum_selected = (Q_sel_all ** 2)  # (B, T, K) - sum over selected bins only (which is just 1 value)

        # CQL = (lam/2) * sum over j != selected of Q^2 = (lam/2) * (sum_all - sum_selected)
        cql_per_elem = (lam / 2) * (Q_sq_sum_all - Q_sq_sum_selected)  # (B, T, K)
        cql_total = (cql_per_elem * mask.unsqueeze(-1)).sum()

        # === Total loss ===
        n_valid = mask.sum() * K  # Total valid (b, t, i) triplets
        n_valid = n_valid.clamp(min=1.0)

        total_loss = (td_total + cql_total) / n_valid
        td_loss = td_total / n_valid
        cql_loss = cql_total / n_valid

        return total_loss, td_loss.item(), cql_loss.item()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Convert numpy arrays to tensors (blocking for safety)
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                dtype = np.float32 if x.dtype == np.float64 else np.int64
                return torch.from_numpy(x.astype(dtype)).to(self.device)
            return x.to(self.device)

        batch = {k: to_tensor(v) for k, v in batch.items()}

        # Zero grad
        self.optimizer.zero_grad()

        # Forward + backward
        loss, td, cql = self._compute_q_loss(**batch)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # Update history
        self.global_step += 1
        self.history['total_loss'].append(float(loss))
        self.history['td_loss'].append(td)
        self.history['cql_loss'].append(cql)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

        return {'total': float(loss), 'td': td, 'cql': cql}

    def train_epoch(
        self,
        data_loader
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {'total': [], 'td': [], 'cql': []}

        for batch in data_loader:
            losses = self.train_step(batch)
            epoch_losses['total'].append(losses['total'])
            epoch_losses['td'].append(losses['td'])
            epoch_losses['cql'].append(losses['cql'])

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    def evaluate(
        self,
        data_loader
    ) -> Dict[str, float]:
        """Evaluate on validation data."""
        self.model.eval()
        eval_losses = {'total': [], 'td': [], 'cql': []}

        with torch.no_grad():
            for batch in data_loader:
                def to_tensor(x):
                    if isinstance(x, np.ndarray):
                        dtype = np.float32 if x.dtype == np.float64 else np.int64
                        return torch.from_numpy(x.astype(dtype)).to(self.device)
                    return x.to(self.device)
                batch = {k: to_tensor(v) for k, v in batch.items()}
                loss, td, cql = self._compute_q_loss(**batch)
                eval_losses['total'].append(float(loss))
                eval_losses['td'].append(td)
                eval_losses['cql'].append(cql)

        return {k: float(np.mean(v)) for k, v in eval_losses.items()}

    def fit(
        self,
        train_loader,
        val_loader=None,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
        print_every: int = 1
    ) -> Dict:
        """
        Full training loop with per-step output option.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            n_epochs: Number of epochs (default: from config)
            verbose: Print progress
            print_every: Print every N steps (1 = every step, 10 = every 10 steps)

        Returns:
            Training history
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        os.makedirs(self.config.save_dir, exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Q-Mamba Training")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  β={self.config.beta}, λ={self.config.lam}, γ={self.config.gamma}")
            print(f"  Backend: {'Mamba' if self.model.uses_mamba else 'GRU (fallback)'}")
            print(f"  Parameters: {self.model.num_parameters:,}")
            print(f"  Print every: {print_every} step(s)")
            print(f"{'='*60}\n")

        # Estimate steps per epoch
        steps_per_epoch = len(train_loader)
        total_steps = n_epochs * steps_per_epoch
        print_step = max(1, print_every)

        import time
        step_times = []

        # Warmup: one dummy forward pass to trigger CUDA JIT compilation
        if verbose:
            print("Warming up CUDA...")
            print("-" * 60, flush=True)
        dummy_batch = train_loader.sample_batch()
        _ = self.train_step(dummy_batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("Warmup done, starting training.\n" + "=" * 60 + "\n", flush=True)

        for epoch in range(self.epoch + 1, self.epoch + n_epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()

            # Reset epoch tracking
            epoch_losses = {'total': [], 'td': [], 'cql': []}

            for step_idx, batch in enumerate(train_loader):
                step_start = time.time()

                # Train step
                losses = self.train_step(batch)
                epoch_losses['total'].append(losses['total'])
                epoch_losses['td'].append(losses['td'])
                epoch_losses['cql'].append(losses['cql'])

                step_time = time.time() - step_start
                step_times.append(step_time)

                # Print every N steps
                global_step = (epoch - 1) * steps_per_epoch + step_idx + 1
                if verbose and global_step % print_step == 0:
                    # Calculate ETA
                    avg_step_time = np.mean(step_times[-100:]) if step_times else 0
                    eta_seconds = avg_step_time * (total_steps - global_step)
                    eta_str = ""
                    if eta_seconds > 60:
                        eta_str = f" | ETA: {eta_seconds/60:.1f}min"
                    elif eta_seconds > 0:
                        eta_str = f" | ETA: {eta_seconds:.0f}s"

                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"Step[{global_step:5d}/{total_steps}] "
                          f"Loss={losses['total']:.4f} "
                          f"(TD={losses['td']:.4f}, CQL={losses['cql']:.4f}) "
                          f"| LR: {lr:.2e}{eta_str}", flush=True)

            # Epoch summary
            train_metrics = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start

            # Evaluate every N epochs
            val_metrics = None
            if val_loader is not None and epoch % max(1, self.config.eval_interval // 10) == 0:
                val_metrics = self.evaluate(val_loader)

            if verbose and epoch % self.config.eval_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch[{epoch:3d}/{n_epochs}] "
                msg += f"Avg Loss: {train_metrics['total']:.4f} "
                msg += f"(TD={train_metrics['td']:.4f}, CQL={train_metrics['cql']:.4f})"
                if val_metrics:
                    msg += f" | Val: {val_metrics['total']:.4f}"
                msg += f" | Time: {epoch_time:.1f}s | LR: {lr:.2e}"
                print(msg)

            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Save best model
            if train_metrics['total'] < self.best_loss:
                self.best_loss = train_metrics['total']
                self.save_checkpoint('best.pth')

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

        # Final save
        self.save_checkpoint('final.pth')

        # Save history
        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        if verbose:
            avg_step = np.mean(step_times) if step_times else 0
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
            print(f"Avg step time: {avg_step*1000:.1f}ms | Total time: {np.sum(step_times)/60:.1f}min")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.save_dir, filename)
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.model.state_dim,
                'K': self.model.K,
                'M': self.model.M,
                'd_model': self.model.d_model,
            },
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        """Load model checkpoint for resume training."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint

    @staticmethod
    def check_checkpoint(path: str) -> bool:
        """Check if checkpoint is valid."""
        if not os.path.exists(path):
            return False
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            return 'model_state_dict' in ckpt
        except Exception:
            return False


class AdaptiveCQLTrainer(QMTrainer):
    """
    Adaptive Conservative Q-Learning Trainer.

    Dynamically adjusts the conservatism coefficient (λ) based on:
    - Q-value optimism: E[max_a Q(s,a) - selected_Q(s,a)]
    - When Q is overestimated → increase λ to penalize
    - When Q is calibrated → decrease λ to allow exploitation

    Features:
    1. Adaptive lambda based on Q-value optimism
    2. Uncertainty estimation via dropout variance
    3. Per-dimension uncertainty weighting for CQL penalty
    """

    def __init__(
        self,
        model: QMamba,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
        lam_init: float = 1.0,
        lam_min: float = 0.01,
        lam_max: float = 0.5,
        optimism_threshold_high: float = 0.5,
        optimism_threshold_low: float = 0.1,
        dropout_p: float = 0.1,
        uncertainty_samples: int = 8
    ):
        """
        Args:
            model: QMamba model
            config: Training configuration
            device: Device to use
            lam_init: Initial lambda value
            lam_min: Minimum lambda
            lam_max: Maximum lambda
            optimism_threshold_high: Increase lambda if optimism > this
            optimism_threshold_low: Decrease lambda if optimism < this
            dropout_p: Dropout probability for uncertainty estimation
            uncertainty_samples: Number of dropout samples for uncertainty
        """
        super().__init__(model, config, device)

        # Adaptive lambda parameters
        self.lam = lam_init
        self.lam_init = lam_init
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.optimism_threshold_high = optimism_threshold_high
        self.optimism_threshold_low = optimism_threshold_low

        # Uncertainty estimation
        self.dropout_p = dropout_p
        self.uncertainty_samples = uncertainty_samples

        # State tracking
        self._q_optimism_ema = 0.0
        self._optimism_alpha = 0.1
        self._uncertainty_ema = None
        self._uncertainty_alpha = 0.1

        # Enable dropout for uncertainty estimation
        self._set_dropout(True)

        # History for adaptive lambda
        self.history['lambda'] = []

    def _set_dropout(self, enabled: bool):
        """Enable/disable dropout for uncertainty estimation."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.p = self.dropout_p if enabled else 0.0

    def _estimate_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate Q-value uncertainty via dropout-induced variance.

        Returns uncertainty score per state-action pair.
        """
        if self.uncertainty_samples <= 1:
            return torch.zeros_like(actions, dtype=torch.float32, device=self.device)

        self._set_dropout(True)
        q_samples = []

        with torch.no_grad():
            for _ in range(self.uncertainty_samples):
                q = self.model(states, actions)
                q_samples.append(q)

        # Compute variance across samples
        q_stack = torch.stack(q_samples, dim=0)  # [S, B, T, K, M]
        uncertainty = q_stack.var(dim=0).mean(dim=[-1])  # [B, T]

        self._set_dropout(False)
        return uncertainty

    def _compute_adaptive_lambda(
        self,
        Q: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """
        Compute adaptive lambda based on Q-value optimism.

        Q-optimism = E[max_a Q(s,a) - selected_Q(s,a)]
        High optimism → increase lambda to penalize overestimation
        Low optimism → decrease lambda to allow exploitation
        """
        B, T, K, _ = Q.shape

        # Compute optimism: difference between max Q and selected Q
        Q_max = Q.max(-1).values  # [B, T, K]
        Q_sel = torch.gather(Q, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # [B, T, K]

        optimism = (Q_max - Q_sel).detach()  # [B, T, K]

        # Only consider valid transitions
        valid_mask = mask.unsqueeze(-1) > 0.5  # [B, T, 1]
        optimism = optimism[valid_mask]

        if optimism.numel() == 0:
            return self.lam

        # Mean optimism across valid transitions
        mean_optimism = optimism.mean().item()

        # Update EMA
        self._q_optimism_ema = (
            (1 - self._optimism_alpha) * self._q_optimism_ema
            + self._optimism_alpha * mean_optimism
        )

        # Adapt lambda based on optimism
        if self._q_optimism_ema > self.optimism_threshold_high:
            # High optimism → increase lambda
            self.lam = min(self.lam * 1.05, self.lam_max)
        elif self._q_optimism_ema < self.optimism_threshold_low:
            # Low optimism → decrease lambda
            self.lam = max(self.lam * 0.95, self.lam_min)
        else:
            # Smooth convergence towards initial value
            self.lam = self.lam + 0.01 * (self.lam_init - self.lam)

        return self.lam

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Compute adaptive CQL loss with uncertainty-based penalty (vectorized).
        """
        B, T, K = actions.shape
        M = self.model.M
        gamma = self.config.gamma
        beta = self.config.beta

        # Forward pass
        Q = self.model(states, actions)

        # Estimate uncertainty for conservative penalty
        uncertainty = self._estimate_uncertainty(states, actions)  # [B, T]

        # Compute adaptive lambda
        adaptive_lam = self._compute_adaptive_lambda(Q, actions, mask)

        with torch.no_grad():
            # Next state Q-values
            dummy_actions = torch.zeros_like(actions)
            Q_next = self.model(next_states, dummy_actions)
            Q1_next_max = Q_next[:, :, 0, :].max(-1).values

            # Target
            target_q = rewards + gamma * Q1_next_max * (1 - dones)
            target_q = torch.clamp(target_q, 0.0, 1.0)

        # === TD Loss (vectorized) ===
        td_total = torch.tensor(0.0, device=self.device)

        # For i < K-1: chain TD
        for i in range(K - 1):
            Q_sel = torch.gather(Q[:, :, i, :], dim=-1, index=actions[:, :, i].unsqueeze(-1)).squeeze(-1)  # (B, T)
            Q_next_i1_max = Q[:, :, i + 1, :].max(-1).values  # (B, T)
            td_i = 0.5 * (Q_sel - Q_next_i1_max.detach()) ** 2
            td_total = td_total + (td_i * mask).sum()

        # For i = K-1 (final): reward TD
        Q_sel_last = torch.gather(Q[:, :, K - 1, :], dim=-1, index=actions[:, :, K - 1].unsqueeze(-1)).squeeze(-1)
        td_last = beta * 0.5 * (Q_sel_last - target_q.detach()) ** 2
        td_total = td_total + (td_last * mask).sum()

        # === CQL Loss with uncertainty (vectorized) ===
        # Normalize uncertainty to [0, 1] per batch
        unc_min = uncertainty.min()
        unc_max = uncertainty.max()
        if unc_max > unc_min:
            unc_norm = (uncertainty - unc_min) / (unc_max - unc_min + 1e-8)
        else:
            unc_norm = torch.zeros_like(uncertainty)

        # Expand for broadcasting: (B, T, K, 1)
        unc_norm_4d = unc_norm.unsqueeze(-1).unsqueeze(-1).expand(B, T, K, M)

        # Q-squared: (B, T, K, M)
        Q_sq = Q ** 2

        # Sum of Q^2 for unselected actions
        actions_exp = actions.unsqueeze(-1).expand(B, T, K, M)
        selected_mask = torch.zeros_like(Q, dtype=torch.bool)
        for i in range(K):
            selected_mask[:, :, i, :] = (actions[:, :, i:i+1] == torch.arange(M, device=actions.device)).transpose(0, 1)

        # Alternative: simpler way - compute sum_all and subtract sum_selected
        Q_sq_sum_all = Q_sq.sum(dim=-1)  # (B, T, K)
        Q_sq_selected = torch.gather(Q_sq, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1) ** 2  # (B, T, K)

        # CQL = (lam * (1 + 0.5 * unc) / 2) * (sum_all - sum_selected)
        unc_weight = adaptive_lam * (1.0 + 0.5 * unc_norm.unsqueeze(-1))  # (B, T, K)
        cql_per_elem = (unc_weight / 2) * (Q_sq_sum_all - Q_sq_selected)  # (B, T, K)
        cql_total = (cql_per_elem * mask.unsqueeze(-1)).sum()

        # === Total loss ===
        n_valid = mask.sum() * K
        n_valid = n_valid.clamp(min=1.0)

        total_loss = (td_total + cql_total) / n_valid
        td_loss = td_total / n_valid
        cql_loss = cql_total / n_valid

        return total_loss, td_loss.item(), cql_loss.item()

    def train_step(self, batch: Dict) -> Dict:
        """Single training step with adaptive CQL."""
        self.model.train()

        # Convert numpy arrays to tensors
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                if x.dtype == np.float64:
                    x = x.astype(np.float32)
                elif x.dtype == np.int64:
                    x = x.astype(np.int64)
                return torch.tensor(x, device=self.device)
            return x.to(self.device)

        batch = {k: to_tensor(v) for k, v in batch.items()}

        # Compute loss
        self.optimizer.zero_grad()
        loss, td, cql = self._compute_q_loss(**batch)

        # Backward
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        # Update tracking
        self.global_step += 1
        self.history['total_loss'].append(float(loss))
        self.history['td_loss'].append(td)
        self.history['cql_loss'].append(cql)
        self.history['lambda'].append(self.lam)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

        return {'total': float(loss), 'td': td, 'cql': cql}

    def fit(
        self,
        train_loader,
        val_loader=None,
        n_epochs: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """Full training loop with adaptive CQL."""
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        os.makedirs(self.config.save_dir, exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Adaptive CQL Training")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  β={self.config.beta}, γ={self.config.gamma}")
            print(f"  λ adaptive: [{self.lam_min}, {self.lam_max}], init={self.lam_init}")
            print(f"  Optimism thresholds: [{self.optimism_threshold_low}, {self.optimism_threshold_high}]")
            print(f"  Uncertainty samples: {self.uncertainty_samples}")
            print(f"{'='*60}\n")

        for epoch in range(self.epoch + 1, self.epoch + n_epochs + 1):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = None
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)

            # Print progress
            if verbose and epoch % self.config.eval_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch[{epoch:3d}/{n_epochs}] "
                msg += f"Train: {train_metrics['total']:.4f} (TD={train_metrics['td']:.4f}, CQL={train_metrics['cql']:.4f})"
                msg += f" | λ={self.lam:.4f}"
                if val_metrics:
                    msg += f" | Val: {val_metrics['total']:.4f}"
                msg += f" | LR: {lr:.2e}"
                print(msg)

            # Save checkpoint
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            # Save best model
            if train_metrics['total'] < self.best_loss:
                self.best_loss = train_metrics['total']
                self.save_checkpoint('best.pth')

        # Final save
        self.save_checkpoint('final.pth')

        # Save history
        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        if verbose:
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
            print(f"Lambda range: [{min(self.history['lambda'])}, {max(self.history['lambda'])}]")

        return self.history