import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .qmamba import QMamba


@dataclass
class TrainingConfig:
    """Training configuration.

    Defaults match configs/default.yaml. main.py overrides all fields from yaml.
    """
    lr: float = 0.001
    gamma: float = 0.99
    beta: float = 10.0
    lam: float = 1.0
    batch_size: int = 32
    n_epochs: int = 100
    grad_clip: float = 100.0
    weight_decay: float = 1e-4
    device: str = 'cpu'
    save_dir: str = './checkpoints'
    eval_interval: int = 10
    checkpoint_interval: int = 50
    scheduler: str = 'none'  # 'none', 'cosine', or 'step'
    algorithm: str = 'Alg0'  # Algorithm type: Alg0, Alg1, or Alg2
    print_every: int = 1  # Print every N epochs
    seed: int = 42  # Random seed for reproducibility


class QMTrainer:
    """Trainer for Q-Mamba with TD learning, CQL regularization, and checkpointing."""

    def __init__(
        self,
        model: QMamba,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None
    ):
        self.model = model

        if config is None:
            config = TrainingConfig()
        self.config = config

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.model.to(device)

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = None
        if config.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs, eta_min=config.lr * 0.01
            )
        elif config.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        elif config.scheduler == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[50, 75, 100, 125], gamma=0.8
            )

        self.history = {
            'total_loss': [],
            'td_loss': [],
            'cql_loss': [],
            'val_loss': [],
            'lr': []
        }
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
        B, T, K = actions.shape
        M, gamma, beta, lam = self.model.M, self.config.gamma, self.config.beta, self.config.lam

        Q = self.model(states, actions)  # (B, T, K, M)

        # --- Chain-style TD (matching Q-Mamba-main) ---
        Q_sel = torch.gather(Q, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # (B, T, K)
        Q_max = Q.max(-1).values  # (B, T, K)

        q_pred_rest = Q_sel[..., :-1]       # (B, T, K-1)
        q_target_rest = Q_max[..., 1:]      # (B, T, K-1)
        td_rest = F.mse_loss(q_pred_rest, q_target_rest.detach())

        q_pred_last = Q_sel[..., -1]        # (B, T)
        q_target_first = Q_max[..., 0]      # (B, T)
        q_target_last = torch.cat([q_target_first[..., 1:], torch.zeros(B, 1, device=self.device)], dim=-1)
        q_target_last = rewards + gamma * q_target_last
        td_last = beta * F.mse_loss(q_pred_last, q_target_last.detach())

        td_loss = td_rest + td_last

        # --- CQL loss (matching Q-Mamba-main) ---
        Q_flat = Q.reshape(B * T * K, M)
        actions_flat = actions.reshape(B * T * K, 1)
        dataset_action_mask = torch.zeros_like(Q_flat).scatter_(-1, actions_flat, torch.ones_like(Q_flat))
        q_actions_not_taken = Q_flat[~dataset_action_mask.bool()]
        cql_loss = ((q_actions_not_taken) ** 2).mean()

        total_loss = 0.5 * td_loss + 0.5 * lam * cql_loss

        return total_loss, td_loss.item(), cql_loss.item()

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                if x.dtype == np.float64:
                    x = x.astype(np.float32)
                elif x.dtype != np.float32 and x.dtype != np.int64:
                    x = x.astype(np.int64)
                return torch.from_numpy(x).to(self.device)
            return x.to(self.device)

        batch = {k: to_tensor(v) for k, v in batch.items()}

        self.optimizer.zero_grad()
        loss, td, cql = self._compute_q_loss(**batch)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        self.global_step += 1
        # Only record finite loss values
        if torch.isfinite(loss):
            self.history['total_loss'].append(float(loss))
            self.history['td_loss'].append(td)
            self.history['cql_loss'].append(cql)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        elif self.history['total_loss']:
            # If loss is NaN/Inf, use last valid value
            self.history['total_loss'].append(self.history['total_loss'][-1])
            self.history['td_loss'].append(self.history['td_loss'][-1])
            self.history['cql_loss'].append(self.history['cql_loss'][-1])
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
                        if x.dtype == np.float64:
                            x = x.astype(np.float32)
                        elif x.dtype != np.float32 and x.dtype != np.int64:
                            x = x.astype(np.int64)
                        return torch.from_numpy(x).to(self.device)
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
        """Full training loop."""
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        os.makedirs(self.config.save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(self.config.save_dir, 'tensorboard'))

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Q-Mamba Training")
            print(f"  Algorithm: {self.config.algorithm}")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  β={self.config.beta}, λ={self.config.lam}, γ={self.config.gamma}")
            print(f"  Backend: {'Mamba' if self.model.uses_mamba else 'GRU (fallback)'}")
            print(f"  Parameters: {self.model.num_parameters:,}")
            print(f"  Print every: {print_every} epoch(s)")
            print(f"{'='*60}\n")

        import time
        total_time = 0

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
            epoch_losses = {'total': [], 'td': [], 'cql': []}

            for _, batch in enumerate(train_loader):
                losses = self.train_step(batch)
                epoch_losses['total'].append(losses['total'])
                epoch_losses['td'].append(losses['td'])
                epoch_losses['cql'].append(losses['cql'])

            train_metrics = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            val_metrics = None
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)

            # Track val_loss in history
            self.history['val_loss'].append(val_metrics['total'] if val_metrics else None)

            # TensorBoard
            writer.add_scalar('Loss/train_total', train_metrics['total'], epoch)
            writer.add_scalar('Loss/train_td', train_metrics['td'], epoch)
            writer.add_scalar('Loss/train_cql', train_metrics['cql'], epoch)
            writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            if val_metrics:
                writer.add_scalar('Loss/val_total', val_metrics['total'], epoch)

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch[{epoch:3d}/{n_epochs}] "
                msg += f"Avg Loss: {train_metrics['total']:.4f} "
                msg += f"(TD={train_metrics['td']:.4f}, CQL={train_metrics['cql']:.4f})"
                if val_metrics:
                    msg += f" | Val: {val_metrics['total']:.4f}"
                msg += f" | Time: {epoch_time:.1f}s | LR: {lr:.2e}"
                print(msg)

            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            if train_metrics['total'] < self.best_loss:
                self.best_loss = train_metrics['total']
                self.save_checkpoint('best.pth')

            if self.scheduler is not None:
                self.scheduler.step()

        self.save_checkpoint('final.pth')

        # Add training summary to history
        self.history['training_time_minutes'] = float(total_time / 60)
        self.history['best_loss'] = float(self.best_loss)

        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        writer.close()

        if verbose:
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
            print(f"Total time: {total_time/60:.1f}min")

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
    """Adaptive CQL trainer with uncertainty-driven λ adjustment (PBRL/UWAC style)."""

    def __init__(
        self,
        model: QMamba,
        config: Optional[TrainingConfig] = None,
        device: Optional[str] = None,
        lam_init: float = 1.0,
        lam_min: float = 0.01,
        lam_max: float = 0.5,
        dropout_p: float = 0.1,
        uncertainty_samples: int = 8,
        uncertainty_interval: int = 10
    ):
        super().__init__(model, config, device)

        self.lam = lam_init  # Start at configured init value
        self.lam_init = lam_init
        self.lam_min = lam_min
        self.lam_max = lam_max

        self.dropout_p = dropout_p
        self.uncertainty_samples = uncertainty_samples
        self.uncertainty_interval = uncertainty_interval

        self._uncertainty_ema = None
        self._uncertainty_alpha = 0.1
        self._cached_uncertainty = None
        self._uncertainty_update_counter = 0

        # Multi-scale uncertainty: track Q-value stability across estimates
        self._prev_q_mean = None
        self._q_stability_ema = 0.0

        K = getattr(model, 'K', 3)
        k_scale = (3.0 / K) ** 0.5  # Alg0=1.0, Alg1=0.55, Alg2=0.43

        self._unc_fast = 0.0
        self._unc_slow = 0.0
        self._alpha_fast = 0.3
        self._alpha_slow = 0.02 * k_scale
        self._decrease_threshold = -0.08 / k_scale
        self._decrease_coef = 0.01 * k_scale
        self._effective_lam_min = max(self.lam_min, self.lam_init * (1.0 - k_scale))
        self._drift_above = 0.01
        self._drift_below = 0.003 / k_scale

        self._set_dropout(True)
        self.history['lambda'] = []

    def _set_dropout(self, enabled: bool):
        """Enable/disable dropout for uncertainty estimation."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.p = self.dropout_p if enabled else 0.0

    def _estimate_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        force_update: bool = False
    ) -> torch.Tensor:
        """Multi-scale uncertainty: dropout variance + Q-value stability."""
        B, T = states.shape[0], states.shape[1]
        should_update = (self._uncertainty_update_counter % self.uncertainty_interval == 0) or force_update

        if should_update and self.uncertainty_samples > 1:
            self._set_dropout(True)
            q_samples = []
            with torch.no_grad():
                for _ in range(self.uncertainty_samples):
                    q = self.model(states, actions)
                    q_samples.append(q)

            q_stack = torch.stack(q_samples, dim=0)
            unc_dropout = q_stack.var(dim=0).mean(dim=[-1]).mean(dim=[-1])  # [B, T]

            # Q-value stability: how much have Q-values changed since last estimate?
            q_mean = q_stack.mean(dim=0).mean()
            if self._prev_q_mean is not None:
                q_change = abs(q_mean.item() - self._prev_q_mean)
                if self._q_stability_ema == 0.0:
                    self._q_stability_ema = q_change  # seed with first value
                else:
                    self._q_stability_ema = 0.5 * self._q_stability_ema + 0.5 * q_change
            self._prev_q_mean = q_mean.item()

            # Blend: dropout variance dominant, stability as light stabilizer
            unc_combined = 0.85 * unc_dropout + 0.15 * self._q_stability_ema

            self._set_dropout(False)

            if self._cached_uncertainty is None:
                self._cached_uncertainty = unc_combined
            else:
                self._cached_uncertainty = (
                    (1 - self._uncertainty_alpha) * self._cached_uncertainty
                    + self._uncertainty_alpha * unc_combined
                )

        self._uncertainty_update_counter += 1

        if self._cached_uncertainty is None:
            return torch.zeros(B, T, dtype=torch.float32, device=self.device)

        return self._cached_uncertainty

    def _compute_adaptive_lambda(
        self,
        uncertainty: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """Dual-baseline adaptive λ with K-dependent decrease conservatism.

        Fast baseline (α=0.3): detects sudden spikes → raise λ (same for all K).

        Slow baseline (α scaled by √(3/K)): detects long-term decreases → lower λ.
        Larger K → slower baseline + harder threshold + smaller steps + higher
        effective λ_min.  This breaks the positive-feedback spiral where lower λ
        → more overfit → lower uncertainty → even lower λ.
        """
        valid_mask = mask > 0.5
        valid_uncertainty = uncertainty[valid_mask]

        if valid_uncertainty.numel() == 0:
            return self.lam

        unc_mean = valid_uncertainty.mean().item()

        if self._unc_fast == 0.0:
            self._unc_fast = unc_mean
            self._unc_slow = unc_mean
            return self.lam

        self._unc_fast = (1 - self._alpha_fast) * self._unc_fast + self._alpha_fast * unc_mean
        self._unc_slow = (1 - self._alpha_slow) * self._unc_slow + self._alpha_slow * unc_mean

        rel_fast = (unc_mean - self._unc_fast) / (self._unc_fast + 1e-8)
        rel_slow = (unc_mean - self._unc_slow) / (self._unc_slow + 1e-8)

        if rel_fast > 0.15:
            step = 0.01 * min(rel_fast - 0.15, 2.0)
        elif rel_slow < self._decrease_threshold:
            step = self._decrease_coef * max(rel_slow - self._decrease_threshold, -2.0)
        else:
            drift_coef = self._drift_above if self.lam > self.lam_init else self._drift_below
            step = drift_coef * (self.lam_init - self.lam) / max(self.lam_max - self.lam_min, 1e-6)

        step = max(-0.015, min(0.015, step))
        self.lam = self.lam + step
        self.lam = max(self._effective_lam_min, min(self.lam_max, self.lam))

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
        """Compute adaptive CQL loss with uncertainty-based penalty."""
        B, T, K = actions.shape
        M, gamma, beta = self.model.M, self.config.gamma, self.config.beta

        Q = self.model(states, actions)
        uncertainty = self._estimate_uncertainty(states, actions)
        adaptive_lam = self._compute_adaptive_lambda(uncertainty, mask)

        Q_sel = torch.gather(Q, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        Q_max = Q.max(-1).values

        q_pred_rest = Q_sel[..., :-1]
        q_target_rest = Q_max[..., 1:]
        td_rest = F.mse_loss(q_pred_rest, q_target_rest.detach())

        q_pred_last = Q_sel[..., -1]
        q_target_first = Q_max[..., 0]
        q_target_last = torch.cat([q_target_first[..., 1:], torch.zeros(B, 1, device=self.device)], dim=-1)
        q_target_last = rewards + gamma * q_target_last
        td_last = beta * F.mse_loss(q_pred_last, q_target_last.detach())

        td_loss = td_rest + td_last

        Q_flat = Q.reshape(B * T * K, M)
        actions_flat = actions.reshape(B * T * K, 1)
        dataset_action_mask = torch.zeros_like(Q_flat).scatter_(-1, actions_flat, torch.ones_like(Q_flat))
        q_actions_not_taken = Q_flat[~dataset_action_mask.bool()]
        cql_loss = ((q_actions_not_taken) ** 2).mean()

        total_loss = 0.5 * td_loss + 0.5 * adaptive_lam * cql_loss

        return total_loss, td_loss.item(), cql_loss.item()

    def train_step(self, batch: Dict) -> Dict:
        """Single training step with adaptive CQL."""
        self.model.train()

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                if x.dtype == np.float64:
                    x = x.astype(np.float32)
                elif x.dtype != np.float32 and x.dtype != np.int64:
                    x = x.astype(np.int64)
                return torch.from_numpy(x).to(self.device)
            return x.to(self.device)

        batch = {k: to_tensor(v) for k, v in batch.items()}

        self.optimizer.zero_grad()
        loss, td, cql = self._compute_q_loss(**batch)
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

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
        verbose: bool = True,
        print_every: int = 1
    ) -> Dict:
        """Full training loop with adaptive CQL."""
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        print_every_cfg = getattr(self.config, 'print_every', print_every)

        os.makedirs(self.config.save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(self.config.save_dir, 'tensorboard'))

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Adaptive CQL Training")
            print(f"  Algorithm: {self.config.algorithm}")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  β={self.config.beta}, γ={self.config.gamma}")
            print(f"  λ adaptive: [{self.lam_min}, {self.lam_max}], init={self.lam_init}")
            print(f"  Uncertainty-driven adaptation (PBRL/UWAC style)")
            print(f"  Uncertainty samples: {self.uncertainty_samples}, interval: {self.uncertainty_interval}")
            print(f"  Print every: {print_every_cfg} epoch(s)")
            print(f"{'='*60}\n")

        import time
        total_time = 0

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
            epoch_losses = {'total': [], 'td': [], 'cql': []}

            for _, batch in enumerate(train_loader):
                losses = self.train_step(batch)
                epoch_losses['total'].append(losses['total'])
                epoch_losses['td'].append(losses['td'])
                epoch_losses['cql'].append(losses['cql'])

            train_metrics = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            val_metrics = None
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)

            # Track val_loss in history
            self.history['val_loss'].append(val_metrics['total'] if val_metrics else None)

            # TensorBoard
            writer.add_scalar('Loss/train_total', train_metrics['total'], epoch)
            writer.add_scalar('Loss/train_td', train_metrics['td'], epoch)
            writer.add_scalar('Loss/train_cql', train_metrics['cql'], epoch)
            writer.add_scalar('Lambda', self.lam, epoch)
            writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            if val_metrics:
                writer.add_scalar('Loss/val_total', val_metrics['total'], epoch)

            if verbose:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f"Epoch[{epoch:3d}/{n_epochs}] "
                msg += f"Avg Loss: {train_metrics['total']:.4f} "
                msg += f"(TD={train_metrics['td']:.4f}, CQL={train_metrics['cql']:.4f})"
                msg += f" | λ={self.lam:.4f}"
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

            if self.scheduler is not None:
                self.scheduler.step()

        # Final save
        self.save_checkpoint('final.pth')

        # Add training summary to history
        self.history['training_time_minutes'] = float(total_time / 60)
        self.history['best_loss'] = float(self.best_loss)

        # Save history
        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        writer.close()

        if verbose:
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
            print(f"Lambda range: [{min(self.history['lambda'])}, {max(self.history['lambda'])}]")
            print(f"Total time: {total_time/60:.1f}min")

        return self.history


class EnsembleAdaptiveCQLTrainer(QMTrainer):
    """Ensemble Q-network trainer with diversity regularisation and per-member adaptive CQL.

    Key differences from AdaptiveCQLTrainer:

    1. **Ensemble variance replaces MC dropout** — uncertainty is measured as the
       disagreement across K independently-initialised Q-networks rather than the
       variance of a single network under dropout masks.  This gives a more
       principled signal, especially in OOD regions where a single network may be
       uniformly overconfident.

    2. **Per-member adaptive lambda** — each ensemble member maintains its own
       dual-baseline EMA (fast/slow) and adjusts its personal CQL strength
       independently.  Members that happen to be more uncertain about the current
       batch will apply stronger regularisation.

    3. **Diversity regularisation** — a mutual-information term is added to the
       loss that encourages members to disagree on OOD actions while staying
       consistent on in-distribution actions.  Without this term, independently-
       initialised networks tend to converge to similar solutions over time.
    """

    def __init__(
        self,
        model,
        config=None,
        device=None,
        lam_init: float = 1.0,
        lam_min: float = 0.01,
        lam_max: float = 2.0,
        diversity_weight: float = 0.1,
        diversity_type: str = 'mi',
        **kwargs,
    ):
        super().__init__(model, config, device)

        self.lam_init = lam_init
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.diversity_weight = diversity_weight
        self.diversity_type = diversity_type

        n_members = model.n_members
        self.n_members = n_members
        K = getattr(model, 'K', 3)

        # Per-member adaptive-lambda state
        self._lam = [lam_init] * n_members
        self._unc_fast = [0.0] * n_members
        self._unc_slow = [0.0] * n_members

        k_scale = (3.0 / K) ** 0.5
        self._alpha_fast = 0.3
        self._alpha_slow = 0.02 * k_scale
        self._decrease_threshold = -0.08 / k_scale
        self._decrease_coef = 0.01 * k_scale
        self._effective_lam_min = max(self.lam_min, self.lam_init * (1.0 - k_scale))
        self._drift_above = 0.01
        self._drift_below = 0.003 / k_scale

        self.history['lambda_mean'] = []
        self.history['lambda_std'] = []
        self.history['diversity_loss'] = []
        self.history['per_member_loss'] = []

    # ------------------------------------------------------------------
    #  Uncertainty estimation — ensemble variance (no MC dropout needed)
    # ------------------------------------------------------------------

    def _estimate_uncertainty(self, states, actions, **_kw) -> torch.Tensor:
        """Use ensemble variance as the uncertainty signal."""
        return self.model.ensemble_variance(states, actions)  # (B, T)

    # ------------------------------------------------------------------
    #  Per-member adaptive lambda
    # ------------------------------------------------------------------

    def _compute_adaptive_lambda(
        self,
        uncertainty: torch.Tensor,
        member_idx: int,
        mask: torch.Tensor,
    ) -> float:
        """Update the adaptive lambda for a single ensemble member."""
        valid_mask = mask > 0.5
        valid_unc = uncertainty[valid_mask]

        if valid_unc.numel() == 0:
            return self._lam[member_idx]

        unc_mean = valid_unc.mean().item()

        if self._unc_fast[member_idx] == 0.0:
            self._unc_fast[member_idx] = unc_mean
            self._unc_slow[member_idx] = unc_mean
            return self._lam[member_idx]

        self._unc_fast[member_idx] = (
            (1 - self._alpha_fast) * self._unc_fast[member_idx]
            + self._alpha_fast * unc_mean
        )
        self._unc_slow[member_idx] = (
            (1 - self._alpha_slow) * self._unc_slow[member_idx]
            + self._alpha_slow * unc_mean
        )

        rel_fast = (unc_mean - self._unc_fast[member_idx]) / (self._unc_fast[member_idx] + 1e-8)
        rel_slow = (unc_mean - self._unc_slow[member_idx]) / (self._unc_slow[member_idx] + 1e-8)

        if rel_fast > 0.15:
            step = 0.01 * min(rel_fast - 0.15, 2.0)
        elif rel_slow < self._decrease_threshold:
            step = self._decrease_coef * max(rel_slow - self._decrease_threshold, -2.0)
        else:
            cur_lam = self._lam[member_idx]
            drift_coef = self._drift_above if cur_lam > self.lam_init else self._drift_below
            step = drift_coef * (self.lam_init - cur_lam) / max(self.lam_max - self.lam_min, 1e-6)

        step = max(-0.015, min(0.015, step))
        self._lam[member_idx] = self._lam[member_idx] + step
        self._lam[member_idx] = max(self._effective_lam_min,
                                    min(self.lam_max, self._lam[member_idx]))

        return self._lam[member_idx]

    # ------------------------------------------------------------------
    #  Diversity regularisation
    # ------------------------------------------------------------------

    def _diversity_loss(
        self,
        Q_all: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Direct variance maximisation on OOD Q-values (Plan B).

        Gradient path:  Q → var → loss   (no softmax bottleneck).

        Minimising this loss:
        1. Maximises variance across members on OOD actions (→ diversity)
        2. Penalises large Q-values on OOD actions (→ stability)
        """
        n_mem, B, T, K_act, M = Q_all.shape

        Q_flat = Q_all.reshape(n_mem, B * T * K_act, M)
        acts_flat = actions.reshape(B * T * K_act, 1)
        dataset_mask = torch.zeros(B * T * K_act, M, dtype=torch.bool, device=Q_flat.device)
        dataset_mask.scatter_(-1, acts_flat, True)
        ood_mask = ~dataset_mask

        if ood_mask.sum() == 0:
            return torch.tensor(0.0, device=Q_flat.device)

        # Select Q-values on OOD actions for each member
        Q_ood = Q_flat[:, ood_mask]               # (n_mem, N_ood)

        # Maximise inter-member variance on OOD actions
        var_ood = Q_ood.var(dim=0).mean()          # scalar — want to maximise

        # Stabilise: keep OOD Q-values from diverging
        q_l2 = (Q_ood ** 2).mean()                 # scalar — want to minimise

        # Loss = -λ_div * variance + λ_div * 0.01 * L2
        #  → minimise → variance goes UP, Q magnitude stays bounded
        return -self.diversity_weight * var_ood + self.diversity_weight * 0.01 * q_l2

    # ------------------------------------------------------------------
    #  Core loss
    # ------------------------------------------------------------------

    def _compute_q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor
    ):
        """Compute ensemble loss = mean per-member (TD + λ_k·CQL) + diversity."""
        B, T, K_act = actions.shape
        M, gamma, beta = self.model.M, self.config.gamma, self.config.beta

        Q_all = self.model(states, actions)  # (n_members, B, T, K, M)
        uncertainty = self._estimate_uncertainty(states, actions)  # (B, T)

        total_member_loss = 0.0
        all_td = 0.0
        all_cql = 0.0

        for k in range(self.n_members):
            Q_k = Q_all[k]

            Q_sel = torch.gather(Q_k, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            Q_max = Q_k.max(-1).values

            q_pred_rest = Q_sel[..., :-1]
            q_target_rest = Q_max[..., 1:]
            td_rest = F.mse_loss(q_pred_rest, q_target_rest.detach())

            q_pred_last = Q_sel[..., -1]
            q_target_first = Q_max[..., 0]
            q_target_last = torch.cat(
                [q_target_first[..., 1:], torch.zeros(B, 1, device=self.device)], dim=-1
            )
            q_target_last = rewards + gamma * q_target_last
            td_last = beta * F.mse_loss(q_pred_last, q_target_last.detach())

            td_loss = td_rest + td_last

            Q_flat = Q_k.reshape(B * T * K_act, M)
            actions_flat = actions.reshape(B * T * K_act, 1)
            dataset_action_mask = torch.zeros_like(Q_flat).scatter_(
                -1, actions_flat, torch.ones_like(Q_flat)
            )
            q_not_taken = Q_flat[~dataset_action_mask.bool()]
            cql_loss = (q_not_taken ** 2).mean()

            lam_k = self._compute_adaptive_lambda(uncertainty, k, mask)
            loss_k = 0.5 * td_loss + 0.5 * lam_k * cql_loss
            total_member_loss = total_member_loss + loss_k
            all_td = all_td + td_loss.item()
            all_cql = all_cql + cql_loss.item()

        avg_member_loss = total_member_loss / self.n_members
        avg_td = all_td / self.n_members
        avg_cql = all_cql / self.n_members

        div_loss = self._diversity_loss(Q_all, actions)

        total_loss = avg_member_loss + div_loss

        return (
            total_loss,
            avg_td,
            avg_cql,
            div_loss.item(),
            sum(self._lam) / len(self._lam),
            float(torch.tensor(self._lam).std()),
        )

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict):
        """Single training step with ensemble diversity + per-member adaptive CQL."""
        self.model.train()

        def to_tensor(x):
            if isinstance(x, np.ndarray):
                if x.dtype == np.float64:
                    x = x.astype(np.float32)
                elif x.dtype != np.float32 and x.dtype != np.int64:
                    x = x.astype(np.int64)
                return torch.from_numpy(x).to(self.device)
            return x.to(self.device)

        batch = {k: to_tensor(v) for k, v in batch.items()}

        self.optimizer.zero_grad()
        loss, td, cql, div, lam_mean, lam_std = self._compute_q_loss(**batch)

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        self.global_step += 1
        self.history['total_loss'].append(float(loss))
        self.history['td_loss'].append(td)
        self.history['cql_loss'].append(cql)
        self.history['diversity_loss'].append(div)
        self.history['lambda_mean'].append(lam_mean)
        self.history['lambda_std'].append(lam_std)
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

        return {
            'total': float(loss), 'td': td, 'cql': cql,
            'diversity': div, 'lam_mean': lam_mean, 'lam_std': lam_std,
        }

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate on validation data (ensemble version — unpacks 6 values)."""
        self.model.eval()
        eval_losses = {'total': [], 'td': [], 'cql': [], 'diversity': []}

        with torch.no_grad():
            for batch in data_loader:
                def to_tensor(x):
                    if isinstance(x, np.ndarray):
                        if x.dtype == np.float64:
                            x = x.astype(np.float32)
                        elif x.dtype != np.float32 and x.dtype != np.int64:
                            x = x.astype(np.int64)
                        return torch.from_numpy(x).to(self.device)
                    return x.to(self.device)
                batch = {k: to_tensor(v) for k, v in batch.items()}
                total, td, cql, div, _lam, _lam_std = self._compute_q_loss(**batch)
                eval_losses['total'].append(float(total))
                eval_losses['td'].append(td)
                eval_losses['cql'].append(cql)
                eval_losses['diversity'].append(div)

        return {k: float(np.mean(v)) for k, v in eval_losses.items()}

    # ------------------------------------------------------------------
    #  Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader=None,
        n_epochs=None,
        verbose: bool = True,
        print_every: int = 1,
    ):
        if n_epochs is None:
            n_epochs = self.config.n_epochs

        print_every_cfg = getattr(self.config, 'print_every', print_every)

        os.makedirs(self.config.save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(self.config.save_dir, 'tensorboard'))

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Ensemble Adaptive CQL Training")
            print(f"  Algorithm: {self.config.algorithm}")
            print(f"  Ensemble members: {self.n_members}")
            print(f"  Diversity: {self.diversity_type} (weight={self.diversity_weight})")
            print(f"  Device: {self.device}")
            print(f"  Epochs: {n_epochs}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  β={self.config.beta}, γ={self.config.gamma}")
            print(f"  λ per-member adaptive: [{self.lam_min}, {self.lam_max}], init={self.lam_init}")
            print(f"  Backend: {'Mamba' if self.model.uses_mamba else 'GRU (fallback)'}")
            print(f"  Parameters: {self.model.num_parameters:,}")
            print(f"  Print every: {print_every_cfg} epoch(s)")
            print(f"{'='*60}\n")

        import time
        total_time = 0

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
            epoch_losses = {
                'total': [], 'td': [], 'cql': [],
                'diversity': [], 'lam_mean': [], 'lam_std': [],
            }

            for _, batch in enumerate(train_loader):
                losses = self.train_step(batch)
                epoch_losses['total'].append(losses['total'])
                epoch_losses['td'].append(losses['td'])
                epoch_losses['cql'].append(losses['cql'])
                epoch_losses['diversity'].append(losses['diversity'])
                epoch_losses['lam_mean'].append(losses['lam_mean'])
                epoch_losses['lam_std'].append(losses['lam_std'])

            train_metrics = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            val_metrics = None
            if val_loader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)

            self.history['val_loss'].append(val_metrics['total'] if val_metrics else None)

            writer.add_scalar('Loss/train_total', train_metrics['total'], epoch)
            writer.add_scalar('Loss/train_td', train_metrics['td'], epoch)
            writer.add_scalar('Loss/train_cql', train_metrics['cql'], epoch)
            writer.add_scalar('Loss/diversity', train_metrics['diversity'], epoch)
            writer.add_scalar('Lambda/mean', train_metrics['lam_mean'], epoch)
            writer.add_scalar('Lambda/std', train_metrics['lam_std'], epoch)
            writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            if val_metrics:
                writer.add_scalar('Loss/val_total', val_metrics['total'], epoch)

            if verbose and (epoch % print_every_cfg == 0 or epoch == 1):
                lr = self.optimizer.param_groups[0]['lr']
                lam = train_metrics['lam_mean']
                lam_s = train_metrics['lam_std']
                msg = (f"Epoch[{epoch:3d}/{n_epochs}] "
                       f"Loss={train_metrics['total']:.4f} "
                       f"(TD={train_metrics['td']:.4f}, CQL={train_metrics['cql']:.4f}, "
                       f"DIV={train_metrics['diversity']:.4f})")
                msg += f" | λ={lam:.3f}±{lam_s:.3f}"
                if val_metrics:
                    msg += f" | Val={val_metrics['total']:.4f}"
                msg += f" | {epoch_time:.1f}s"
                print(msg)

            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            if train_metrics['total'] < self.best_loss:
                self.best_loss = train_metrics['total']
                self.save_checkpoint('best.pth')

            if self.scheduler is not None:
                self.scheduler.step()

        self.save_checkpoint('final.pth')

        self.history['training_time_minutes'] = float(total_time / 60)
        self.history['best_loss'] = float(self.best_loss)

        history_path = os.path.join(self.config.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        writer.close()

        if verbose:
            final_lams = self._lam
            print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
            print(f"Final per-member lambdas: {[f'{l:.4f}' for l in final_lams]}")
            print(f"Total time: {total_time/60:.1f}min")

        return self.history

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.config.save_dir, filename)
        os.makedirs(self.config.save_dir, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.get_config(),
            'trainer_config': {
                'lam_init': self.lam_init,
                'lam_min': self.lam_min,
                'lam_max': self.lam_max,
                'n_members': self.n_members,
                'diversity_weight': self.diversity_weight,
                'diversity_type': self.diversity_type,
            },
            'per_member_lam': self._lam,
            'history': self.history,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)