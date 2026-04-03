# Q-Mamba: Offline Meta Black-Box Optimization

A PyTorch implementation of Q-Mamba for learning to configure evolutionary algorithms via offline reinforcement learning.

## Project Structure

```
Q/
├── algorithms/           # Low-level optimization algorithms
│   ├── alg0.py          # DE/current-to-rand/1/exponential + LPSR
│   ├── alg1.py          # Hybrid GA + DE (10 params)
│   └── alg2.py          # 4-subgroup heterogeneous (16 params)
├── data/                 # Dataset collection
│   ├── bbob_suite.py    # CoCo BBOB test suite
│   ├── trajectory.py    # Trajectory representation
│   └── meta_dataset.py   # E&E Dataset builder
├── env/                  # Environment
│   ├── state.py         # 9-dim state representation
│   └── action.py        # Action discretization/tokenization
├── model/               # Q-Mamba and baselines
│   ├── qmamba.py        # Q-Mamba model
│   ├── trainer.py       # Training pipeline
│   ├── agent.py         # Inference agent
│   └── baselines/       # Baseline methods
│       ├── dt.py        # Decision Transformer
│       ├── dema.py      # Mamba DT
│       ├── meta_bbo.py  # RLPSO, LDE, GLEET
│       ├── qdt.py       # Q-DT
│       ├── qt.py        # Q-Transformer
│       └── q_transformer.py
├── utils/               # Evaluation and visualization
│   ├── evaluation.py    # Benchmarking
│   └── visualization.py # Plotting
├── configs/             # Configuration files
│   └── default.yaml     # Default config
├── main.py              # Main entry point
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install -r requirements.txt

# Optional: Install Mamba SSM for GPU acceleration
# pip install mamba-ssm

# Optional: For MuJoCo neuroevolution tasks
# pip install mujoco-py
```

## Quick Start

### Training

```bash
python main.py --mode train --config configs/default.yaml
```

### Evaluation

```bash
python main.py --mode eval --checkpoint checkpoints/best.pth
```

### Q-Loss Formula

**TD Loss (Temporal Difference):**
- For i < K: `TD = 0.5 * (Q_{i,a_i} - max_j Q_{i+1,j})^2`
- For i = K: `TD = β * 0.5 * (Q_{K,a_K} - (r + γ * max_j Q_{1,j}^{next}))^2`

**CQL Loss (Conservative):**
- `CQL = (λ/2) * Σ_{j≠a_i} (Q_{i,j})^2`

**Total Loss:**
```
total_loss = (TD_total + CQL_total) / n_valid
where n_valid = mask.sum() × K
```

## Configuration

Modify `configs/default.yaml`:

```yaml
# === Dataset Configuration ===
dataset:
  dim: 5                    # Problem dimension (5, 10, 20, 50)
  train_instances: 16        # Training instances per function
  test_instances: 8         # Testing instances
  mu: 0.5                   # Mix ratio (exploit/total)
  n_total_trajectories: 10000  # Total trajectory count D
  trajectory_length: 500     # T: Generations per trajectory

# === State/Action Configuration ===
state_action:
  state_dim: 9              # State dimension
  K: 3                      # Action parameters (F1, F2, Cr)
  M: 16                     # Bins per parameter

# === Training Configuration ===
training:
  lr: 0.005                # Learning rate (5e-3)
  gamma: 0.99               # Discount factor
  beta: 10.0               # Final action TD weight
  lam: 1.0                 # CQL regularization
  batch_size: 64           # Batch size
  n_epochs: 300            # Training epochs
  grad_clip: 0.5           # Gradient clipping
  eval_interval: 10        # Eval every N epochs
  checkpoint_interval: 50   # Save checkpoint every N epochs
  print_every: 1            # Print every N steps
```

## Key Components

### Algorithms

| Algorithm | Description | Parameters |
|-----------|-------------|------------|
| **Alg0** | DE/current-to-rand/1 + exponential + LPSR | F1, F2, Cr |
| **Alg1** | Hybrid GA + DE | 10 params (GA: sigma, Cr, ratio, elite; DE: F1, F2, Cr, ratio; cm1, cm2) |
| **Alg2** | 4-subgroup heterogeneous | 16 params (4 subgroups × 4 params) |

### State Representation (9-dim)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Norm. mean fitness | (f_mean - f_mean) / f_std = 0 |
| 2 | Norm. best fitness | (f_best - f_mean) / f_std |
| 3 | Norm. worst fitness | (f_worst - f_mean) / f_std |
| 4 | Population diversity | std(distances to centroid) |
| 5 | Population correlation | Mean pairwise dimension correlation |
| 6 | Progress | t / T |
| 7 | Remaining | 1 - t/T |
| 8 | Log progress | log(t+1) / log(T+1) |
| 9 | Improvement | (best_so_far - f_best) / |best_so_far| |

### Action Representation

- **Discretization**: M=16 bins per parameter
- **Tokenization**: 5-bit binary encoding (00000~01111)
- **Start token**: 11111 (31)
- **Autoregressive**: Q1 → Q2 → ... → QK

## MetaBBO Baselines (Pretrained)

The project supports using pretrained MetaBBO agents for exploitation trajectories:

```python
from model.baselines import RLPSO, LDE, GLEET, MetaBBOManager

# Load pretrained baselines
manager = MetaBBOManager(device='cuda')
manager.load_baseline('rlpso', 'path/to/rlpso.pth', state_dim=9)
manager.load_baseline('lde', 'path/to/lde.pth', state_dim=9)
manager.load_baseline('gleet', 'path/to/gleet.pth', state_dim=9)

# Build dataset with pretrained baselines
meta_agents = {
    'rlpso': manager.models['rlpso'],
    'lde': manager.models['lde'],
    'gleet': manager.models['gleet']
}

train_trajs, val_trajs = builder.build(
    n_total=10000,
    meta_agents=meta_agents  # Use pretrained for exploitation
)
```

## Adaptive CQL Trainer

For adaptive regularization:

```python
from model.trainer import AdaptiveCQLTrainer

trainer = AdaptiveCQLTrainer(
    model=model,
    lam_init=1.0,           # Initial λ
    lam_min=0.01,            # Min λ
    lam_max=0.5,             # Max λ
    optimism_threshold_high=0.5,
    optimism_threshold_low=0.1,
    dropout_p=0.1,
    uncertainty_samples=8
)
```

Adaptive CQL:
- Increases λ when Q-values are overestimated (high optimism)
- Decreases λ when Q-values are calibrated (low optimism)
- Uses dropout variance for uncertainty estimation

## Citation


