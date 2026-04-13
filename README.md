# Q-Mamba: Offline Meta Black-Box Optimization

A PyTorch implementation of Q-Mamba for learning to configure evolutionary algorithms via offline reinforcement learning.

## Project Structure

```
Offline_QLearning/
├── algorithms/              # Low-level optimization algorithms
│   ├── __init__.py
│   ├── alg0.py             # DE/current-to-rand/1/exponential + LPSR
│   ├── alg1.py             # Hybrid GA + DE (10 params)
│   └── alg2.py             # 4-subgroup heterogeneous (16 params)
├── data/                    # Dataset collection
│   ├── __init__.py
│   ├── bbob_suite.py       # CoCo BBOB test suite
│   ├── trajectory.py       # Trajectory representation
│   ├── meta_dataset.py     # E&E Dataset builder
│   └── ee_dataset.pkl      # Cached E&E dataset
├── env/                     # Environment
│   ├── __init__.py
│   ├── state.py            # 9-dim state representation
│   └── action.py           # Action discretization/tokenization
├── model/                   # Q-Mamba and baselines
│   ├── __init__.py
│   ├── qmamba.py           # Q-Mamba model
│   ├── trainer.py          # Training pipeline
│   ├── agent.py            # Inference agent
│   └── baselines/          # Baseline methods
│       ├── __init__.py
│       ├── dt.py           # Decision Transformer
│       ├── dema.py         # Mamba DT
│       ├── meta_bbo.py      # RLPSO, LDE, GLEET
│       ├── qdt.py          # Q-DT
│       ├── qt.py           # Q-Transformer
│       └── q_transformer.py
├── utils/                   # Evaluation and visualization
│   ├── __init__.py
│   ├── evaluation.py       # Benchmarking
│   └── visualization.py    # Plotting
├── configs/                 # Configuration files
│   └── default.yaml        # Default config
├── Trained_model/           # Trained model checkpoints
│   └── Alg0_CQL0.1/        # Example: Alg0 with λ=0.1
├── results/                 # Evaluation results
├── main.py                  # Main entry point
└── requirements.txt        # Dependencies
```

## Installation

```bash
cd Offline_QLearning
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
python main.py --mode eval --checkpoint Trained_model/Alg0_CQL0.1/final.pth
```

## Configuration

Modify `configs/default.yaml`:

```yaml
# === Dataset Configuration ===
dataset:
  dim: 5                     # Problem dimension (5, 10, 20, or 50)
  train_instances: 16       # Number of training instances per function
  test_instances: 1         # Number of testing instances per function
  mu: 0.5                   # Mix ratio (fraction from pretrained baselines)
  n_total_trajectories: 10000  # Total trajectory count D
  trajectory_length: 500    # T: Number of generations per trajectory

# === State/Action Configuration ===
state_action:
  state_dim: 9              # State dimension
  K: 3                      # Number of action parameters (3 for Alg0)
  M: 16                     # Number of bins per parameter

# === Algorithm Configuration ===
algorithm:
  type: "Alg2"              # Alg0, Alg1, or Alg2
  pop_size: 20              # Population size
  use_lpsr: true            # Use Linear Population Size Reduction
  min_pop_size: 4           # Minimum population size for LPSR

# === Model Configuration ===
model:
  type: "qmamba"            # qmamba, dt, dema, qdt, qt, q_transformer
  d_model: 128              # Hidden dimension
  d_state: 16               # Mamba state dimension
  n_layers: 1               # Number of Mamba/Transformer layers

# === Training Configuration ===
training:
  lr: 0.001                 # Learning rate
  gamma: 0.99               # Discount factor
  beta: 10.0                # TD loss weight for final action
  lam: 0.001                # Conservative regularization coefficient
  batch_size: 64            # Batch size
  n_epochs: 100             # Number of training epochs
  grad_clip: 0.5           # Gradient clipping
  weight_decay: 0.0001      # Weight decay
  eval_interval: 10         # Evaluation interval (epochs)
  checkpoint_interval: 50   # Checkpoint saving interval
  scheduler: cosine         # Learning rate scheduler: cosine, step, or none
```

## Algorithms

| Algorithm | Description | Parameters |
|-----------|-------------|------------|
| **Alg0** | DE/current-to-rand/1 + exponential + LPSR | F1, F2, Cr |
| **Alg1** | Hybrid GA + DE | 10 params (GA: sigma, Cr, ratio, elite; DE: F1, F2, Cr, ratio; cm1, cm2) |
| **Alg2** | 4-subgroup heterogeneous | 16 params (4 subgroups × 4 params) |

### Switching Algorithms

Change `algorithm.type` in config:
- `"Alg0"` → DE with LPSR (3 params)
- `"Alg1"` → Hybrid GA-DE (10 params)
- `"Alg2"` → 4-subgroup heterogeneous (16 params)

## State Representation (9-dim)

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
| 9 | Improvement | (best_so_far - f_best) / \|best_so_far\| |

## Action Representation

- **Discretization**: M=16 bins per parameter
- **Tokenization**: 5-bit binary encoding (00000~01111)
- **Start token**: 11111 (31)
- **Autoregressive**: Q1 → Q2 → ... → QK

## Q-Loss Formula

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
