"""
Visualization Module

Implements:
- Training loss curves
- Convergence curves
- Box plots for bin ablation
- Neuroevolution curves
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import json


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training loss curves.

    Args:
        history: Dict with keys 'total_loss', 'td_loss', 'cql_loss'
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    if 'total_loss' in history and history['total_loss']:
        axes[0].plot(history['total_loss'], alpha=0.6)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

    # TD loss
    if 'td_loss' in history and history['td_loss']:
        axes[1].plot(history['td_loss'], alpha=0.6, color='orange')
        axes[1].set_title('TD Loss')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)

    # CQL loss
    if 'cql_loss' in history and history['cql_loss']:
        axes[2].plot(history['cql_loss'], alpha=0.6, color='green')
        axes[2].set_title('CQL Loss')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence(
    results_dict: Dict[str, List[List[float]]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 'Convergence Curves'
):
    """
    Plot convergence curves for multiple algorithms.

    Args:
        results_dict: Dict of {name: [list of convergence curves per run]}
        labels: Optional list of display names
        save_path: Path to save figure
        show: Whether to display the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if labels is None:
        labels = list(results_dict.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for i, (name, curves) in enumerate(results_dict.items()):
        curves = np.array(curves)

        # Mean and std across runs
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        x = np.arange(len(mean_curve))

        label = labels[i] if i < len(labels) else name
        ax.plot(x, mean_curve, label=label, color=colors[i])
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except UserWarning:
        pass  # Ignore tight_layout warnings

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_boxplot(
    data_dict: Dict[str, List[float]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 'Performance by Bin Count'
):
    """
    Plot box plot for ablation study.

    Args:
        data_dict: Dict of {name: list of performance values}
        labels: Optional list of display names
        save_path: Path to save figure
        show: Whether to display the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(data_dict.keys())
    values = [data_dict[n] for n in names]

    if labels is None:
        labels = names

    bp = ax.boxplot(values, labels=labels, patch_artist=True)

    # Color boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Performance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_neuroevolution(
    results_dict: Dict[str, List[List[float]]],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 'Neuroevolution on MuJoCo'
):
    """
    Plot neuroevolution optimization curves.

    Args:
        results_dict: Dict of {name: [list of fitness curves per run]}
        labels: Optional list of display names
        save_path: Path to save figure
        show: Whether to display the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if labels is None:
        labels = list(results_dict.keys())

    colors = plt.cm.Set2(np.linspace(0, 1, len(results_dict)))

    for i, (name, curves) in enumerate(results_dict.items()):
        curves = np.array(curves)

        # Mean and std across runs
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)
        x = np.arange(len(mean_curve))

        label = labels[i] if i < len(labels) else name
        ax.plot(x, mean_curve, label=label, color=colors[i], linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color=colors[i])

    ax.set_xlabel('Generation')
    ax.set_ylabel('Policy Fitness')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_ablation_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = 'Ablation Study',
    cmap: str = 'viridis'
):
    """
    Plot heatmap for ablation study.

    Args:
        data: (n_rows, n_cols) array of performance values
        row_labels: Labels for rows (e.g., λ values)
        col_labels: Labels for columns (e.g., β values)
        save_path: Path to save figure
        show: Whether to display the plot
        title: Plot title
        cmap: Colormap name
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Performance', rotation=-90, va='bottom')

    ax.set_title(title)
    ax.set_xlabel('β')
    ax.set_ylabel('λ')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def save_results(results: Dict, path: str):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_results(path: str) -> Dict:
    """Load evaluation results from JSON."""
    with open(path, 'r') as f:
        return json.load(f)