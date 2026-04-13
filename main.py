"""Q-Mamba: Offline Meta Black-Box Optimization."""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Q-Mamba: Offline MetaBBO')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'ablation', 'all'],
                        help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        config = {}

    # Set device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Execute mode
    if args.mode == 'train' or args.mode == 'all':
        from model.qmamba import QMamba
        from model.trainer import QMTrainer, TrainingConfig
        from data.bbob_suite import BBOBSuite
        from data.meta_dataset import EEDatasetBuilder, MetaDataLoader
        from algorithms.alg0 import Alg0Optimizer
        from algorithms.alg1 import Alg1Optimizer
        from algorithms.alg2 import Alg2Optimizer

        # Get algorithm type from config
        alg_type = config.get('algorithm', {}).get('type', 'Alg0')
        alg_map = {'Alg0': Alg0Optimizer, 'Alg1': Alg1Optimizer, 'Alg2': Alg2Optimizer}
        OptimizerClass = alg_map.get(alg_type, Alg0Optimizer)

        # Build dataset
        print("\n[1/4] Building E&E dataset...")
        bbob_suite = BBOBSuite(
            dim=config.get('dataset', {}).get('dim', 5),
            train_instances=config.get('dataset', {}).get('train_instances', 16),
            test_instances=config.get('dataset', {}).get('test_instances', 8)
        )

        K = config.get('state_action', {}).get('K', 3)
        M = config.get('state_action', {}).get('M', 16)
        pop_size = config.get('algorithm', {}).get('pop_size', 20)
        use_lpsr = config.get('algorithm', {}).get('use_lpsr', True)
        min_pop_size = config.get('algorithm', {}).get('min_pop_size', 4)

        builder = EEDatasetBuilder(
            bbob_suite=bbob_suite,
            optimizer_class=OptimizerClass,
            K=K,
            M=M,
            pop_size=pop_size,
            mu=config.get('dataset', {}).get('mu', 0.5),
            T=config.get('dataset', {}).get('trajectory_length', 500),
            seed=42,
            use_lpsr=use_lpsr,
            min_pop_size=min_pop_size
        )

        dataset_path = config.get('paths', {}).get('dataset_path', './data/ee_dataset.pkl')
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        # Load existing dataset if available
        if os.path.exists(dataset_path):
            print(f"  Loading existing dataset from {dataset_path}")
            train_trajs, val_trajs, _ = EEDatasetBuilder.load_dataset(dataset_path)
        else:
            print(f"  Building new dataset...")
            train_trajs, val_trajs = builder.build(
                n_total=config.get('dataset', {}).get('n_total_trajectories', 10000),
                save_path=dataset_path
            )

        # Create data loaders
        train_loader = MetaDataLoader(
            train_trajs,
            batch_size=config.get('training', {}).get('batch_size', 64),
            K=K
        )
        val_loader = MetaDataLoader(
            val_trajs,
            batch_size=config.get('training', {}).get('batch_size', 64),
            K=K
        )

        # Create model
        print("\n[2/4] Creating Q-Mamba model...")
        state_dim = config.get('state_action', {}).get('state_dim', 9)
        d_model = config.get('model', {}).get('d_model', 128)
        d_state = config.get('model', {}).get('d_state', 16)
        n_layers = config.get('model', {}).get('n_layers', 1)

        model = QMamba(
            state_dim=state_dim,
            K=K,
            M=M,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers
        )

        print(f"  Model parameters: {model.num_parameters:,}")
        print(f"  Backend: {'Mamba' if model.uses_mamba else 'GRU (fallback)'}")

        # Train
        print("\n[3/4] Training...")
        train_config = TrainingConfig(
            lr=config.get('training', {}).get('lr', 5e-3),
            gamma=config.get('training', {}).get('gamma', 0.99),
            beta=config.get('training', {}).get('beta', 10.0),
            lam=config.get('training', {}).get('lam', 1.0),
            batch_size=config.get('training', {}).get('batch_size', 64),
            n_epochs=config.get('training', {}).get('n_epochs', 300),
            grad_clip=config.get('training', {}).get('grad_clip', 0.5),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-4),
            device=device,
            save_dir=config.get('paths', {}).get('checkpoint_dir', './checkpoints'),
            eval_interval=config.get('training', {}).get('eval_interval', 10),
            checkpoint_interval=config.get('training', {}).get('checkpoint_interval', 50),
            scheduler=config.get('training', {}).get('scheduler', 'none'),
            algorithm=alg_type
        )

        trainer = QMTrainer(model, train_config, device)
        history = trainer.fit(train_loader, val_loader, verbose=True)

        # Save history
        results_dir = config.get('paths', {}).get('results_dir', './results')
        os.makedirs(results_dir, exist_ok=True)

        from utils.visualization import plot_training_curves
        plot_training_curves(
            history,
            save_path=os.path.join(results_dir, 'training_curves.png'),
            show=False
        )
        print(f"\n  Training curves saved to {results_dir}/training_curves.png")

    if args.mode == 'eval' or args.mode == 'all':
        from model.agent import QMAgent
        from utils.evaluation import benchmark_in_distribution
        from data.bbob_suite import BBOBSuite

        checkpoint_path = args.checkpoint or config.get('paths', {}).get('checkpoint_dir', './checkpoints') + '/best.pth'

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Please train a model first or provide --checkpoint")
            return

        print(f"\n[4/4] Evaluating on BBOB test set...")
        agent = QMAgent.from_checkpoint(checkpoint_path, device=device)

        bbob_suite = BBOBSuite(
            dim=config.get('dataset', {}).get('dim', 5),
            test_instances=config.get('dataset', {}).get('test_instances', 8)
        )

        results = benchmark_in_distribution(
            agent=agent,
            bbob_suite=bbob_suite,
            n_runs=config.get('evaluation', {}).get('n_runs', 19),
            pop_size=config.get('algorithm', {}).get('pop_size', 20),
            T=config.get('dataset', {}).get('trajectory_length', 500)
        )

        # Save results
        results_dir = config.get('paths', {}).get('results_dir', './results')
        os.makedirs(results_dir, exist_ok=True)

        from utils.visualization import save_results, plot_convergence
        save_results(results, os.path.join(results_dir, 'evaluation_results.json'))

        # Plot convergence curves
        convergence_data = {}
        for name, res in results['results'].items():
            if 'convergence_curve' in res:
                convergence_data[name] = [res['convergence_curve']]

        if convergence_data:
            plot_convergence(
                convergence_data,
                save_path=os.path.join(results_dir, 'convergence_curves.png'),
                show=False,
                title='BBOB Test Set Convergence'
            )

        print(f"\n  Results saved to {results_dir}/")
        print(f"  Mean Performance: {results['summary']['mean_performance']:.4f} ± {results['summary']['std_performance']:.4f}")

    if args.mode == 'ablation':
        print("\n[Ablation Study]")
        # Ablation studies would be run here
        print("  Run with specific ablation configs...")


if __name__ == '__main__':
    main()