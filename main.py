import argparse
import yaml
import os
import sys


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
        from model.qmamba import QMamba, QEnsemble
        from model.trainer import QMTrainer, AdaptiveCQLTrainer, EnsembleAdaptiveCQLTrainer, TrainingConfig
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
        print("\n[1/3] Building E&E dataset...")
        bbob_suite = BBOBSuite(
            dim=config.get('dataset', {}).get('dim', 5),
            n_train_instances=config.get('dataset', {}).get('train_instances', 1),
            n_test_instances=config.get('dataset', {}).get('test_instances', 1),
            seed=config.get('dataset', {}).get('seed', 42)
        )

        K = config.get('state_action', {}).get('K', 3)
        # Override K based on algorithm type (Alg0=3, Alg1=10, Alg2=16)
        alg_K_map = {'Alg0': 3, 'Alg1': 10, 'Alg2': 16}
        K = alg_K_map.get(alg_type, K)
        M = config.get('state_action', {}).get('M', 16)
        pop_size = config.get('algorithm', {}).get('pop_size', 20)
        use_lpsr = config.get('algorithm', {}).get('use_lpsr', True)
        min_pop_size = config.get('algorithm', {}).get('min_pop_size', 4)

        # Dataset path includes algorithm type for separate datasets per algorithm
        base_dataset_path = config.get('paths', {}).get('dataset_path', './data/ee_dataset.pkl')
        dataset_path = base_dataset_path.replace('.pkl', f'_{alg_type}.pkl')

        seed = config.get('dataset', {}).get('seed', 42)

        builder = EEDatasetBuilder(
            bbob_suite=bbob_suite,
            optimizer_class=OptimizerClass,
            K=K,
            M=M,
            pop_size=pop_size,
            mu=config.get('dataset', {}).get('mu', 0.5),
            T=config.get('dataset', {}).get('trajectory_length', 100),
            seed=seed,
            use_lpsr=use_lpsr,
            min_pop_size=min_pop_size
        )

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

        # Load existing dataset if available
        if os.path.exists(dataset_path):
            print(f"  Loading existing dataset for {alg_type} from {dataset_path}")
            train_trajs, val_trajs, dataset_config = EEDatasetBuilder.load_dataset(dataset_path)

            # Validate algorithm matches
            dataset_alg = dataset_config.get('algorithm', '')
            # Extract Alg0/Alg1/Alg2 from full class name like "Alg0Optimizer"
            dataset_alg_type = dataset_alg.replace('Optimizer', '') if dataset_alg else ''

            if dataset_alg_type != alg_type:
                print(f"  [WARNING] Dataset algorithm ({dataset_alg_type}) does not match config ({alg_type})")
                print(f"  [WARNING] Dataset will be rebuilt for {alg_type}...")
                print(f"  To use existing dataset, change algorithm.type in config to '{dataset_alg_type}'")
                train_trajs, val_trajs = builder.build(
                    n_total=config.get('dataset', {}).get('n_total_trajectories', 10000),
                    save_path=dataset_path
                )
            else:
                print(f"  Dataset validated: algorithm={dataset_alg_type}, dim={dataset_config.get('dim')}, K={dataset_config.get('K')}")
        else:
            print(f"  Building new dataset for {alg_type}...")
            train_trajs, val_trajs = builder.build(
                n_total=config.get('dataset', {}).get('n_total_trajectories', 10000),
                save_path=dataset_path,
                n_workers=config.get('dataset', {}).get('n_workers', None),
            )

        # Create data loaders
        T_max = config.get('dataset', {}).get('trajectory_length', 100)
        train_loader = MetaDataLoader(
            train_trajs,
            batch_size=config.get('training', {}).get('batch_size', 32),
            K=K,
            T_max=T_max
        )
        val_loader = MetaDataLoader(
            val_trajs,
            batch_size=config.get('training', {}).get('batch_size', 32),
            K=K,
            T_max=T_max
        )

        # Create model
        print("\n[2/3] Creating Q-Mamba model...")
        state_dim = config.get('state_action', {}).get('state_dim', 9)
        d_model = config.get('model', {}).get('d_model', 14)
        d_state = config.get('model', {}).get('d_state', 32)
        n_layers = config.get('model', {}).get('n_layers', 1)
        num_hidden_mlp = config.get('model', {}).get('num_hidden_mlp', 32)

        model = QMamba(
            state_dim=state_dim,
            K=K,
            M=M,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            num_hidden_mlp=num_hidden_mlp
        )

        print(f"  Model parameters: {model.num_parameters:,}")
        print(f"  Backend: {'Mamba' if model.uses_mamba else 'GRU (fallback)'}")

        # Train
        print("\n[3/3] Training...")
        train_config = TrainingConfig(
            lr=config.get('training', {}).get('lr', 0.001),
            gamma=config.get('training', {}).get('gamma', 0.99),
            beta=config.get('training', {}).get('beta', 10.0),
            lam=config.get('training', {}).get('lam', 1.0),
            batch_size=config.get('training', {}).get('batch_size', 32),
            n_epochs=config.get('training', {}).get('n_epochs', 100),
            grad_clip=config.get('training', {}).get('grad_clip', 100.0),
            weight_decay=config.get('training', {}).get('weight_decay', 1e-4),
            device=device,
            save_dir=config.get('paths', {}).get('checkpoint_dir', './checkpoints'),
            eval_interval=config.get('training', {}).get('eval_interval', 10),
            checkpoint_interval=config.get('training', {}).get('checkpoint_interval', 50),
            scheduler=config.get('training', {}).get('scheduler', 'none'),
            algorithm=alg_type,
            print_every=config.get('training', {}).get('print_every', 1),
            seed=seed
        )

        # Get adaptive CQL parameters from config
        adaptive_cql_config = config.get('training', {}).get('adaptive_cql', {})
        ensemble_config = config.get('training', {}).get('ensemble', {})
        use_ensemble = ensemble_config.get('enabled', False)
        use_adaptive_cql = adaptive_cql_config.get('enabled', False)
        lam_init = adaptive_cql_config.get('lam_init', config.get('training', {}).get('lam', 1.0))
        lam_min = adaptive_cql_config.get('lam_min', 0.01)
        lam_max = adaptive_cql_config.get('lam_max', 2.0)
        dropout_p = adaptive_cql_config.get('dropout_p', 0.1)
        uncertainty_samples = adaptive_cql_config.get('uncertainty_samples', 8)
        uncertainty_interval = adaptive_cql_config.get('uncertainty_interval', 10)

        if use_ensemble:
            # --- Ensemble training ---
            n_members = ensemble_config.get('n_members', 5)
            div_weight = ensemble_config.get('diversity_weight', 0.1)
            div_type = ensemble_config.get('diversity_type', 'mi')
            base_seed = ensemble_config.get('base_seed', 42)

            print(f"\n[2/3] Creating Q-Ensemble model ({n_members} members)...")
            model = QEnsemble(
                n_members=n_members,
                state_dim=state_dim,
                K=K,
                M=M,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                num_hidden_mlp=num_hidden_mlp,
                force_cpu=(device == 'cpu'),
                base_seed=base_seed,
            )
            print(f"  Total parameters: {model.num_parameters:,}")
            print(f"  Backend: {'Mamba' if model.uses_mamba else 'GRU (fallback)'}")

            print(f"\n[3/3] Training with Ensemble Adaptive CQL...")
            print(f"  {n_members} members, diversity={div_type} (weight={div_weight})")
            print(f"  λ per-member adaptive: [{lam_min}, {lam_max}], init={lam_init}")

            trainer = EnsembleAdaptiveCQLTrainer(
                model,
                train_config,
                device,
                lam_init=lam_init,
                lam_min=lam_min,
                lam_max=lam_max,
                diversity_weight=div_weight,
                diversity_type=div_type,
            )
        elif use_adaptive_cql:
            # --- Single-model adaptive CQL ---
            print(f"\n[2/3] Creating Q-Mamba model...")
            model = QMamba(
                state_dim=state_dim,
                K=K,
                M=M,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                num_hidden_mlp=num_hidden_mlp
            )
            print(f"  Model parameters: {model.num_parameters:,}")
            print(f"  Backend: {'Mamba' if model.uses_mamba else 'GRU (fallback)'}")

            print(f"\n[3/3] Training with Adaptive CQL...")
            print(f"  λ∈[{lam_min}, {lam_max}], init={lam_init}")

            trainer = AdaptiveCQLTrainer(
                model,
                train_config,
                device,
                lam_init=lam_init,
                lam_min=lam_min,
                lam_max=lam_max,
                dropout_p=dropout_p,
                uncertainty_samples=uncertainty_samples,
                uncertainty_interval=uncertainty_interval
            )
        else:
            # --- Standard CQL ---
            print(f"\n[2/3] Creating Q-Mamba model...")
            model = QMamba(
                state_dim=state_dim,
                K=K,
                M=M,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                num_hidden_mlp=num_hidden_mlp
            )
            print(f"  Model parameters: {model.num_parameters:,}")
            print(f"  Backend: {'Mamba' if model.uses_mamba else 'GRU (fallback)'}")

            print(f"\n[3/3] Training with standard CQL...")
            trainer = QMTrainer(model, train_config, device)
        trainer.fit(train_loader, val_loader, verbose=True)

    if args.mode == 'ablation':
        print("\n[Ablation Study]")
        # Ablation studies would be run here
        print("  Run with specific ablation configs...")


if __name__ == '__main__':
    main()
