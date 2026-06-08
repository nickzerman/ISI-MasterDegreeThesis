"""V4 — UnifiedTransformer (task + region + time, Kendall loss)."""
import os
import torch
import optuna
from core.models.UnifiedTransformer import UnifiedTransformer
from core.training import train_unified

device = 'cuda' if torch.cuda.is_available() else 'cpu'

info = torch.load('data/prepared_data.pt', map_location=device, weights_only=False)
n = info['n']
data = {
    'train_tasks':   info['data_tasks'][:n],
    'val_tasks':     info['data_tasks'][n:],
    'train_regions': info['data_regions'][:n],
    'val_regions':   info['data_regions'][n:],
    'train_times':   info['data_times'][:n],
    'val_times':     info['data_times'][n:],
}
vocab_size_tasks   = info['vocab_size_tasks']
vocab_size_regions = info['vocab_size_regions']

FIXED = {'max_iters': 500, 'eval_iters': 100, 'eval_interval': 100}


def objective(trial):
    config = {
        'block_size': trial.suggest_categorical('block_size', [64, 128, 256, 512]),
        'n_embd':     trial.suggest_categorical('n_embd', [64, 128, 256]),
        'n_head':     trial.suggest_categorical('n_head', [2, 4, 8]),
        'n_layer':    trial.suggest_categorical('n_layer', [1, 2, 4, 8]),
        'dropout':    trial.suggest_float('dropout', 0.1, 0.4),
        'lr':         trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        **FIXED,
    }
    model = UnifiedTransformer(
        vocab_size_region=vocab_size_regions,
        vocab_size_task=vocab_size_tasks,
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        separated_task=True,
        predict_task=True,
    ).to(device)
    return train_unified(model, data, config, device, trial=trial)


def run(n_trials=50):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    storage = 'sqlite:///results/optuna.db'
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    study = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v4_unified',
        storage=storage, load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    print(f"[V4 UnifiedTransformer] Best val_loss: {study.best_value:.4f} | Params: {study.best_params}")

    torch.save({
        'UnifiedTransformer': study.best_params,
    }, '../data/v4_best_params.pt')

    return study


if __name__ == '__main__':
    run()
