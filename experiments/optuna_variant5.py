"""V5 — TaskTransformer (task only) + UnifiedTransformer (region + time, no task prediction)."""
import gc
import torch
import optuna
from config import DATA_DIR, RESULTS_DIR
from core.models import TaskTransformer
from core.models.UnifiedTransformer import UnifiedTransformer
from core.training import train_task, train_unified_onlyregion

device = 'cuda' if torch.cuda.is_available() else 'cpu'

info = torch.load(DATA_DIR / 'prepared_data.pt', map_location=device, weights_only=False)
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

FIXED_TASK    = {'max_iters': 1000, 'eval_iters': 100, 'eval_interval': 100}
FIXED_UNIFIED = {'max_iters': 1000, 'eval_iters': 100, 'eval_interval': 100}


def objective_task(trial):
    config = {
        'block_size': trial.suggest_categorical('block_size', [64, 128, 256]),
        'n_embd':     trial.suggest_categorical('n_embd', [64, 128, 256]),
        'n_head':     trial.suggest_categorical('n_head', [2, 4, 8]),
        'n_layer':    trial.suggest_categorical('n_layer', [1, 2, 4, 8]),
        'dropout':    trial.suggest_float('dropout', 0.2, 0.4),
        'lr':         trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        **FIXED_TASK,
    }
    model = TaskTransformer(
        task_vocab_size=vocab_size_tasks,
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
    ).to(device)
    try:
        return train_task(model, data, config, device, data_key='tasks', trial=trial)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def objective_unified(trial):
    config = {
        'block_size': trial.suggest_categorical('block_size', [64, 128, 256, 512]),
        'n_embd':     trial.suggest_categorical('n_embd', [64, 128, 256]),
        'n_head':     trial.suggest_categorical('n_head', [2, 4, 8]),
        'n_layer':    trial.suggest_categorical('n_layer', [1, 2, 4, 8]),
        'dropout':    trial.suggest_float('dropout', 0.1, 0.4),
        'lr':         trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        **FIXED_UNIFIED,
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
        predict_task=False,
    ).to(device)
    try:
        return train_unified_onlyregion(model, data, config, device, trial=trial)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run(n_trials_task=50, n_trials_unified=50):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    storage = f'sqlite:///{RESULTS_DIR / "optuna.db"}'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    study_task = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v5_task',
        storage=storage, load_if_exists=True,
    )
    study_task.optimize(objective_task, n_trials=n_trials_task)
    print(f"[V5 TaskTransformer] Best val_loss: {study_task.best_value:.4f} | Params: {study_task.best_params}")

    study_unified = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v5_unified_onlyregion',
        storage=storage, load_if_exists=True,
    )
    study_unified.optimize(objective_unified, n_trials=n_trials_unified)
    print(f"[V5 UnifiedTransformer] Best val_loss: {study_unified.best_value:.4f} | Params: {study_unified.best_params}")

    torch.save({
        'TaskTransformer':    study_task.best_params,
        'UnifiedTransformer': study_unified.best_params,
    }, DATA_DIR / 'v5_best_params.pt')

    return study_task, study_unified


if __name__ == '__main__':
    run()
