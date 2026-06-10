"""V6 — UnifiedTransformer (combined complete token, region + time, no separate task)."""
import gc
import torch
import optuna
from config import DATA_DIR, RESULTS_DIR
from core.models.UnifiedTransformer import UnifiedTransformer
from core.training import train_unified_taskregion

device = 'cuda' if torch.cuda.is_available() else 'cpu'

info = torch.load(DATA_DIR / 'prepared_data.pt', map_location=device, weights_only=False)
n = info['n']
data = {
    'train_complete': info['data_complete'][:n],
    'val_complete':   info['data_complete'][n:],
    'train_times':    info['data_times'][:n],
    'val_times':      info['data_times'][n:],
}
vocab_size_complete = info['vocab_size_complete']

def objective(trial, fixed_unified):
    config = {
        'block_size': trial.suggest_categorical('block_size', [64, 128, 256, 512]),
        'n_embd':     trial.suggest_categorical('n_embd', [64, 128, 256]),
        'n_head':     trial.suggest_categorical('n_head', [2, 4, 8]),
        'n_layer':    trial.suggest_categorical('n_layer', [1, 2, 4, 8]),
        'dropout':    trial.suggest_float('dropout', 0.1, 0.4),
        'lr':         trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        **fixed_unified,
    }
    model = UnifiedTransformer(
        vocab_size_region=vocab_size_complete,
        block_size=config['block_size'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        n_head=config['n_head'],
        n_layer=config['n_layer'],
        separated_task=False,
        predict_task=False,
    ).to(device)
    try:
        return train_unified_taskregion(model, data, config, device, trial=trial)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run(n_trials=50, fixed_unified=None):
    if fixed_unified is None:
        fixed_unified = {'max_iters': 1500, 'eval_iters': 100, 'eval_interval': 100}

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    storage = f'sqlite:///{RESULTS_DIR / "optuna.db"}'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v6_unified_taskregion',
        storage=storage, load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, fixed_unified), n_trials=n_trials)
    print(f"[V6 UnifiedTransformer] Best val_loss: {study.best_value:.4f} | Params: {study.best_params}")

    torch.save({
        'UnifiedTransformer': study.best_params,
    }, DATA_DIR / 'v6_best_params.pt')

    return study


if __name__ == '__main__':
    run()
