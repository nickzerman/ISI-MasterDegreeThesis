"""V1 — TaskTransformer (complete token) + TimeTransformer (not separated)."""
import gc
import torch
import optuna
from config import DATA_DIR, RESULTS_DIR
from core.models import TaskTransformer, TimeTransformer
from core.training import train_task, train_time
from utils import get_optuna_search_space, get_fixed_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_FILE = DATA_DIR / 'prepared_data.pt'


def run(n_trials_task=50, n_trials_time=50, fixed_task=None, fixed_time=None, data_file=None):
    if fixed_task is None:
        fixed_task = {'max_iters': 1000, 'eval_iters': 100, 'eval_interval': 100}
    if fixed_time is None:
        fixed_time = {'max_iters': 500, 'eval_iters': 100, 'eval_interval': 100}

    info = torch.load(DATA_DIR / data_file if data_file else DATA_FILE, map_location=device, weights_only=False)
    n = info['n']
    data = {
        'train_complete': info['data_complete'][:n],
        'val_complete': info['data_complete'][n:],
        'train_times': info['data_times'][:n],
        'val_times': info['data_times'][n:],
    }
    vocab_size = info['vocab_size_complete']
    max_len = info.get('max_trace_len', 64)
    if fixed_task is None:
        fixed_task = get_fixed_params(max_len, model_type='task')
    if fixed_time is None:
        fixed_time = get_fixed_params(max_len, model_type='time')

    def objective_task(trial, fixed_task):
        sp = get_optuna_search_space(max_len, model_type='task')
        config = {
            'block_size':   trial.suggest_categorical('block_size', sp['block_size']),
            'n_embd':       trial.suggest_categorical('n_embd', sp['n_embd']),
            'n_head':       trial.suggest_categorical('n_head', sp['n_head']),
            'n_layer':      trial.suggest_categorical('n_layer', sp['n_layer']),
            'dropout':      trial.suggest_float('dropout', *sp['dropout_range']),
            'lr':           trial.suggest_float('lr', *sp['lr_range'], log=True),
            'weight_decay': trial.suggest_float('weight_decay', *sp['wd_range'], log=True),
            'batch_size':   trial.suggest_categorical('batch_size', sp['batch_size']),
            **fixed_task,
        }
        model = TaskTransformer(
            task_vocab_size=vocab_size,
            block_size=config['block_size'],
            n_embd=config['n_embd'],
            dropout=config['dropout'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
        ).to(device)
        try:
            return train_task(model, data, config, device, data_key='complete', trial=trial)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def objective_time(trial, fixed_time):
        sp = get_optuna_search_space(max_len, model_type='time')
        config = {
            'block_size':   trial.suggest_categorical('block_size', sp['block_size']),
            'n_embd':       trial.suggest_categorical('n_embd', sp['n_embd']),
            'n_head':       trial.suggest_categorical('n_head', sp['n_head']),
            'n_layer':      trial.suggest_categorical('n_layer', sp['n_layer']),
            'dropout':      trial.suggest_float('dropout', *sp['dropout_range']),
            'lr':           trial.suggest_float('lr', *sp['lr_range'], log=True),
            'weight_decay': trial.suggest_float('weight_decay', *sp['wd_range'], log=True),
            'batch_size':   trial.suggest_categorical('batch_size', sp['batch_size']),
            **fixed_time,
        }
        model = TimeTransformer(
            vocab_size_task=vocab_size,
            block_size=config['block_size'],
            n_embd=config['n_embd'],
            dropout=config['dropout'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            separated_regions=False,
        ).to(device)
        try:
            return train_time(model, data, config, device, trial=trial)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    storage = f'sqlite:///{RESULTS_DIR / "optuna.db"}'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    study_task = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v1_task_complete',
        storage=storage, load_if_exists=True,
    )
    study_task.optimize(lambda trial: objective_task(trial, fixed_task), n_trials=n_trials_task)
    print(f"[V1 TaskTransformer] Best val_loss: {study_task.best_value:.4f} | Params: {study_task.best_params}")

    study_time = optuna.create_study(
        direction='minimize', pruner=pruner, study_name='v1_time',
        storage=storage, load_if_exists=True,
    )
    study_time.optimize(lambda trial: objective_time(trial, fixed_time), n_trials=n_trials_time)
    print(f"[V1 TimeTransformer] Best val_loss: {study_time.best_value:.4f} | Params: {study_time.best_params}")

    torch.save({
        'TaskTransformer': study_task.best_params,
        'TimeTransformer': study_time.best_params,
    }, DATA_DIR / 'v1_best_params.pt')

    return study_task, study_time


if __name__ == '__main__':
    run()
