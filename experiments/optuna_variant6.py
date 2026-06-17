"""V6 — UnifiedTransformer (combined complete token, region + time, no separate task)."""
import gc
import torch
import optuna
from config import DATA_DIR, RESULTS_DIR, SEED
from utils import set_seed
from core.models.UnifiedTransformer import UnifiedTransformer
from core.training import train_unified_taskregion
from utils import get_optuna_search_space, get_fixed_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_FILE = DATA_DIR / 'prepared_data.pt'


def run(n_trials=50, fixed_unified=None, data_file=None):
    if fixed_unified is None:
        fixed_unified = {'max_iters': 1500, 'eval_iters': 100, 'eval_interval': 100}

    set_seed(SEED)
    info = torch.load(DATA_DIR / data_file if data_file else DATA_FILE, map_location=device, weights_only=False)
    n = info['n']
    data = {
        'train_complete': info['data_complete'][:n],
        'val_complete':   info['data_complete'][n:],
        'train_times':    info['data_times'][:n],
        'val_times':      info['data_times'][n:],
    }
    vocab_size_complete = info['vocab_size_complete']
    max_len = info.get('max_trace_len', 64)
    if fixed_unified is None:
        fixed_unified = get_fixed_params(max_len, model_type='unified')

    def objective(trial, fixed_unified):
        sp = get_optuna_search_space(max_len, model_type='unified')
        config = {
            'block_size':   trial.suggest_categorical('block_size', sp['block_size']),
            'n_embd':       trial.suggest_categorical('n_embd', sp['n_embd']),
            'n_head':       trial.suggest_categorical('n_head', sp['n_head']),
            'n_layer':      trial.suggest_categorical('n_layer', sp['n_layer']),
            'dropout':      trial.suggest_float('dropout', *sp['dropout_range']),
            'lr':           trial.suggest_float('lr', *sp['lr_range'], log=True),
            'weight_decay': trial.suggest_float('weight_decay', *sp['wd_range'], log=True),
            'batch_size':   trial.suggest_categorical('batch_size', sp['batch_size']),
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

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    storage = f'sqlite:///{RESULTS_DIR / "optuna.db"}'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction='minimize', sampler=sampler, pruner=pruner, study_name='v6_unified_taskregion',
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
