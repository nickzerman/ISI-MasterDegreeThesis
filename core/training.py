import optuna

from utils.transformer_utils import *

def train_task(model, data, param, device, data_key='complete', trial=None, printing=False):
    """Train TaskTransformer. data_key: 'complete' o 'tasks'."""
    train_data = data[f'train_{data_key}']
    val_data = data[f'val_{data_key}']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val = float('inf')

    for iter in range(param['max_iters']):
        xb, yb = get_batch_task('train', train_data, val_data, param['batch_size'], param['block_size'], device) # Pesco tracce
        logits, loss = model(xb, targets_task=yb) # Esegue il forward e predice
        optimizer.zero_grad(set_to_none=True) # Reset gradienti (puliamo i calcoli del giro precedente)
        loss.backward() #Errore per neurone
        optimizer.step() # Aggiorna automaticamente i pesi per sbagliare di meno al giro dopo

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1: # Ogni tot stampo la loss corrente
            losses = estimate_loss(model, param['eval_iters'], train_data, val_data, param['batch_size'], param['block_size'], device) # La stimo
            val_loss = losses['val'].item()

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            best_val = min(best_val, val_loss) # Salvo il best_val corrente

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_task_separated(model, data, param, device, trial=None, printing=False):
    """Train TaskTransformer con separated=True (dual head task + region)."""
    train_task_d = data['train_tasks']
    val_task_d = data['val_tasks']
    train_region_d = data['train_regions']
    val_region_d = data['val_regions']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val = float('inf')

    for iter in range(param['max_iters']):
        x_task, x_region, y_task, y_region = get_batch_task_separated(
            'train', train_task_d, val_task_d, train_region_d, val_region_d,
            param['batch_size'], param['block_size'], device
        )
        *_, loss = model(x_task, x_region, targets_task=y_task, targets_region=y_region)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_task_separated(
                model, param['eval_iters'],
                train_task_d, val_task_d, train_region_d, val_region_d,
                param['batch_size'], param['block_size'], device
            )
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_region(model, data, param, device, trial=None, printing=False):
    """Train RegionTransformer."""
    train_task_d = data['train_tasks']
    val_task_d = data['val_tasks']
    train_region_d = data['train_regions']
    val_region_d = data['val_regions']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val = float('inf')

    for iter in range(param['max_iters']):
        x_task, x_region, y = get_batch_region('train', train_task_d, val_task_d, train_region_d, val_region_d, param['batch_size'], param['block_size'], device)
        _, loss = model(x_region, x_task, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses   = estimate_loss_region(model, param['eval_iters'], train_task_d, train_region_d, val_task_d, val_region_d, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_time(model, data, param, device, trial=None, printing=False):
    """Train TimeTransformer (not separated, task+region together)."""
    train_data = data['train_complete']
    val_data = data['val_complete']
    train_times = data['train_times']
    val_times = data['val_times']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val = float('inf')

    for iter in range(param['max_iters']):
        xd, xt, y = get_batch_time('train', train_data, train_times, val_data, val_times, param['batch_size'], param['block_size'], device)
        _, loss = model(xd, xt, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_times(model, param['eval_iters'], train_data, train_times, val_data, val_times, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_time_v2(model, data, param, device, trial=None, printing=False):
    """Train TimeTransformer (separated regions)."""
    train_task_d = data['train_tasks']
    val_task_d = data['val_tasks']
    train_region_d = data['train_regions']
    val_region_d = data['val_regions']
    train_times = data['train_times']
    val_times = data['val_times']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val = float('inf')

    for iter in range(param['max_iters']):
        xta, xr, xt, y = get_batch_time_v2('train', train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
        _, loss = model(xta, xt, xr, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_times_v2(model, param['eval_iters'], train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_unified(model, data, param, device, trial=None, printing=False):
    """Train UnifiedTransformer (Kendall loss interna al modello)."""
    train_task_d = data['train_tasks']
    val_task_d = data['val_tasks']
    train_region_d = data['train_regions']
    val_region_d = data['val_regions']
    train_times = data['train_times']
    val_times = data['val_times']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val  = float('inf')

    for iter in range(param['max_iters']):
        x_task, x_region, x_times, y_task, y_region, y_times = get_batch_unified('train', train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
        *_, loss = model(x_region, x_times, idx_task=x_task, targets_task=y_task, targets_region=y_region, targets_time=y_times)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_unified(model, param['eval_iters'], train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_unified_onlyregion(model, data, param, device, trial=None, printing=False):
    """Train UnifiedTransformer (Kendall loss interna al modello)."""
    train_task_d = data['train_tasks']
    val_task_d = data['val_tasks']
    train_region_d = data['train_regions']
    val_region_d = data['val_regions']
    train_times = data['train_times']
    val_times = data['val_times']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val  = float('inf')

    for iter in range(param['max_iters']):
        x_task, x_region, x_times, y_region, y_times = get_batch_unified_onlyregion('train', train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
        *_, loss = model(x_region, x_times, idx_task=x_task, targets_region=y_region, targets_time=y_times)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_unified_onlyregion(model, param['eval_iters'], train_task_d, train_region_d, train_times, val_task_d, val_region_d, val_times, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val

def train_unified_taskregion(model, data, param, device, trial=None, printing=False):
    """Train UnifiedTransformer (Kendall loss interna al modello)."""
    train_complete = data['train_complete']
    val_complete = data['val_complete']
    train_times = data['train_times']
    val_times = data['val_times']
    optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param.get('weight_decay', 0.01))
    best_val  = float('inf')

    for iter in range(param['max_iters']):
        x_data, x_times, y_data, y_times = get_batch_unified_taskregion('train', train_complete, train_times, val_complete, val_times, param['batch_size'], param['block_size'], device)
        *_, loss = model(x_data, x_times, targets_region=y_data, targets_time=y_times)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % param['eval_interval'] == 0 or iter == param['max_iters'] - 1:
            losses = estimate_loss_unified_taskregion(model, param['eval_iters'], train_complete, train_times, val_complete, val_times, param['batch_size'], param['block_size'], device)
            val_loss = losses['val'].item()
            best_val = min(best_val, val_loss)

            if printing:
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if trial is not None:
                trial.report(val_loss, iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return best_val