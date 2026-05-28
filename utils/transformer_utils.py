import torch

def get_batch_model(split, train_data, val_data, batch_size, block_size, device):
    """
    Get Batch Function for REGIONE + TASK model or ONLY task model

    Parameters
    ----------
    split: 'train' or 'val'
    train_data
    val_data
    batch_size
    block_size
    device

    Returns
    ----------
    x,y: batch (x input, y output)

    """
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,)) # Prendo batch_size indici casuali

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x,y = x.to(device), y.to(device)
    return x, y

def get_batch_model_time(split, train_data, train_times, val_data, val_times, batch_size, block_size, device):
    """
    Get Batch Function for Time Transformer that uses REGIONE + TASK columns together
    In here we slides the time vector by 1 to the right

    Parameters
    ----------
    split: 'train' or 'val'
    train_data
    train_times
    val_data
    val_times
    batch_size
    block_size
    device

    Returns
    ----------
    x_data,x_times,y: batch (x_data input of REGIONE + TASK columns, x_times times slided by one, y output of column's correct times)

    """
    data = train_data if split == 'train' else val_data
    time = train_times if split == 'train' else val_times

    ix = torch.randint(1, len(data) - block_size, (batch_size,))

    x_data = torch.stack([data[i:i+block_size] for i in ix])
    x_times = torch.stack([time[i-1:i+block_size-1] for i in ix]) # Slido i tempi di 1 a destra (ho tutto sfalsato di 1) --> predicto quello dopo
    y = torch.stack([time[i:i+block_size] for i in ix])

    x_data, x_times, y = x_data.to(device), x_times.to(device), y.to(device)
    return x_data, x_times, y

def get_batch_model_time_v2(split, train_task, train_region, train_times, val_task, val_region, val_times, batch_size, block_size, device):
    """
    Get Batch Function for Time Transformer that uses separated REGIONE and TASK columns
    In here we slide the time vector by 1 to the right

    Parameters
    ----------
    split: 'train' or 'val'
    train_task
    train_region
    train_times
    val_task
    val_region
    val_times
    batch_size
    block_size
    device

    Returns
    ----------
    x_task,x_region,x_times,y: batch (x_task input of task column, x_region input of region column, x_times times slided by one, y output of column's correct times)

    """
    task = train_task if split == 'train' else val_task
    region = train_region if split == 'train' else val_region
    time = train_times if split == 'train' else val_times

    ix = torch.randint(1, len(task) - block_size, (batch_size,))

    x_task = torch.stack([task[i:i+block_size] for i in ix])
    x_region = torch.stack([region[i:i+block_size] for i in ix])
    x_times = torch.stack([time[i-1:i+block_size-1] for i in ix])
    y = torch.stack([time[i:i+block_size] for i in ix])

    x_task, x_region, x_times, y = x_task.to(device), x_region.to(device), x_times.to(device), y.to(device)
    return x_task, x_region, x_times, y

def get_batch_model_region(split, train_task, val_task, train_region, val_region, batch_size, block_size, device):
    """
    Get Batch Function for region model
    In here we use the task to predict region, we slide the region vector by 1 to the right

    Parameters
    ----------
    split: 'train' or 'val'
    train_task
    val_task
    train_region
    val_region
    batch_size
    block_size
    device

    Returns
    ----------
    x_task,x_region,y: batch (x_task input of task column, x_region input of region column, y output of correct column)

    """
    task = train_task if split == 'train' else val_task
    region = train_region if split == 'train' else val_region

    ix = torch.randint(1, len(task) - block_size, (batch_size,))

    x_task = torch.stack([task[i:i+block_size] for i in ix])
    x_region = torch.stack([region[i-1:i+block_size-1] for i in ix])
    y = torch.stack([region[i:i+block_size] for i in ix])

    x_task, x_region, y = x_task.to(device), x_region.to(device), y.to(device)
    return x_task, x_region, y

@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, batch_size, block_size, device):
    """
    Function that makes an estimation about loss of REGION + TASK model or ONLY task model

    Parameters
    ----------
    model
    eval_iters: how many batch function have to test before calculating losses in model.eval() mode
    train_data
    val_data
    batch_size
    block_size
    device

    Returns
    ----------
    out: loss for both train and val split
    """
    out = {}
    model.eval() # Metto il modello in modalità eval (non train)

    # Calcolo la loss su eval_iters batch
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_model(split, train_data, val_data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

@torch.no_grad()
def estimate_loss_region(model, eval_iters, train_task, train_region, val_task, val_region, batch_size, block_size, device):
    """
    Function that makes an estimation about loss of region model

    Parameters
    ----------
    model
    eval_iters: how many batch function have to test before calculating losses in model.eval() mode
    train_task
    train_region
    val_task
    val_region
    batch_size
    block_size
    device

    Returns
    ----------
    out: loss for both train and val split
    """
    out = {}
    model.eval() # Metto il modello in modalità eval (non train)

    # Calcolo la loss su eval_iters batch
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_task, X_region, Y = get_batch_model_region(split, train_task, val_task, train_region, val_region, batch_size, block_size, device)
            logits, loss = model(X_region, X_task, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

@torch.no_grad()
def estimate_loss_times(model, eval_iters, train_data, train_times, val_data, val_times, batch_size, block_size, device):
    """
    Function that makes an estimation about loss of times model that uses REGIONE + TASK columns together

    Parameters
    ----------
    model
    eval_iters: how many batch function have to test before calculating losses in model.eval() mode
    train_data
    train_times
    val_data
    val_times
    batch_size
    block_size
    device

    Returns
    ----------
    out: loss for both train and val split
    """
    out = {}
    model.eval() # Metto il modello in modalità eval (non train)

    # Calcolo la loss su eval_iters batch
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_data, X_times, Y = get_batch_model_time(split, train_data, train_times, val_data, val_times, batch_size, block_size, device)
            logits, loss = model(X_data, X_times, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

@torch.no_grad()
def estimate_loss_times_v2(model, eval_iters, train_task, train_region, train_times, val_task, val_region, val_times, batch_size, block_size, device):
    """
    Function that makes an estimation about loss of times model that uses separated REGIONE and TASK columns

    Parameters
    ----------
    model
    eval_iters: how many batch function have to test before calculating losses in model.eval() mode
    train_task
    train_region
    train_times
    val_task
    val_region
    val_times
    batch_size
    block_size
    device

    Returns
    ----------
    out: loss for both train and val split
    """
    out = {}
    model.eval() # Metto il modello in modalità eval (non train)

    # Calcolo la loss su eval_iters batch
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_task, X_region, X_times, Y = get_batch_model_time_v2(split, train_task, train_region, train_times, val_task, val_region, val_times, batch_size, block_size, device)
            logits, loss = model(X_task, X_times, X_region, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out