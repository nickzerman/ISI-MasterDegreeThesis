import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, block_size, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)    # Trasformano i bit delle regioni in vettori astratti per calcolare quanto ogni istante della traccia sia rilevante per prevedere il successivo
        self.query = nn.Linear(n_embd, head_size, bias=False)  # ""
        self.value = nn.Linear(n_embd, head_size, bias=False)  # ""
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #Matrice triangolare per studiare ogni passaggio della traccia passata

        self.dropout = nn.Dropout(dropout) # Per evitare l'overfitting, dimentica %dropout neuroni

    def forward(self, x):
        # Input di size (batch, time-step, channels)
        # Output di size (batch, time-step, head size)

        B, T, C = x.shape # batch, time_step, channels
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        # Affinities - Attention Scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, block_size, n_embd, dropout, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, block_size, head_size, n_embd, dropout) #Self attention heads part
        self.ffwd = FeedForward(n_embd, dropout) # Feed Forward part
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Add & Norm
        x = x + self.ffwd(self.ln2(x)) # Add & Norm
        return x

class UnifiedTransformer(nn.Module):
    def __init__(self, vocab_size_region, block_size, n_embd, dropout=0.2, n_head=1, n_layer=1, vocab_size_task=None, separated_task=False, predict_task=False):
        '''
        Args:
            vocab_size_region: region vocabulary
            vocab_size_task: task vocabulary (required if separated_task=True)
            block_size: sliding window size
            n_embd: number of embeddings
            dropout: dropout rate
            n_head: number of attention heads
            n_layer: number of layers
            separated_task: if True, task and region have separate embeddings
            predict_task: if True, predict task
        '''

        assert not separated_task or vocab_size_task is not None
        assert not predict_task or vocab_size_task is not None
        assert not predict_task or separated_task # Non posso dover predictare le task se non le ho separate rispetto alle regioni

        super().__init__()

        self.separated_task = separated_task
        self.predict_task = predict_task

        self.region_embedding_table = nn.Embedding(vocab_size_region, n_embd)
        self.time_proj = nn.Linear(1, n_embd)

        if self.separated_task:
            self.task_embedding_table = nn.Embedding(vocab_size_task, n_embd)

        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(block_size, n_embd, dropout, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        if self.predict_task:
            self.task_head = nn.Linear(n_embd, vocab_size_task)
            self.log_var_task = nn.Parameter(torch.zeros(1))

        self.region_head = nn.Linear(n_embd, vocab_size_region)
        self.time_head = nn.Linear(n_embd, 1)

        # Kendall uncertainty: parametri learnable per bilanciare le loss automaticamente
        # log_var = log(sigma^2), inizializzati a 0 (sigma=1, nessun peso iniziale)
        self.log_var_time = nn.Parameter(torch.zeros(1))
        self.log_var_region = nn.Parameter(torch.zeros(1))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_region, idx_times, idx_task=None, targets_task=None, targets_region=None, targets_time=None):
        B, T = idx_region.shape

        region_emb = self.region_embedding_table(idx_region)
        time_emb = self.time_proj(idx_times.unsqueeze(-1).float())
        pos_emb = self.positional_embedding(torch.arange(T, device=device))

        if self.separated_task:
            task_emb = self.task_embedding_table(idx_task)
            x = task_emb + region_emb + time_emb + pos_emb  # (B,T,C)
        else:
            x = region_emb + time_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)
        x = self.ln_f(x)

        time_pred = F.softplus(self.time_head(x))  # (B,T,1) — softplus evita tempi negativi

        if self.separated_task and self.predict_task:
            logits_task = self.task_head(x)      # (B,T,vocab_size_task)
            logits_region = self.region_head(x)  # (B,T,vocab_size_region)

            loss = None
            if targets_task is not None and targets_region is not None and targets_time is not None:
                B, T, C_task = logits_task.shape
                _, _, C_region = logits_region.shape
                loss_task = F.cross_entropy(logits_task.view(B * T, C_task), targets_task.view(B * T))
                loss_region = F.cross_entropy(logits_region.view(B * T, C_region), targets_region.view(B * T))
                loss_time = F.mse_loss(time_pred, targets_time.unsqueeze(-1).float())

                # Kendall uncertainty weighting (Kendall et al., 2018)
                # Classificazione: exp(-s) * L_ce + s
                # Regressione (likelihood gaussiana): 0.5 * exp(-s) * L_mse + 0.5 * s
                loss = (torch.exp(-self.log_var_task) * loss_task + self.log_var_task +
                        torch.exp(-self.log_var_region) * loss_region + self.log_var_region +
                        0.5 * torch.exp(-self.log_var_time) * loss_time + 0.5 * self.log_var_time)

            return logits_task, logits_region, time_pred, loss

        else:
            logits_region = self.region_head(x)  # (B,T,vocab_size_region)

            loss = None
            if targets_region is not None and targets_time is not None:
                B, T, C_region = logits_region.shape
                loss_region = F.cross_entropy(logits_region.view(B * T, C_region), targets_region.view(B * T))
                loss_time = F.mse_loss(time_pred, targets_time.unsqueeze(-1).float())

                loss = (torch.exp(-self.log_var_region) * loss_region + self.log_var_region +
                        0.5 * torch.exp(-self.log_var_time) * loss_time + 0.5 * self.log_var_time)

            return logits_region, time_pred, loss

    @torch.no_grad()
    def predict_next(self, idx_region, idx_times, block_size, idx_task=None):
        idx_region_cond = idx_region[:, -block_size:]
        idx_times_cond = idx_times[:, -block_size:]

        if self.separated_task and self.predict_task:
            idx_task_cond = idx_task[:, -block_size:]
            logits_task, logits_region, time_pred, _ = self(idx_region_cond, idx_times_cond, idx_task_cond)
            next_task = torch.multinomial(F.softmax(logits_task[:, -1, :], dim=-1), num_samples=1)    # (B,1)
            next_region = torch.multinomial(F.softmax(logits_region[:, -1, :], dim=-1), num_samples=1)  # (B,1)
        elif self.separated_task:
            idx_task_cond = idx_task[:, -block_size:]
            logits_region, time_pred, _ = self(idx_region_cond, idx_times_cond, idx_task_cond)
            next_task = None
            next_region = torch.multinomial(F.softmax(logits_region[:, -1, :], dim=-1), num_samples=1)  # (B,1)
        else:
            logits_region, time_pred, _ = self(idx_region_cond, idx_times_cond)
            next_task = None
            next_region = torch.multinomial(F.softmax(logits_region[:, -1, :], dim=-1), num_samples=1)  # (B,1)

        next_time = time_pred[:, -1, :]  # (B,1)

        return next_task, next_region, next_time

