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

class TimeTransformer(nn.Module):
    def __init__(self, vocab_size_task, block_size, n_embd, dropout=0.2, n_head=1, n_layer=1, vocab_size_region=None, separated_regions=True):
        '''
        Args:
            vocab_size_task: task or task+region vocabulary
            vocab_size_region: region vocabulary
            block_size: sliding window size
            n_embd: number of embeddings
            dropout: dropout rate
            n_head: number of attention heads
            n_layer: number of layers
        '''

        super().__init__()

        self.separated_regions = separated_regions

        self.task_embedding_table = nn.Embedding(vocab_size_task, n_embd)  # Embedding per i token regione/task
        self.time_proj = nn.Linear(1, n_embd)  # Linear per il tempo
        if self.separated_regions:
            self.region_embedding_table = nn.Embedding(vocab_size_region, n_embd)

        self.positional_embedding = nn.Embedding(block_size, n_embd) # Classico Positional Embedding
        self.blocks = nn.Sequential(*[Block(block_size, n_embd,dropout, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.time_head = nn.Linear(n_embd, 1) # Da n_embd a previsione

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_tasks, idx_times, idx_region=None, targets=None):
        B, T = idx_tasks.size()

        # idx and targets are both (B,T) tensor of integers
        task_emb = self.task_embedding_table(idx_tasks)
        time_emb = self.time_proj(idx_times.unsqueeze(-1).float()) #unqueeze prende come riferimento la dimensione che inserisci
        pos_emb = self.positional_embedding(torch.arange(T, device=device))  # (T,C)

        if self.separated_regions:
            region_emb = self.region_embedding_table(idx_region)
            x = task_emb + region_emb + time_emb + pos_emb  # (B,T,C)
        else:
            x = task_emb + time_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        time_pred = self.time_head(x)  # (B,T,vocab_size)

        time_pred = F.softplus(time_pred) # Freno a mano per evitare tempi negativi

        if targets is None:
            loss = None
        else:
            targets = targets.unsqueeze(-1).float()
            loss = F.mse_loss(time_pred, targets)

        return time_pred, loss

    @torch.no_grad()
    def predict_next_time(self, idx_tasks, idx_times, block_size, idx_regions=None):
        idx_tasks_cond = idx_tasks[:, -block_size:] # Per prendere grandezza corretta (dipende dalla block_size)
        idx_times_cond = idx_times[:, -block_size:]

        if self.separated_regions:
            idx_regions_cond = idx_regions[:, -block_size:]
            time_pred, _ = self(idx_tasks_cond, idx_times_cond, idx_regions_cond)
        else:
            time_pred, _ = self(idx_tasks_cond, idx_times_cond)

        next_time = time_pred[:, -1, :]  # Forma: (Batch, 1)

        return next_time  # Ritorna SOLO il tempo in float

