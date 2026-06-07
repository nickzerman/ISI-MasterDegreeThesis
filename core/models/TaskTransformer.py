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

class TaskTransformer(nn.Module):
    def __init__(self, task_vocab_size, block_size, n_embd, dropout=0.2, n_head=1, n_layer=1, region_vocab_size=None, separated=False):
        '''
        Args:
            task_vocab_size: task vocabulary
            region_vocab_size: region vocabulary (it could be None in some variants)
            block_size: sliding window size
            n_embd: number of embeddings
            dropout: dropout rate
            n_head: number of attention heads
            n_layer: number of layers
            separated: in some variants region and task could be separated
        '''

        assert not separated or region_vocab_size is not None

        super().__init__()
        self.separated = separated

        #self.token_projection = nn.Linear(num_bits, n_embd)
        self.task_embedding_table = nn.Embedding(task_vocab_size, n_embd)

        if self.separated:
            self.region_embedding_table = nn.Embedding(region_vocab_size, n_embd)

        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(block_size, n_embd,dropout, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm

        if self.separated:
            self.task_head = nn.Linear(n_embd, task_vocab_size)
            self.region_head = nn.Linear(n_embd, region_vocab_size)
        else:
            self.lm_head = nn.Linear(n_embd, task_vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_task, idx_region=None, targets_task=None, targets_region=None):
        B, T = idx_task.shape

        task_emb = self.task_embedding_table(idx_task)
        pos_emb = self.positional_embedding(torch.arange(T, device=device))  # (T,C)
        if self.separated:
            region_emb = self.region_embedding_table(idx_region)
            x = task_emb + region_emb + pos_emb  # (B,T,C)
        else:
            x = task_emb + pos_emb
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)

        if self.separated:
            task_logits = self.task_head(x)    # (B,T,task_vocab_size)
            region_logits = self.region_head(x)  # (B,T,region_vocab_size)

            loss = None
            if targets_task is not None and targets_region is not None:
                task_loss   = F.cross_entropy(task_logits.view(B * T, -1),   targets_task.view(B * T))
                region_loss = F.cross_entropy(region_logits.view(B * T, -1), targets_region.view(B * T))
                loss = task_loss + region_loss
            return task_logits, region_logits, loss
        else:
            logits = self.lm_head(x)  # (B,T,task_vocab_size)

            loss = None
            if targets_task is not None:
                loss = F.cross_entropy(logits.view(B * T, -1), targets_task.view(B * T))

            return logits, loss

    @torch.no_grad()
    def predict_next(self, idx_task, idx_region, block_size):
        """Solo per separated=True. Ritorna (idx_task_next, idx_region_next) in un'unica forward pass."""
        # Taglia per la finestra temporale
        idx_tasks_cond = idx_task[:, -block_size:]
        idx_regions_cond = idx_region[:, -block_size:]

        # Chiede al modello le probabilità (prende solo l'ultimo step)
        task_logits, region_logits, _ = self(idx_tasks_cond, idx_regions_cond)

        task_probs = F.softmax(task_logits[:, -1, :],   dim=-1)
        region_probs = F.softmax(region_logits[:, -1, :], dim=-1)

        ids_task_next = torch.multinomial(task_probs, num_samples=1)
        ids_region_next = torch.multinomial(region_probs, num_samples=1)
        return ids_task_next, ids_region_next

    @torch.no_grad()
    def predict_next_task(self, idx_task, block_size):
        """Solo per separated=False."""
        idx_tasks_cond = idx_task[:, -block_size:]
        logits, _ = self(idx_tasks_cond)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        ids_next = torch.multinomial(probs, num_samples=1)
        return ids_next