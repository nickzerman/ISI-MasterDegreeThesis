import re
import random
import numpy as np
import torch
from lark import Tree
from pm4py.objects.process_tree.obj import Operator


def set_seed(seed: int) -> None:
    """Imposta il seed globale per riproducibilità su CPU e GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def createNAryTree(tree):
    """
        From binary tree to n-ary tree

        Parameters
        ----------
        tree
            binary tree

        Returns
        ----------
        tree
            n-ary tree
    """
    if tree.data == 'task': #CASO BASE: se il tree è una task torno direttamente la task
        return tree

    children_processed = [createNAryTree(child) for child in tree.children] #Processo ogni figlio del nodo (saranno al massimo due visto che l'albero in input è nario)

    children = []
    for i,child in enumerate(children_processed): #Ciclo i figli processati
        #if child.data == tree.data and not(child.data=="loop" and i!=0): #Se il tipo di nodo è lo stesso elimino il nodo e copio i figli
        if child.data == tree.data:
            children.extend(child.children)
        else: #Altrimenti copio tutto il sottoalbero col tipo di nodo diverso
            children.append(child)

    return Tree(tree.data, children) #Ritorno l'albero n-ario

# Mapping attività → codice T01, T02, ...
# Le attività che iniziano con 'R' (Release, Return) colliderebbero con i nomi delle regioni
# (R0, R1, ...) in PetriNetP, che distingue regioni da task tramite il primo carattere.
def get_activities_from_tree(node):
    if node.operator is None:
        return [node.label] if node.label is not None else []
    return [a for child in node.children for a in get_activities_from_tree(child)]

# Conversione ProcessTree (pm4py) → stringa SESE del progetto
def process_tree_to_sese(node, mapping):
    """
    Converte ricorsivamente un ProcessTree pm4py nella stringa SESE.
    I nodi tau (silenziosi) vengono ignorati.
    Nel loop pm4py *(body, redo), si prende solo il corpo (primo figlio).
    Il wrapping tra parentesi garantisce il parsing corretto della grammatica.
    """
    if node.operator is None:
        if node.label is None:
            return None  # nodo tau: ignorato
        return mapping[node.label]

    children_sese = [
        s for child in node.children
        if (s := process_tree_to_sese(child, mapping)) is not None
    ]

    if not children_sese:
        return None
    if len(children_sese) == 1:
        return f'->({children_sese[0]})' if node.operator == Operator.LOOP else children_sese[0]

    def wrap(s):
        return s if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', s) else f'({s})'

    if node.operator == Operator.SEQUENCE:
        return ','.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.XOR:
        return '^'.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.PARALLEL:
        return '||'.join(wrap(c) for c in children_sese)
    elif node.operator == Operator.LOOP:
        return f'->{children_sese[0]}'
    return None


def get_optuna_search_space(max_len: int, model_type: str = 'task') -> dict:
    """
    Restituisce lo spazio di ricerca adattivo per Optuna basato sulla
    lunghezza massima delle tracce nel dataset corrente.

    Soglie (5 livelli):
        max_len <  10  → "tiny"   (processi elementari, sequenze cortissime)
        max_len <  24  → "small"  (XOR/SEQ semplici, pochi task)
        max_len <  50  → "medium" (strutture miste moderate)
        max_len < 100  → "large"  (loop/paralleli moderati)
        max_len >= 100 → "xlarge" (loop annidati, paralleli profondi)

    Parameters
    ----------
    max_len : int
        Lunghezza massima delle tracce nel dataset (in token).
    model_type : str
        Tipo di modello: 'task', 'time', 'region', 'unified'.

    Returns
    -------
    dict con chiavi 'block_size', 'n_embd', 'n_head', 'n_layer',
    'dropout_range', 'lr_range', 'wd_range', 'batch_size'.
    """
    if max_len < 6:
        tier = 'micro'
    elif max_len < 10:
        tier = 'tiny'
    elif max_len < 24:
        tier = 'small'
    elif max_len < 50:
        tier = 'medium'
    elif max_len < 100:
        tier = 'large'
    else:
        tier = 'xlarge'

    # --- Block size ---
    block_sizes = {
        'micro':  [2,  4,  8],
        'tiny':   [4,  8,  16],
        'small':  [8,  16, 32],
        'medium': [16, 32, 64],
        'large':  [32, 64, 128],
        'xlarge': [64, 128, 256],
    }[tier]

    # --- Architettura: embd, heads, layers ---
    # n_embd cresce con la complessità; per micro/tiny è volutamente piccolo
    # per evitare overfitting su pattern semplici.
    # n_head: tutti i valori devono dividere il minimo di n_embd del tier.
    n_embd = {
        'micro':  [4,  8,  16],   # head_size min = 4/2 = 2
        'tiny':   [8,  16, 32],   # head_size min = 8/4 = 2
        'small':  [16, 32, 64],   # head_size min = 16/4 = 4
        'medium': [32, 64, 128],  # head_size min = 32/8 = 4
        'large':  [64, 128, 256], # head_size min = 64/8 = 8
        'xlarge': [64, 128, 256],
    }[tier]

    n_head = {
        'micro':  [1, 2],
        'tiny':   [1, 2, 4],
        'small':  [1, 2, 4],
        'medium': [2, 4, 8],
        'large':  [2, 4, 8],
        'xlarge': [2, 4, 8],
    }[tier]

    n_layer = {
        'micro':  [1, 2],
        'tiny':   [1, 2],
        'small':  [1, 2],
        'medium': [1, 2, 4],
        'large':  [2, 4, 6],
        'xlarge': [2, 4, 6, 8],
    }[tier]

    # --- Regolarizzazione ---
    # Dropout basso sui modelli piccoli: non vogliamo sotto-fittare
    # architetture già ridotte. Cresce con la complessità.
    dropout_range = {
        'micro':  (0.1, 0.2),
        'tiny':   (0.15, 0.25),
        'small':  (0.2, 0.3),
        'medium': (0.2, 0.4),
        'large':  (0.2, 0.4),
        'xlarge': (0.2, 0.5),
    }[tier]

    lr_range = (5e-5, 3e-3)
    wd_range = (1e-4, 1e-1)

    # --- Batch size: più grande sulle sequenze corte (meno memoria per token) ---
    batch_size = {
        'micro':  [128, 256, 512],
        'tiny':   [64, 128, 256],
        'small':  [64, 128, 256],
        'medium': [32, 64, 128],
        'large':  [32, 64, 128],
        'xlarge': [16, 32, 64],
    }[tier]

    return {
        'block_size':    block_sizes,
        'n_embd':        n_embd,
        'n_head':        n_head,
        'n_layer':       n_layer,
        'dropout_range': dropout_range,
        'lr_range':      lr_range,
        'wd_range':      wd_range,
        'batch_size':    batch_size,
    }


def get_fixed_params(max_len: int, model_type: str = 'task') -> dict:
    """
    Restituisce i parametri fissi di training (max_iters, eval_iters, eval_interval)
    adattati alla complessità del processo, usando le stesse soglie di get_optuna_search_space.

    Parameters
    ----------
    max_len : int
        Lunghezza massima delle tracce nel dataset (in token).
    model_type : str
        Tipo di modello: 'task', 'time', 'region', 'unified'.

    Returns
    -------
    dict con chiavi 'max_iters', 'eval_iters', 'eval_interval'.
    """
    if max_len < 6:
        tier = 'micro'
    elif max_len < 10:
        tier = 'tiny'
    elif max_len < 24:
        tier = 'small'
    elif max_len < 50:
        tier = 'medium'
    elif max_len < 100:
        tier = 'large'
    else:
        tier = 'xlarge'

    if model_type == 'time':
        # Il time transformer converge sempre prima (regressione scalare più semplice)
        iters = {'micro': 80, 'tiny': 150, 'small': 300, 'medium': 500, 'large': 600, 'xlarge': 700}[tier]
    elif model_type == 'unified':
        # L'unified è il più complesso (task + region + time in un solo modello)
        iters = {'micro': 150, 'tiny': 300, 'small': 500, 'medium': 900, 'large': 1200, 'xlarge': 1500}[tier]
    else:
        # task e region hanno complessità simile
        iters = {'micro': 100, 'tiny': 200, 'small': 500, 'medium': 800, 'large': 1000, 'xlarge': 1200}[tier]

    return {
        'max_iters':     iters,
        'eval_iters':    100,
        'eval_interval': 100,
    }