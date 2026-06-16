import re
from lark import Tree
import torch
from pm4py.objects.process_tree.obj import Operator

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

    L'idea è semplice: processi con tracce corte (pochi token) non hanno
    bisogno di un Transformer enorme. Un modello sovra-parametrizzato su
    dati semplici spreca tempo di calcolo e va facilmente in overfitting.
    Al contrario, processi con tracce lunghe e complesse richiedono un
    context window grande e più layer per catturare dipendenze a lungo raggio.

    Soglie:
        max_len <= 40   → "small"  (XOR/SEQ semplici, pochi task)
        max_len <= 120  → "medium" (strutture miste moderate)
        max_len >  120  → "large"  (loop annidati, paralleli profondi)

    Parameters
    ----------
    max_len : int
        Lunghezza massima delle tracce nel dataset (in token).
    model_type : str
        Tipo di modello: 'task', 'time', 'region', 'unified'.
        Influenza i range di block_size e n_embd.

    Returns
    -------
    dict con chiavi 'block_size', 'n_embd', 'n_head', 'n_layer',
    'dropout_range', 'lr_range', 'wd_range', 'batch_size'.
    """
    # --- Determina il "tier" di complessità ---
    if max_len <= 40:
        tier = 'small'
    elif max_len <= 120:
        tier = 'medium'
    else:
        tier = 'large'

    # --- Block size: deve sempre coprire max_len per non "tagliare" il contesto ---
    # Prendiamo come minimo la prima potenza di 2 >= max_len, poi lasciamo
    # a Optuna di scegliere tra valori più grandi (più contesto non nuoce mai).
    if tier == 'small':
        block_sizes = [32, 64]
    elif tier == 'medium':
        block_sizes = [64, 128, 256]
    else:
        block_sizes = [128, 256, 512]

    # --- Architettura: embd, heads, layers ---
    if tier == 'small':
        n_embd   = [16, 32, 64]
        n_head   = [1, 2, 4]
        n_layer  = [1, 2]
    elif tier == 'medium':
        n_embd   = [32, 64, 128]
        n_head   = [2, 4, 8]
        n_layer  = [1, 2, 4]
    else:
        n_embd   = [64, 128, 256]
        n_head   = [2, 4, 8]
        n_layer  = [2, 4, 6, 8]

    # --- Regolarizzazione: dropout più alto su modelli grandi ---
    if tier == 'small':
        dropout_range = (0.2, 0.3)
    elif tier == 'medium':
        dropout_range = (0.2, 0.4)
    else:
        dropout_range = (0.2, 0.5)

    # --- Learning rate e weight decay (invarianti per tier) ---
    lr_range = (5e-5, 3e-3)
    wd_range = (1e-4, 1e-1)

    # --- Batch size: più piccola sulle sequenze lunghe (più memoria usata per token) ---
    if tier == 'small':
        batch_size = [32, 64, 128]
    elif tier == 'medium':
        batch_size = [32, 64, 128]
    else:
        batch_size = [16, 32, 64]

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

    Logica:
        - Processi "small": il modello converge in fretta → pochi iter bastano.
        - Processi "medium": convergenza moderata → iter intermedi.
        - Processi "large": strutture complesse → servono più iterazioni per
          superare il warmup e stabilizzare la curva di apprendimento.

    Il TimeTransformer converge sempre più in fretta degli altri (loss più
    semplice, regressione su un singolo scalare), quindi ha max_iters ridotto.

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
    if max_len <= 40:
        tier = 'small'
    elif max_len <= 120:
        tier = 'medium'
    else:
        tier = 'large'

    if model_type == 'time':
        # Il time transformer converge sempre prima
        iters = {'small': 300, 'medium': 500, 'large': 600}[tier]
    elif model_type == 'unified':
        # L'unified è il più complesso (task + region + time in un solo modello)
        iters = {'small': 500, 'medium': 900, 'large': 1200}[tier]
    else:
        # task e region hanno complessità simile
        iters = {'small': 500, 'medium': 800, 'large': 1000}[tier]

    # eval_interval e eval_iters fissi a 100 per tutti i tier
    eval_interval = 100
    eval_iters    = 100

    return {
        'max_iters':     iters,
        'eval_iters':    eval_iters,
        'eval_interval': eval_interval,
    }