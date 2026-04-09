import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from core import *
from utils import *
from lark import Tree, Token
from pm4py import save_vis_petri_net
import pandas as pd
import sys
#print(f"Versione Python: {sys.version}")

#print(torch.cuda.is_available())
#print(torch.__version__)

# SETTINGS
NARY = 1
PROBABILITIES = 0.25,0.25,0.25,0.25
FILE_PATH_PNG = "petri_net_output.png"
TRACE_ENC_REG = "data/regions_full.csv"
TRACE_ENC_TAS = "data/tasks_full.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4


'''def get_batch(split, train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size, device):
    # Seleziona i dati corretti
    X_data = train_data_X if split == 'train' else val_data_X
    Y_data = train_data_Y if split == 'train' else val_data_Y

    # Genera indici casuali
    ix = torch.randint(len(X_data), (batch_size,))

    # Estrae e sposta sul device
    x = X_data[ix].to(device)
    y = Y_data[ix].to(device)

    return x, y'''

def get_batch(split, train_data, val_data, batch_size, block_size, device):
    # Seleziona i dati corretti
    data = train_data if split == 'train' else val_data

    # Genera indici casuali
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x,y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data, val_data, batch_size, block_size, device):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # Richiama get_batch passando i parametri ricevuti
            X, Y = get_batch(split, train_data, val_data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

def main():
    iterations = 7  # Quante task diverse
    current_string = SEED_STRING
    for _ in range(iterations):
        current_string = replace_random_underscore(current_string, PROBABILITIES)

    #print(current_string)

    process = replace_underscores(current_string)
    tree = PARSER.parse(process)

    '''
    # ALBERO GIOCATTOLO
    # Da rimuovere
    tree = Tree('xor', [
        # 1. Primo figlio di XOR (T1)
        Tree('task', [Token('NAME', 'T1')]),

        # 2. Secondo figlio di XOR (Blocco Sequential)
        Tree('sequential', [

            # 2.1 Primo figlio di Sequential (Parallel)
            Tree('parallel', [
                # 2.1.1 Ramo sinistro del Parallel (Xor annidati)
                Tree('xor', [
                    Tree('task', [Token('NAME', 'T2')]),
                    Tree('xor', [
                        Tree('task', [Token('NAME', 'T3')]),
                        Tree('task', [Token('NAME', 'T4')])
                    ])
                ]),
                # 2.1.2 Ramo destro del Parallel (Parallel T5, T6)
                Tree('parallel', [
                    Tree('task', [Token('NAME', 'T5')]),
                    Tree('task', [Token('NAME', 'T6')])
                ])
            ]),

            # 2.2 Secondo figlio di Sequential (Sequential T7, T8)
            Tree('sequential', [
                Tree('task', [Token('NAME', 'T7')]),
                Tree('task', [Token('NAME', 'T8')])
            ])
        ])
    ])
    '''

    tree = Tree('xor', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('xor', [Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')])])])]), Tree('parallel', [Tree('task', [Token('NAME', 'T4')]), Tree('loop', [Tree('task', [Token('NAME', 'T5')])])])])

    # sei arrivato qui nicolò
    # hai il tuo albero giocattolo coi loop in teoria
    # adesso devi fixare quello che ci sarà sbagliato nell'encoding e in teoria ci sei

    if NARY:
        tree = createNAryTree(tree)

    #print(tree)


    # Oggetto PetriNetP - si inizializza automaticamente con il suo costruttore
    net = PetriNetP(tree)

    save_vis_petri_net(
        net.net,
        net.initial_marking,
        net.final_marking,
        FILE_PATH_PNG,
        format="png"  # Specifica il formato
    )

    #print(net.open_clauses)
    #print(net.end_clauses)

    # Oggetto Generator
    generator = Generator(5, net)
    for i in generator.generatedTraces:
        print(i)

    # Creazione matrice identità delle regioni
    df_region_identity = pd.DataFrame.from_dict(net.node_identity, orient='index').sort_index()
    df_region_identity.columns = ['X', '+', '->', '<>']
    #print(df_region_identity)

    # Creazione matrice regioni-figli per le regioni
    df_region_children = pd.Series(net.node_children).explode()
    df_region_children = pd.crosstab(df_region_children.index, df_region_children)
    df_region_children = df_region_children.reindex(index=net.regions, columns=net.regions + net.tasks, fill_value=0)
    df_region_children = df_region_children.astype(int)
    df_region_children.index.name = None
    df_region_children.columns.name = None
    #print(df_region_children)

    traceEncoded_regions, traceEncoded_tasks = getEncoding(generator.generatedTraces, net.regions, net.tasks, net.open_clauses, net.end_clauses)

    num_regions = len([i for i in traceEncoded_regions.index if str(i).startswith('R')])
    num_tasks = len([i for i in traceEncoded_tasks.index if str(i).startswith('T')])

    traceEncoded_regions.to_csv(TRACE_ENC_REG, index=True)
    traceEncoded_tasks.to_csv(TRACE_ENC_TAS, index=True)

    df_traces = pd.concat([traceEncoded_regions, traceEncoded_tasks], axis=0)
    print(df_traces)

    '''

    df_traces = df_traces.T

    df_tracescopy = df_traces.copy()

    unique_columns = df_traces.drop_duplicates()
    unique_tuple = [tuple(x) for x in unique_columns.values]

    # 2. Creiamo i dizionari di mappatura
    # bit_to_id: trasforma la colonna di 12 bit in un numero
    # id_to_bit: trasforma il numero nei 12 bit originali (per la generazione)
    bit_to_id = {v: i for i, v in enumerate(unique_tuple)}
    id_to_bit = {i: v for i, v in enumerate(unique_tuple)}

    vocab_size = len(unique_columns)

    encode = lambda a: [bit_to_id[tuple(x)] for x in a]
    decode = lambda b: [id_to_bit[tuple(x)] for x in b]

    data = torch.tensor(encode(df_traces.values), dtype=torch.long)

    #traces = cutTraces(traceEncoded_regions, traceEncoded_tasks)

    block_size = 8
    n_embd = 128
    dropout = 0.2
    n_head = 4
    n_layer = 4
    batch_size = 32
    eval_iters = 200
    eval_interval = 500
    max_iters = 5000

    #X, Y = create_training_set(traces,8)

    model = BPMNTransformer(vocab_size, num_regions+num_tasks, block_size, n_embd, dropout, n_head, n_layer)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    n = int(0.8 * len(df_traces))
    
    train_data = data[:n]
    val_data = data[n:]

    #train_data_X = X[:n]
    #train_data_Y = Y[:n]
    #val_data_X = X[n:]
    #val_data_Y = Y[n:]
    '''

    '''xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)
    print(xb)
    print(yb)'''

    '''
    # Dentro il ciclo for iter in range(max_iters):
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, train_data, val_data, batch_size, block_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # A. Pesca i dati (usando la funzione esterna)
        xb, yb = get_batch('train', train_data, val_data, batch_size, block_size, device)

        # B. FORWARD PASS: Il modello "gira" e produce una previsione
        logits, loss = model(xb, yb)

        # C. RESET GRADIENTI: Puliamo i calcoli del giro precedente
        optimizer.zero_grad(set_to_none=True)

        # D. BACKWARD PASS: Calcoliamo l'errore per ogni neurone (L'Anima del training)
        loss.backward()

        # E. OPTIMIZER STEP: Aggiorniamo i pesi per sbagliare meno al prossimo giro
        optimizer.step()
    '''

if __name__ == '__main__':
    main()