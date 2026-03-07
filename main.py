import torch

from core import *
from utils import *
from lark import Tree, Token
from pm4py import save_vis_petri_net
import pandas as pd

# SETTINGS
NARY = 1
CURRENT_STRING = SEED_STRING
PROBABILITIES = 0.34, 0.33, 0.33
FILE_PATH_PNG = "petri_net_output.png"
TRACE_ENC_REG = "data/regions_full.csv"
TRACE_ENC_TAS = "data/tasks_full.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-4


def get_batch(split, train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size, device):
    # Seleziona i dati corretti
    X_data = train_data_X if split == 'train' else val_data_X
    Y_data = train_data_Y if split == 'train' else val_data_Y

    # Genera indici casuali
    ix = torch.randint(len(X_data), (batch_size,))

    # Estrae e sposta sul device
    x = X_data[ix].to(device)
    y = Y_data[ix].to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters, train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size, device):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # Richiama get_batch passando i parametri ricevuti
            X, Y = get_batch(split, train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

def main():
    iterations = 7  # Quante task diverse
    for _ in range(iterations):
        current_string = replace_random_underscore(CURRENT_STRING, PROBABILITIES)

    process = replace_underscores(current_string)
    tree = PARSER.parse(process)

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

    if NARY:
        tree = createNAryTree(tree)

    # Oggetto PetriNetP - si inizializza automaticamente con il suo costruttore
    net = PetriNetP(tree)

    save_vis_petri_net(
        net.net,
        net.initial_marking,
        net.final_marking,
        FILE_PATH_PNG,
        format="png"  # Specifica il formato
    )

    # Oggetto Generator
    generator = Generator(300, net)

    # Creazione matrice identità delle regioni
    df_region_identity = pd.DataFrame.from_dict(net.node_identity, orient='index').sort_index()
    df_region_identity.columns = ['X', '+', '->']
    print(df_region_identity)

    # Creazione matrice regioni-figli per le regioni
    df_region_children = pd.Series(net.node_children).explode()
    df_region_children = pd.crosstab(df_region_children.index, df_region_children)
    df_region_children = df_region_children.reindex(index=net.regions, columns=net.regions + net.tasks, fill_value=0)
    df_region_children = df_region_children.astype(int)
    df_region_children.index.name = None
    df_region_children.columns.name = None
    print(df_region_children)

    traceEncoded_regions, traceEncoded_tasks = getEncoding(generator.generatedTraces, net.regions, net.tasks, net.open_clauses, net.end_clauses)

    num_regions = len([i for i in traceEncoded_regions.index if str(i).startswith('R')])
    num_tasks = len([i for i in traceEncoded_tasks.index if str(i).startswith('T')])

    traceEncoded_regions.to_csv(TRACE_ENC_REG, index=True)
    traceEncoded_tasks.to_csv(TRACE_ENC_TAS, index=True)

    traces = cutTraces(traceEncoded_regions, traceEncoded_tasks)

    block_size = 8
    n_embd = 64
    dropout = 0.2
    n_head = 4 #Bisogna implementare la multihead attention in caso (basta copiare il codice)
    n_layer = 3

    X, Y = create_training_set(traces,8)

    model = BPMNTransformer(num_regions+num_tasks, block_size, n_embd, dropout, n_head, n_layer)
    m = model.to(device)

    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    n = int(0.8 * len(X))

    train_data_X = X[:n]
    train_data_Y = Y[:n]
    val_data_X = X[n:]
    val_data_Y = Y[n:]

    batch_size = 32
    eval_iters = 200
    eval_interval = 300
    max_iters = 2000

    # Dentro il ciclo for iter in range(max_iters):
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, eval_iters, train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size,
                                   device)
            print(f"Step {iter}: Train Loss {losses['train']:.4f} | Val Loss {losses['val']:.4f}")

        # A. Pesca i dati (usando la funzione esterna)
        xb, yb = get_batch('train', train_data_X, train_data_Y, val_data_X, val_data_Y, batch_size, device)

        # B. FORWARD PASS: Il modello "gira" e produce una previsione
        logits, loss = model(xb, yb)

        # C. RESET GRADIENTI: Puliamo i calcoli del giro precedente
        optimizer.zero_grad(set_to_none=True)

        # D. BACKWARD PASS: Calcoliamo l'errore per ogni neurone (L'Anima del training)
        loss.backward()

        # E. OPTIMIZER STEP: Aggiorniamo i pesi per sbagliare meno al prossimo giro
        optimizer.step()

    print("\n--- TEST DI GENERAZIONE CON SEED REALE ---")
    model.eval()
    with torch.no_grad():
        # Prendi i primi 8 step della prima traccia di validazione
        # context shape: (1, 8, num_bits)
        context = val_data_X[0:1].to(device)

        print("Inizio traccia reale fornito al modello...")

        generated_steps = []
        for _ in range(15):
            cond_context = context[:, -block_size:, :]
            logits, _ = model(cond_context)
            probs = torch.sigmoid(logits)

            # PROVA QUESTO: Invece di > 0.5, guarda i valori grezzi delle probabilità
            # se sono tutti 0.99, c'è un problema di scala nella loss
            next_step = (probs > 0.5).float()

            context = torch.cat((context, next_step.unsqueeze(1)), dim=1)
            generated_steps.append(next_step.cpu().squeeze().numpy())

        # Stampa
        for i, step in enumerate(generated_steps):
            print(f"Step {i + 1} previsto: {step.astype(int)}")

if __name__ == '__main__':
    main()