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

    generator = Generator(3, net)

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

    traceEncoded_regions.to_csv(TRACE_ENC_REG, index=True)
    traceEncoded_tasks.to_csv(TRACE_ENC_TAS, index=True)

if __name__ == '__main__':
    main()