import torch
from lark import Tree
import itertools
import pandas as pd

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
    for child in children_processed: #Ciclo i figli processati
        if child.data == tree.data: #Se il tipo di nodo è lo stesso elimino il nodo e copio i figli
            children.extend(child.children)
        else: #Altrimenti copio tutto il sottoalbero col tipo di nodo diverso
            children.append(child)

    return Tree(tree.data, children) #Ritorno l'albero n-ario

def getEncoding(traces, regions, tasks, open_clauses, end_clauses):
    """
    Encoding traces generated
    1 --> if task/region starts or is executed in current step
    0 --> if task/region ends or is not executed in current step

    Parameters
    ----------
    traces
        traces generated
    regions
        regions of the process/net
    tasks
        tasks of the process/net
    open_clauses
        opening clauses of the regions/tasks
    end_clauses
        ending clauses of the regions/tasks

    Returns
    ----------
    df_regions, df_tasks
    df_regions
        dataframe with trace regions encoded
    df_tasks
        dataframe with trace tasks encoded
    """
    #Inizializzo le matrici per la codifica delle tracce (regioni e task)
    traceEncoded_regions = {r: [] for r in regions}
    traceEncoded_tasks = {t: [] for t in tasks}

    for trace in traces:
        first_step = True
        ended_tasks = []

        for step in trace:
            regions_step = regions.copy() #Per tenere 'traccia' (brutto gioco di parole) delle ragioni che finiscono o iniziano nello step corrente

            current_task = step.split("_")[1] #Prendo la current_task per aiutarmi nel traceEncoded_tasks

            if step.startswith("start"):
                traceEncoded_tasks[current_task].append(1) #Se task inizia allo step corrente, il suo valore nella matrice diventa 1 (prima e precedente era 0)
                for key, clauses in open_clauses.items(): #Se l'inizio del task appartiene all'open_clauses di una regione, il valore codificato della matrice è 1
                    if {step} in clauses:
                        regions_step.remove(key)
                        traceEncoded_regions[key].append(1)
            else:
                ended_tasks.append(step)
                traceEncoded_tasks[current_task].append(0) #Se task finisce  allo step corrente, il suo valore nella matrice diventa 0 (prima e precedente era 1)

                #Genero tutte le combinazioni possibili di task finite per controllare se alcuni regioni finiscono allo step corrente (potrei avere una combinazione di end di task)
                combinations = [
                    set(c)
                    for i in range(2, len(ended_tasks) + 1)
                    for c in itertools.combinations(ended_tasks, i)
                ]
                combinations.append({step}) #Aggiungo la singola end task dello step corrente (potrebbe chiudere da sola una o più regioni)

                for key, clauses in end_clauses.items(): #Se una delle combinazioni appartiene all'end_clauses di una regione, il valore codificato della matrice torna 0
                    for c in combinations:
                        if c in clauses:
                            regions_step.remove(key)
                            traceEncoded_regions[key].append(0)
                            break

            for region in regions_step: #Sistemo le regioni non considerate fino ad ora dallo step corrente (0 se siamo al primo step, altrimenti prendiamo il valore precedente --> se una regione è iniziata allo step precedente e non è finita, il valore deve rimanere 1)
                if first_step:
                    traceEncoded_regions[region].append(0)
                else:
                    traceEncoded_regions[region].append(traceEncoded_regions[region][-1])

            for task in tasks: #Stessa cosa delle regioni, fatta per i task
                if first_step and task!=current_task:
                    traceEncoded_tasks[task].append(0)
                elif task!=current_task:
                    traceEncoded_tasks[task].append(traceEncoded_tasks[task][-1])

            first_step = False

    #Creo i due dataframe codificati delle regioni e delle task
    df_regions = pd.DataFrame(traceEncoded_regions).T
    df_tasks = pd.DataFrame(traceEncoded_tasks).T

    return df_regions, df_tasks


def cutTraces(df_regions, df_tasks):
    # Combino i due dataframe uno sotto l'altro, in modo da avere la colonna completa
    df_combined = pd.concat([df_regions, df_tasks], axis=0)

    divided_traces = []
    current = []

    for col in df_combined.columns:
        if (df_combined[col] == 0).all(): #Se arrivo in un punto in cui la colonna è tutta zero, significa che ho una nuova traccia, resetto current e salvo la traccia iterata fino ad ora
            if current:
                divided_traces.append(df_combined[current])
                current = []
        else:
            current.append(col)

    return divided_traces

def create_training_set(df_traces, block_size):
    X, Y = [], []
    for trace in df_traces:
        data = torch.tensor(trace.values.T, dtype=torch.float32) #Tolgo numero colonne e numero regioni, creando il tensor

        padding = torch.zeros((block_size,data.shape[1]))
        padding_final = torch.zeros((1,data.shape[1]))
        extended_trace = torch.cat((padding, data, padding_final), dim=0)

        for i in range(block_size, len(extended_trace)):
            X.append(extended_trace[i-block_size : i])
            Y.append(extended_trace[i])

    return torch.stack(X), torch.stack(Y)