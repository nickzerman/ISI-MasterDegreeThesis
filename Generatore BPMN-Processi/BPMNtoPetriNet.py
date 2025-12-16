import pandas as pd
import itertools
import random

from pm4py.objects.petri_net.utils.petri_utils import remove_place
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

from random_diagram_generation import SEED_STRING, replace_random_underscore, replace_underscores
from sese_diagram import PARSER
from lark import Tree, Token
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py import save_vis_petri_net

NARY = 1 #1 Se vogliamo l'albero n-ario

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

def createSubnet(net, tree, parent_place, counter):
    """
        Create a petri net recursively

        Parameters
        ----------
        net
            Petri net
        tree
            BPMN converted
        parent_place
            root
        counter
            counter for xor/parallel

        Returns
        ----------
        [p_start, p_end, counter]
        p_start
            Starting place
        p_end
            Ending place
        counter
            Keep counting of xor and parallel region for naming transitions
    """

    if tree.data == 'task':
        task = tree.children[0].value  # Prendo il nome della task in per esempio: T1 per --> [Tree('task', [Token('NAME', 'T1')])

        # Creo tutti i nodi e tutte le transizioni
        start = "start_" + task
        end = "end_" + task
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start = petri_utils.add_place(net, start)
        p_end = petri_utils.add_place(net, end)

        # Credo tutti gli archi
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start, net)
        petri_utils.add_arc_from_to(p_start, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [p_start, p_end, counter]

    if tree.data == 'sequential':
        p_start = parent_place
        for child in tree.children: #Creo la sottorete di ogni figlio e le collego in maniera sequenziale
            subnet = createSubnet(net, child, p_start, counter)
            p_start = subnet[1] #La fine della sottorete è l'inizio della sottorete successiva (o comunque l'end del sequenziale che ritornerò)
            counter = subnet[2] #Aggiorno il counter

        return [parent_place, p_start, counter]

    if tree.data == 'xor':
        start = "start_X" + str(counter)
        inter = "inter_X" + str(counter)
        end = "end_X" + str(counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start = petri_utils.add_place(net, start)
        p_inter = petri_utils.add_place(net, inter)
        p_end = petri_utils.add_place(net, end) #Per collegare la fine dello xor
        counter += 1

        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start, net)

        t_end_c = [] #Salvo la transizione finale di ogni figlio che poi andrò a collegare al p_end
        for child in tree.children:
            subnet = createSubnet(net, child, p_start, counter) #Il padre è sempre parent_place

            in_arcs = subnet[1].in_arcs
            for arc in in_arcs:
                t_end_c.append(arc.source)
            remove_place(net, subnet[1]) #Rimuovo il place finale (per semplificare la petri net, collego tutto a p_end dello xor)

            counter = subnet[2] #Aggiorno il counter

        for arc in t_end_c: #Collego tutte le transizioni dei figli a p_end
            petri_utils.add_arc_from_to(arc, p_inter, net)
        petri_utils.add_arc_from_to(p_inter, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, counter]

    if tree.data == 'parallel':
        start = "start_P" + str(counter) #Creo start e end per transizioni e place della regione parallela
        end = "end_P" + str(counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)

        p_start = {}
        for child in tree.children:
            p_start_c = petri_utils.add_place(net, start)
            p_start[child] = p_start_c

        p_end = petri_utils.add_place(net, end)
        counter += 1

        petri_utils.add_arc_from_to(parent_place, t_start, net) #Transizione di split per la regione parallela

        p_end_c = [] #Mi salvo la fine di ogni figlio, che poi collegherò alla transizione di fine regione parallela
        for child in tree.children:
            petri_utils.add_arc_from_to(t_start, p_start[child], net) #Creo l'arco di inizio collegato alla transizione di split iniziale
            subnet = createSubnet(net, child, p_start[child], counter)
            p_end_c.append(subnet[1])
            counter = subnet[2] #Aggiorno il counter

        for p in p_end_c: #Per ogni place di fine figlio, lo collego alla transizione di fine regione parallela
            petri_utils.add_arc_from_to(p, t_end, net)

        petri_utils.add_arc_from_to(t_end, p_end, net) #Collegamento transizione finale - place finale di regione

        return [parent_place, p_end, counter]

def createClause(tree, open_clauses, end_clauses, node_identity, node_children, region_counter):
    """
        Create open and close clause of tree's nodes
        Create x,+,-> matrix
        Create children region matrix

        Parameters
        ----------
        tree
            BPMN converted
        open_clauses
            dict --> key: node, value: list of clauses
        end_clauses
            dict --> key: node, value: list of clauses
        node_identity
            dict --> key: node, value: list [x,+,->] where we have 1 if the region is that type
        node_children
            dict --> key: node, value: list of nodes
        region_counter
            Keeps the counter for the regions of the Petri Net
        Returns
        ----------
        [open_clauses, end_clauses, region_counter]
        open_clauses
            dict --> key: node, value: list of clauses (tuple)
        end_clauses
            dict --> key: node, value: list of clauses
        name
            name of the sub task/region
        node_identity
            dict --> key: node, value: list [x,+,->] where we have 1 if the region is that type
        node_children
            dict --> key: node, value: list of nodes
        region_counter
            Keeps the counter for the regions of the Petri Net
    """

    if tree.data == "task":
        task = tree.children[0].value  # Prendo il nome della task in per esempio: [Tree('task', [Token('NAME', 'T1')])
        start = "start_" + task
        end = "end_" + task
        open_clauses[task] = [{start}] #Tuple
        end_clauses[task] = [{end}]

        #Lo start è l'inizio della task, mentre l'end è la fine della task
        return [open_clauses, end_clauses, task, node_identity, node_children, region_counter]

    # Inizializzo la nuova regione
    region = "R" + str(region_counter)

    # Aggiorno il region_counter come nella funzione del createSubnet e chiamo ricorsivamente il createClause
    region_counter += 1
    children = [] #Tengo traccia dei figli, in questo modo facilmente creo open_clauses ed end_clauses
    for child in tree.children:
        open_clauses, end_clauses, name, node_identity, node_children, region_counter = createClause(child, open_clauses, end_clauses, node_identity, node_children, region_counter)
        children.append(name)

    node_children[region] = children #Per ogni regione, ho la lista solamente dei figli

    if tree.data == "sequential":
        node_identity[region] = [0,0,1] #Identificazione del tipo di regione

        open_clauses[region] = open_clauses[children[0]] #Prendo solo il primo figlio, che aprirà la regione sequenziale
        end_clauses[region] = end_clauses[children[-1]] #Prendo solo l'ultimo figlio, che chiuderà la regione sequenziale

    elif tree.data == "xor":
        node_identity[region] = [1,0,0] #Identificazione del tipo di regione

        #Per ogni figlio, le sue clauses open e di end valgono per lo xor
        open_clauses_current = []
        end_clauses_current = []
        for child in children:
            open_clauses_current += open_clauses[child]
            end_clauses_current += end_clauses[child]
        open_clauses[region] = open_clauses_current
        end_clauses[region] = end_clauses_current

    elif tree.data == "parallel":
        node_identity[region] = [0,1,0] #Identificazione del tipo di regione

        #Per ogni figlio, le sue clauses open valgono per il parallelo
        open_clauses_current = []
        for child in children:
            open_clauses_current += open_clauses[child]
        open_clauses[region] = open_clauses_current

        #Mentre devo fare le combinazioni tra le varie end clauses dei figli
        all_children_end_clauses = [end_clauses[child] for child in children]

        # L'asterisco * scompatta la lista in argomenti separati per product
        end_clauses[region] = [
            set().union(*combination)
            for combination in itertools.product(*all_children_end_clauses)
        ]

    return open_clauses, end_clauses, region, node_identity, node_children, region_counter

def visitePetriNetClauses(clauses):
    """
        From the set of open_clauses, take all the possible regions and tasks

        Parameters
        ----------
        clauses
            dict --> key: node, value: list of clauses

        Returns
        ----------
        regions, tasks
        regions
            regions of the process/net
        tasks
            tasks of the process/net
    """
    regions = [] #Aggiungo R0 di base
    tasks = []

    for label in clauses.keys():
        if label[0] == "R": #Regione
            if label not in regions:
                regions.append(label)
        else:
            if label not in tasks:
                tasks.append(label)

    return regions, tasks

def generateTrace(net, initial_marking, final_marking, check):
    """
        Generate a trace of Petri Net.
        First all end_R/P, then all start_R/P if they're from a place with only 1 output arc, else choose randomly from all the enabled transitions.

        After the generation, the trace is cleaned.

        Parameters
        ----------
        net
            Petri Net
        initial_marking
            Initial Marking of the Petri Net
        final_marking
            Final Marking of the Petri Net
        check
            list of char like 'P' and 'X', we use them to consume start and end region when we can

        Returns
        ----------
        trace
            Trace Generated
    """
    current_marking = initial_marking
    trace = []
    start = tuple(["start_" + c for c in check])
    end = tuple(["end_" + c for c in check])

    while current_marking != final_marking: #Fino a quando il marking non corrisponde al finale, quindi fino a quando non ho finito di generare la traccia
        transitions = enabled_transitions(net, current_marking) #Lista delle possibili transizioni possibili al current marking

        #Prima di tutto chiudo le regioni aperte (che posso chiudere ovviamente)
        items_end = [item for item in transitions if item.label.startswith(end)]
        if items_end:
            choice = random.choice(items_end)
            trace.append(choice.label)
            current_marking = execute(choice, net, current_marking)
        else:
            #Poi apro tutte le regioni apribili MA il place da cui l'arco di input proviene deve avere solo un arco uscente, altrimenti NON POSSO dare precedenza (tipo tra regione e task darei sempre precedenza ad una regione in uno xor e NON va bene)
            items_start = [item for item in transitions if item.label.startswith(start)]
            items_start_updated = []
            for item in items_start: #Check spiegato in precedenza
                in_arc = list(item.in_arcs)
                source = in_arc[0].source
                if len(list(source.out_arcs)) == 1:
                    items_start_updated.append(item)

            if items_start_updated: #Se c'è qualche regione che parte che posso consumare senza problemi, scelgo randomicamente da queste
                choice = random.choice(items_start_updated)
                trace.append(choice.label)
                current_marking = execute(choice,net,current_marking)
            else: #Altrimenti tiro a caso su cosa fare tra tutte le transizioni rimanenti
                choice = random.choice(list(transitions))
                trace.append(choice.label)
                current_marking = execute(choice,net,current_marking)

    #Pulisco la traccia
    trace_cleaned = []
    for t in trace:
        if not (t.startswith(start) or t.startswith(end)):
            trace_cleaned.append(t)

    return trace_cleaned

def filterClauses(clauses):
    """
    This function remove the key-task in open_clauses and end_clauses (we don't need them)

    Parameters
    ----------
    clauses
        dict --> key: node, value: list of clauses

    Returns
    ----------
    clauses_filtered
        dict --> key: node, value: list of clauses (without TASK in keys)
    """
    clauses_filtered = {k:v for k,v in clauses.items() if not k.startswith("T")}
    return clauses_filtered

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

if __name__ == "__main__":
    current_string = SEED_STRING
    probabilities = 0.34,0.33,0.33 #xor, parallel, seq

    iterations = 7 #Quante task diverse
    for _ in range(iterations):
        current_string = replace_random_underscore(current_string, probabilities)

    process = replace_underscores(current_string)
    tree = PARSER.parse(process)

    #ALBERO GIOCATTOLO
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

    #Inizializzo Oggetto Petri Net
    net = PetriNet('Net')
    initial_marking = Marking()
    final_marking = Marking()
    root = petri_utils.add_place(net, "root")

    netValue = createSubnet(net, tree, root, 0)  # Avvio il processo per creare la Petri Net
    initial_marking[netValue[0]] = 1
    final_marking[netValue[1]] = 1

    #Creo output .png della Petri Net
    file_path_png = "petri_net_output.png"

    save_vis_petri_net(
        net,
        initial_marking,
        final_marking,
        file_path_png,
        format="png"  # Specifica il formato
    )

    generated_traces = []
    for i in range(3):
        generated_traces.append(generateTrace(net, initial_marking, final_marking, ["P", "X"]))

    #Creo le clauses e tutte le matrici, sia per aiutarmi nell'encoding sia per le identificazioni delle regioni
    open_clauses, end_clauses, _, node_identity, node_children, _ = createClause(tree, {}, {}, {}, {}, 0)

    #Creazione matrice identità delle regioni
    df_region_identity = pd.DataFrame.from_dict(node_identity, orient='index').sort_index()
    df_region_identity.columns = ['X', '+', '->']
    print(df_region_identity)

    #Mi prendo tutte le varie regioni e tutte le task in modo tale da poter lavorare sulle tracce come preferisco
    regions, tasks = visitePetriNetClauses(open_clauses)
    open_clauses=filterClauses(open_clauses)
    end_clauses=filterClauses(end_clauses)

    regions.sort()
    tasks.sort()

    #Creazione matrice regioni-figli per le regioni
    df_region_children = pd.Series(node_children).explode()
    df_region_children = pd.crosstab(df_region_children.index, df_region_children)
    df_region_children = df_region_children.reindex(index=regions, columns=regions + tasks, fill_value=0)
    df_region_children = df_region_children.astype(int)
    df_region_children.index.name = None
    df_region_children.columns.name = None
    print(df_region_children)

    traceEncoded_regions, traceEncoded_tasks = getEncoding(generated_traces, regions, tasks, open_clauses,end_clauses)

    traceEncoded_regions.to_csv("regions_full.csv", index=True)
    traceEncoded_tasks.to_csv("tasks_full.csv", index=True)

    for trace in generated_traces:
        print(trace)