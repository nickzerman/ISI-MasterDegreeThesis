from datetime import datetime
import pandas as pd
import itertools
import random

from pm4py.objects.petri_net.utils.petri_utils import remove_place, get_transition_by_name
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

from random_diagram_generation import SEED_STRING, replace_random_underscore, replace_underscores
from sese_diagram import PARSER, print_sese_diagram, print_tree, dot_tree
from stats import max_nested_xor, max_independent_xor
from lark import Lark, Tree, Token
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py import view_petri_net, save_vis_petri_net

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo import simulation

REGION = 1 #1 se vogliamo Petri Net con Regioni, 0 se vogliamo quella collassata

def createSubnetRegion(net, tree, parent_place, region_counter):
    """
        Create a petri net recursively with region transitions

        Parameters
        ----------
        net
            Petri net
        tree
            BPMN converted
        parent_place
            root
        region_counter
            Keeps the counter for the regions of the Petri Net

        Returns
        ----------
        [p_start, p_end, region_counter]
        p_start
            Starting place
        p_end
            Ending place
        region_counter
            Keeps the counter for the regions of the Petri Net
    """
    type = tree.data

    if type == "task":
        task = tree.children[0].value #Prendo il nome della task in per esempio: [Tree('task', [Token('NAME', 'T1')])

        # Creo tutti i nodi e tutte le transizioni
        start = "start_" + task
        end = "end_" + task
        t_start = petri_utils.add_transition(net,start,start)
        t_end = petri_utils.add_transition(net,end,end)
        p_start = petri_utils.add_place(net,start)
        p_end = petri_utils.add_place(net,end)

        # Credo tutti gli archi
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start, net)
        petri_utils.add_arc_from_to(p_start, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [p_start, p_end, region_counter]

    if type == "sequential":
        #Inizializzo la nuova regione sequenziale (2 places - uno per inizio e fine e 2 transitions - stessa cosa)
        start = "start_R" + str(region_counter)
        end = "end_R" + str(region_counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start = petri_utils.add_place(net,start)
        p_end = petri_utils.add_place(net, end)
        region_counter += 1

        #Creo i primi archi - dal padre alla transizione di inizio regione sequenziale al place di inizio regione sequenziale
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start, net)

        #Prima 'sottorete' collegata al place di inizio regione
        subnet1 = createSubnetRegion(net, tree.children[0], p_start, region_counter)
        p_end1 = subnet1[1]
        region_counter = subnet1[2]

        #Secondo 'sottorete' collegata al place di fine 'sottorete' precedente
        subnet2 = createSubnetRegion(net, tree.children[1], p_end1, region_counter)
        p_end2 = subnet2[1]
        region_counter = subnet2[2]

        #Creo gli ultimi archi - dalla fine della seconda 'sottorete' alla transizione di fine regione sequenziale al place di fine regione sequenziale
        petri_utils.add_arc_from_to(p_end2, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, region_counter]

    if type == "xor":
        # Inizializzo la nuova regione xor (3 places - uno per inizio, uno intermedio e uno per la fine e 2 transitions - una fine e un inizio comune)
        start = "start_R" + str(region_counter)
        inter = "inter_R" + str(region_counter)
        end = "end_R" + str(region_counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start = petri_utils.add_place(net, start)
        p_inter = petri_utils.add_place(net, inter)
        p_end = petri_utils.add_place(net, end)
        region_counter += 1

        # Creo i primi archi - dal padre alla transizione di inizio regione disgiunzione al place di inizio regione disgiunzione
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start, net)

        # Prima 'sottorete' collegata al place di inizio regione
        subnet1 = createSubnetRegion(net, tree.children[0], p_start, region_counter)
        region_counter = subnet1[2]

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        name = subnet1[1].name
        t_end1 = get_transition_by_name(net, name)
        remove_place(net, subnet1[1])

        # Secondo 'sottorete' collegata al place di inizio regione
        subnet2 = createSubnetRegion(net, tree.children[1], p_start, region_counter)
        region_counter = subnet2[2]

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        name = subnet2[1].name
        t_end2 = get_transition_by_name(net, name)
        remove_place(net, subnet2[1])

        # Creo gli ultimi archi - dalle transazioni di fine sottorete al place intermedio di regione, poi alla transizione di fine regione e successivamente al place di fine regione
        petri_utils.add_arc_from_to(t_end1, p_inter, net)
        petri_utils.add_arc_from_to(t_end2, p_inter, net)
        petri_utils.add_arc_from_to(p_inter, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, region_counter]

    if type == "parallel":
        # Inizializzo la nuova regione parallel (2 transitions - inizio e fine comune ad entrambi i rami e 3 places - 1 inizio per ramo e 1 fine in comune)
        start = "start_R" + str(region_counter)
        end = "end_R" + str(region_counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start1 = petri_utils.add_place(net, start)
        p_start2 = petri_utils.add_place(net, start)
        p_end = petri_utils.add_place(net, end)
        region_counter += 1

        # Creo i primi archi - dal padre alla transizione di inizio regione parallela ai place di inizio regione parallela privata ai rami
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start1, net)
        petri_utils.add_arc_from_to(t_start, p_start2, net)

        # Prima 'sottorete' collegata al place di inizio regione del ramo 'A'
        subnet1 = createSubnetRegion(net, tree.children[0], p_start1, region_counter)
        p_end1 = subnet1[1]
        region_counter = subnet1[2]

        # Secondo 'sottorete' collegata al place di inizio regione del ramo 'B'
        subnet2 = createSubnetRegion(net, tree.children[1], p_start2, region_counter)
        p_end2 = subnet2[1]
        region_counter = subnet2[2]

        # Creo gli ultimi archi - dalla fine delle due 'sottoreti' alla transizione comune di fine regione parallela al place di fine regione parallela
        petri_utils.add_arc_from_to(p_end1, t_end, net)
        petri_utils.add_arc_from_to(p_end2, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, region_counter]

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
    type = tree.data

    if type == "task":
        task = tree.children[0].value  # Prendo il nome della task in per esempio: [Tree('task', [Token('NAME', 'T1')])

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

    if type == "sequential":
        # Inizializzo la nuova regione sequenziale

        # Prima 'sottorete' collegata al place del componente precedente (al padre)
        subnet1 = createSubnet(net, tree.children[0], parent_place, counter)

        # Secondo 'sottorete' collegata al place di fine 'sottorete' precedente
        subnet2 = createSubnet(net, tree.children[1], subnet1[1], subnet1[2])

        return [parent_place, subnet2[1], subnet2[2]]

    if type == "xor":
        # Inizializzo la nuova regione xor
        end = "end_X" + str(counter)
        p_end = petri_utils.add_place(net, end)

        counter += 1

        # Prima 'sottorete' collegata al place di inizio regione
        subnet1 = createSubnet(net, tree.children[0], parent_place, counter)

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        in_arcs = subnet1[1].in_arcs
        t_end = []
        for arc in in_arcs:
            t_end.append(arc.source)
        remove_place(net, subnet1[1])

        # Secondo 'sottorete' collegata al place di inizio regione
        subnet2 = createSubnet(net, tree.children[1], parent_place, subnet1[2])

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        in_arcs = subnet2[1].in_arcs
        for arc in in_arcs:
            t_end.append(arc.source)
        remove_place(net, subnet2[1])

        # Creo gli ultimi archi - dalle transazioni di fine sottorete al place intermedio di regione, poi alla transizione di fine regione e successivamente al place di fine regione
        for arc in t_end:
            petri_utils.add_arc_from_to(arc, p_end, net)

        return [parent_place, p_end, subnet2[2]]

    if type == "parallel":
        # Inizializzo la nuova regione parallel (2 transitions - inizio e fine comune ad entrambi i rami e 3 places - 1 inizio per ramo e 1 fine in comune)
        start = "start_P" + str(counter)
        end = "end_P" + str(counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start1 = petri_utils.add_place(net, start)
        p_start2 = petri_utils.add_place(net, start)
        p_end = petri_utils.add_place(net, end)

        counter += 1

        # Creo i primi archi - dal padre alla transizione di inizio regione parallela ai place di inizio regione parallela privata ai rami
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start1, net)
        petri_utils.add_arc_from_to(t_start, p_start2, net)

        # Prima 'sottorete' collegata al place di inizio regione del ramo 'A'
        subnet1 = createSubnet(net, tree.children[0], p_start1, counter)
        p_end1 = subnet1[1]

        # Secondo 'sottorete' collegata al place di inizio regione del ramo 'B'
        subnet2 = createSubnet(net, tree.children[1], p_start2, subnet1[2])
        p_end2 = subnet2[1]

        # Creo gli ultimi archi - dalla fine delle due 'sottoreti' alla transizione comune di fine regione parallela al place di fine regione parallela
        petri_utils.add_arc_from_to(p_end1, t_end, net)
        petri_utils.add_arc_from_to(p_end2, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, subnet2[2]]

def createClause(tree, open_clauses, end_clauses, region_counter):
    """
        Create open and close clause of tree's nodes

        Parameters
        ----------
        tree
            BPMN converted
        open_clauses
            dict --> key: node, value: list of clauses
        end_clauses
            dict --> key: node, value: list of clauses
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
        region_counter
            Keeps the counter for the regions of the Petri Net
    """
    type = tree.data

    if type == "task":
        task = tree.children[0].value  # Prendo il nome della task in per esempio: [Tree('task', [Token('NAME', 'T1')])
        start = "start_" + task
        end = "end_" + task
        open_clauses[task] = [{start}] #Tuple
        end_clauses[task] = [{end}]

        #Lo start è l'inizio della task, mentre l'end è la fine della task
        return [open_clauses, end_clauses, task, region_counter]

    # Inizializzo la nuova regione sequenziale
    region = "R" + str(region_counter)

    # Aggiorno il region_counter come nella funzione del createSubnet e chiamo ricorsivamente il createClause
    region_counter += 1
    open_clauses, end_clauses, nameleft, region_counter = createClause(tree.children[0], open_clauses, end_clauses, region_counter)

    # Stessa cosa per la parte a destra, come clausole passo quelle aggiornate (N.B. in teoria non dovrebbe creare problemi)
    open_clauses, end_clauses, nameright, region_counter = createClause(tree.children[1], open_clauses, end_clauses, region_counter)

    if type == "sequential":
        open_clauses[region] = open_clauses[nameleft]
        end_clauses[region] = end_clauses[nameright]

    elif type == "xor":
        open_clauses[region] = open_clauses[nameleft] + open_clauses[nameright]
        end_clauses[region] = end_clauses[nameleft] + end_clauses[nameright]

    elif type == "parallel":
        open_clauses[region] = open_clauses[nameleft] + open_clauses[nameright]

        end_clauses[region] = [
            set().union(*p)
            for p in itertools.product(end_clauses[nameleft], end_clauses[nameright])
        ]

    return open_clauses, end_clauses, region, region_counter

'''
def visitePetriNetTransitions(transitions):
    """
    FUNZIONE VECCHIA - NON IN USO
    """
    regions = ["R0"] #Aggiungo R0 di base
    tasks = []

    for transition in transitions:
        print(transition.label)
        splits = transition.label.split("_")
        if splits[0]=="start":
            if splits[1][0] == "R": #Regione
                if splits[1] not in regions:
                    regions.append(splits[1])
            else:
                if splits[1] not in tasks:
                    tasks.append(splits[1])

    return regions, tasks
'''

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
            Value depends on the type of the net ("P" if net simplified, "R" if net not simplified)

        Returns
        ----------
        trace
            Trace Generated
    """
    current_marking = initial_marking
    trace = []
    start = "start_" + check
    end = "end_" + check

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

    #print(trace)

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

    iterations = 8 #Quante task diverse
    for _ in range(iterations):
        current_string = replace_random_underscore(current_string, probabilities)

    process = replace_underscores(current_string)
    tree = PARSER.parse(process)
    print(tree)

    #Inizializzo Oggetto Petri Net
    net = PetriNet('Net')
    initial_marking = Marking()
    final_marking = Marking()
    root = petri_utils.add_place(net, "root")

    if REGION:
        netValue = createSubnetRegion(net, tree, root, 0)
    else:
        netValue = createSubnet(net, tree, root, 0)  # Avvio il processo per creare la Petri Net

    open_clauses, end_clauses, _, _ = createClause(tree, {}, {}, 0)

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
        if REGION:
            generated_traces.append(generateTrace(net, initial_marking, final_marking,"R"))
        else:
            generated_traces.append(generateTrace(net, initial_marking, final_marking,"P"))

    #Mi prendo tutte le varie regioni e tutte le task in modo tale da poter lavorare sulle tracce come preferisco
    #regions, tasks = visitePetriNetTransitions(net.transitions)
    regions, tasks = visitePetriNetClauses(open_clauses)
    open_clauses=filterClauses(open_clauses)
    end_clauses=filterClauses(end_clauses)
    #print("regions:", regions)
    #print("tasks:", tasks)

    regions.sort()
    tasks.sort()



    traceEncoded_regions, traceEncoded_tasks = getEncoding(generated_traces, regions, tasks, open_clauses,end_clauses)

    traceEncoded_regions.to_csv("regions_full.csv", index=True)
    traceEncoded_tasks.to_csv("tasks_full.csv", index=True)

    for trace in generated_traces:
        print(trace)

    '''#Creazione delle tracce effettive
    num_traces = 10

    #Definiamo i parametri richiesti dalla funzione di simulazione
    parameters = {
        "noTraces": num_traces,
        "initialTimestamp": datetime.now()
    }

    simulated_log = simulation.playout.petri_net.algorithm.apply(net,initial_marking,final_marking,parameters) #Variante di Default --> BASIC PLAYOUT (in teoria è random)

    print(simulated_log)

    print(f"Esempio Case ID: {simulated_log[0].attributes['concept:name']}")
    print(f"Timestamp Start: {simulated_log[0][0]['time:timestamp']}")
    print(f"Timestamp Seconda Attività: {simulated_log[0][1]['time:timestamp']}")

    traceEncoded_regions, traceEncoded_tasks, cancelletto = getEncoding(simulated_log, regions, tasks)'''



