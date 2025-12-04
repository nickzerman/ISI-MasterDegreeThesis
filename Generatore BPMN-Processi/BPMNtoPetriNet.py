from datetime import datetime
import pandas as pd
import itertools

from pm4py.objects.petri_net.utils.petri_utils import remove_place, get_transition_by_name

from random_diagram_generation import SEED_STRING, replace_random_underscore, replace_underscores
from sese_diagram import PARSER, print_sese_diagram, print_tree, dot_tree
from stats import max_nested_xor, max_independent_xor
from lark import Lark, Tree, Token
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py import view_petri_net, save_vis_petri_net

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo import simulation

def createSubnet(net, tree, parent_place, region_counter):
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
        subnet1 = createSubnet(net, tree.children[0], p_start, region_counter)
        p_end1 = subnet1[1]
        region_counter = subnet1[2]

        #Secondo 'sottorete' collegata al place di fine 'sottorete' precedente
        subnet2 = createSubnet(net, tree.children[1], p_end1, region_counter)
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
        subnet1 = createSubnet(net, tree.children[0], p_start, region_counter)
        region_counter = subnet1[2]

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        name = subnet1[1].name
        t_end1 = get_transition_by_name(net, name)
        remove_place(net, subnet1[1])

        # Secondo 'sottorete' collegata al place di inizio regione
        subnet2 = createSubnet(net, tree.children[1], p_start, region_counter)
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
        subnet1 = createSubnet(net, tree.children[0], p_start1, region_counter)
        p_end1 = subnet1[1]
        region_counter = subnet1[2]

        # Secondo 'sottorete' collegata al place di inizio regione del ramo 'B'
        subnet2 = createSubnet(net, tree.children[1], p_start2, region_counter)
        p_end2 = subnet2[1]
        region_counter = subnet2[2]

        # Creo gli ultimi archi - dalla fine delle due 'sottoreti' alla transizione comune di fine regione parallela al place di fine regione parallela
        petri_utils.add_arc_from_to(p_end1, t_end, net)
        petri_utils.add_arc_from_to(p_end2, t_end, net)
        petri_utils.add_arc_from_to(t_end, p_end, net)

        return [parent_place, p_end, region_counter]

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
        open_clauses[task] = {(start,)} #Tuple
        end_clauses[task] = {(end,)}

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
        open_clauses[region] = open_clauses[nameleft] | open_clauses[nameright]
        end_clauses[region] = end_clauses[nameleft] | end_clauses[nameright]

    elif type == "parallel":
        open_clauses[region] = open_clauses[nameleft] | open_clauses[nameright]
        product = itertools.product(end_clauses[nameleft], end_clauses[nameright])
        end_clauses[region] = {tuple(itertools.chain.from_iterable(p)) for p in product} #

    return open_clauses, end_clauses, region, region_counter

def visitePetriNetTransitions(transitions):
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

def visitePetriNetClauses(clauses):
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

def getEncoding(log,regions,tasks):
    traceEncoded_regions = {r: [] for r in regions}
    traceEncoded_tasks = {t: [] for t in tasks}
    cancelletto = []
    for trace in log:

        for event in trace:
            operation = event["concept:name"]
            ciao = "ciao"

    return traceEncoded_regions, traceEncoded_tasks, cancelletto


if __name__ == "__main__":
    current_string = SEED_STRING
    probabilities = 0.5,0.5,0 #xor, parallel, seq

    iterations = 4 #Quante task diverse
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
    region_counter = 0
    netValue = createSubnet(net, tree, root, region_counter) #Avvio il processo per creare la Petri Net
    open_clauses, end_clauses, _, _ = createClause(tree, {}, {}, region_counter)
    #print(open_clauses)
    #print(end_clauses)

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

    #Mi prendo tutte le varie regioni e tutte le task in modo tale da poter lavorare sulle tracce come preferisco
    #regions, tasks = visitePetriNetTransitions(net.transitions)
    regions, tasks = visitePetriNetClauses(open_clauses)
    #print("regions:", regions)
    #print("tasks:", tasks)

    #Creazione delle tracce effettive
    num_traces = 10

    #Definiamo i parametri richiesti dalla funzione di simulazione
    parameters = {
        "noTraces": num_traces,
        "initialTimestamp": datetime.now()
    }

    simulated_log = simulation.playout.petri_net.algorithm.apply(net,initial_marking,final_marking,parameters) #Variante di Default --> BASIC PLAYOUT (in teoria è random)

    '''print(simulated_log)

    print(f"Esempio Case ID: {simulated_log[0].attributes['concept:name']}")
    print(f"Timestamp Start: {simulated_log[0][0]['time:timestamp']}")
    print(f"Timestamp Seconda Attività: {simulated_log[0][1]['time:timestamp']}")'''

    traceEncoded_regions, traceEncoded_tasks, cancelletto = getEncoding(simulated_log, regions, tasks)



