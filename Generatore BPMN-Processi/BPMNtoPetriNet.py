from datetime import datetime
import pandas as pd
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
        p_end1 = subnet1[1]
        region_counter = subnet1[2]

        # Rimuovo l'ultimo place della sottorete perchè poi andrò a collegare la transition al place di regione intermedia
        name = subnet1[1].name
        t_end1 = get_transition_by_name(net, name)
        remove_place(net, subnet1[1])

        # Secondo 'sottorete' collegata al place di inizio regione
        subnet2 = createSubnet(net, tree.children[1], p_start, region_counter)
        p_end2 = subnet2[1]
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

def visitePetriNet(transitions):
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
    probabilities = 0.34,0.33,0.33 #xor, parallel, seq

    iterations = 5 #Quante task diverse
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
    regions, tasks = visitePetriNet(net.transitions)
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



