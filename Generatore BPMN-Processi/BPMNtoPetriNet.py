from datetime import datetime
import pandas as pd

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
        #if not tree.children[0].data == "task": #Sto entrando in una nuova regione, quindi aumento il counter
        #    region_counter += 1
        subnet1 = createSubnet(net, tree.children[0], parent_place, region_counter)
        region_counter = subnet1[2]

        #if not tree.children[1].data == "task": #Sto entrando in una nuova regione, quindi aumento il counter
        #    region_counter += 1
        subnet2 = createSubnet(net, tree.children[1], subnet1[1], region_counter)
        region_counter = subnet2[2]

        return [parent_place, subnet2[1], region_counter]

    if type == "xor":
        region_counter += 1
        end = ""

        '''
        SE VOGLIAMO EVITARE DI METTERE IL NOME DELLE REGIONI ANCHE SE CI SONO SOLAMENTE TASK (PERO' NON RIUSCIAMO A VEDERE LE EFFETTIVE REGIONI NELLA PETRI NET
        
        #Se il figlio è un task non vado ad iniziare una nuova regione
        end = "" #Ho solo un processo di end
        if tree.children[0].data == "task":
            start1 = ""
            end1 = ""
            end += "._"
        else:
            region_counter += 1
            start1 = "start_R" + str(region_counter)
            end1 = "end_R" + str(region_counter)
            end += "end_R" + str(region_counter) + "_"'''

        start1 = "start_R" + str(region_counter)
        end1 = "end_R" + str(region_counter)
        end += "end_R" + str(region_counter)

        # Creo transizioni e place per la prima parte
        t_start1 = petri_utils.add_transition(net,start1,start1)
        t_end1 = petri_utils.add_transition(net, end1, end1)
        p_start1 = petri_utils.add_place(net,start1)

        subnet1 = createSubnet(net, tree.children[0], p_start1, region_counter)
        region_counter = subnet1[2]

        # Se il figlio è un task non vado ad iniziare una nuova regione
        '''if tree.children[1].data == "task":
            start2 = ""
            end2 = ""
            end += "."
        else:
            start2 = "start_R" + str(region_counter)
            end2 = "end_R" + str(region_counter)
            end += "end_R" + str(region_counter)'''

        start2 = "start_R" + str(region_counter)
        end2 = "end_R" + str(region_counter)
        end += "end_R" + str(region_counter)

        # Creo transizioni e place per la seconda parte
        t_start2 = petri_utils.add_transition(net, start2, start2)
        t_end2 = petri_utils.add_transition(net, end2, end2)
        p_start2 = petri_utils.add_place(net, start2)

        p_end = petri_utils.add_place(net, end)

        subnet2 = createSubnet(net, tree.children[1], p_start2, region_counter)
        region_counter = subnet2[2]

        # Creo tutti gli archi
        petri_utils.add_arc_from_to(parent_place, t_start1, net)
        petri_utils.add_arc_from_to(parent_place, t_start2, net)
        petri_utils.add_arc_from_to(t_start1, p_start1, net)
        petri_utils.add_arc_from_to(t_start2, p_start2, net)
        petri_utils.add_arc_from_to(subnet1[1], t_end1, net)
        petri_utils.add_arc_from_to(subnet2[1], t_end2, net)
        petri_utils.add_arc_from_to(t_end1, p_end, net)
        petri_utils.add_arc_from_to(t_end2, p_end, net)

        return [parent_place, p_end, region_counter]

    if type == "parallel":
        region_counter += 1

        # Creo subito transizioni e place
        start = "start_R" + str(region_counter)
        end = "end_R" + str(region_counter)
        t_start = petri_utils.add_transition(net, start, start)
        t_end = petri_utils.add_transition(net, end, end)
        p_start1 = petri_utils.add_place(net,start)
        p_start2 = petri_utils.add_place(net,end)
        p_end = petri_utils.add_place(net,end)

        subnet1 = createSubnet(net, tree.children[0], p_start1, region_counter)
        region_counter = subnet1[2]

        subnet2 = createSubnet(net, tree.children[1], p_start2, region_counter)
        region_counter = subnet2[2]

        #Creo archi
        petri_utils.add_arc_from_to(parent_place, t_start, net)
        petri_utils.add_arc_from_to(t_start, p_start1, net)
        petri_utils.add_arc_from_to(t_start, p_start2, net)
        petri_utils.add_arc_from_to(subnet1[1], t_end, net)
        petri_utils.add_arc_from_to(subnet2[1], t_end, net)
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

    iterations = 3 #Quante task diverse
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



