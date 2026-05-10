from pm4py.objects.petri_net.utils.petri_utils import remove_place
from pm4py.objects.petri_net.utils import petri_utils
import itertools
from pm4py.objects.petri_net.obj import PetriNet, Marking

class PetriNetP:
    def __init__(self, tree): #Build object from tree
        """
        Args:
            tree: initial tree from which we create PetriNetP
            net: net
            initial_marking: starting marking
            final_marking: end marking
            root: root
            open_clauses: open_clauses for each tasks/regions (only for regions after filter)
            end_clauses: same as open but with end_clauses
            node_identity: identity of regions
            node_children: children of each region (can be tasks and/or regions)
            regions: regions of the PetriNetP
            tasks: tasks of the PetriNetP
        """
        self.tree = tree

        self.net = PetriNet('Net')
        self.initial_marking = Marking()
        self.final_marking = Marking()
        self.loop_regions = []
        self.tasks = []
        self.root = petri_utils.add_place(self.net, "root")

        netValue = self.createSubnet(self.net, self.tree, self.root, 0)
        self.initial_marking[netValue[0]] = 1
        self.final_marking[netValue[1]] = 1

        self.open_clauses, self.end_clauses, _, self.node_identity, self.node_children, _ = self.createClause(self.tree,{},{},{},{},0)

        self.regions, self.tasks = self.visitePetriNetClauses(self.open_clauses)
        self.open_clauses = self.filterClauses(self.open_clauses)
        self.end_clauses = self.filterClauses(self.end_clauses)

        self.regions.sort()
        self.tasks.sort()

    def createSubnet(self, net, tree, parent_place, counter):
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
            self.tasks.append(task)

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
            counter += 1
            for child in tree.children:  # Creo la sottorete di ogni figlio e le collego in maniera sequenziale
                subnet = self.createSubnet(net, child, p_start, counter)
                p_start = subnet[1]  # La fine della sottorete è l'inizio della sottorete successiva (o comunque l'end del sequenziale che ritornerò)
                counter = subnet[2]  # Aggiorno il counter

            return [parent_place, p_start, counter]

        if tree.data == 'xor':
            start = "start_X" + str(counter)
            inter = "inter_X" + str(counter)
            end = "end_X" + str(counter)
            t_start = petri_utils.add_transition(net, start, start)
            t_end = petri_utils.add_transition(net, end, end)
            p_start = petri_utils.add_place(net, start)
            p_inter = petri_utils.add_place(net, inter)
            p_end = petri_utils.add_place(net, end)  # Per collegare la fine dello xor
            counter += 1

            petri_utils.add_arc_from_to(parent_place, t_start, net)
            petri_utils.add_arc_from_to(t_start, p_start, net)

            t_end_c = []  # Salvo la transizione finale di ogni figlio che poi andrò a collegare al p_end
            for child in tree.children:
                subnet = self.createSubnet(net, child, p_start, counter)  # Il padre è sempre parent_place

                in_arcs = subnet[1].in_arcs
                for arc in in_arcs:
                    t_end_c.append(arc.source)
                remove_place(net, subnet[1])  # Rimuovo il place finale (per semplificare la petri net, collego tutto a p_end dello xor)

                counter = subnet[2]  # Aggiorno il counter

            for arc in t_end_c:  # Collego tutte le transizioni dei figli a p_end
                petri_utils.add_arc_from_to(arc, p_inter, net)
            petri_utils.add_arc_from_to(p_inter, t_end, net)
            petri_utils.add_arc_from_to(t_end, p_end, net)

            return [parent_place, p_end, counter]

        if tree.data == 'parallel':
            start = "start_P" + str(counter)  # Creo start e end per transizioni e place della regione parallela
            end = "end_P" + str(counter)
            t_start = petri_utils.add_transition(net, start, start)
            t_end = petri_utils.add_transition(net, end, end)

            p_start = {}
            for child in tree.children:
                p_start_c = petri_utils.add_place(net, start)
                p_start[child] = p_start_c

            p_end = petri_utils.add_place(net, end)
            counter += 1

            petri_utils.add_arc_from_to(parent_place, t_start, net)  # Transizione di split per la regione parallela

            p_end_c = []  # Mi salvo la fine di ogni figlio, che poi collegherò alla transizione di fine regione parallela
            for child in tree.children:
                petri_utils.add_arc_from_to(t_start, p_start[child],net)  # Creo l'arco di inizio collegato alla transizione di split iniziale
                subnet = self.createSubnet(net, child, p_start[child], counter)
                p_end_c.append(subnet[1])
                counter = subnet[2]  # Aggiorno il counter

            for p in p_end_c:  # Per ogni place di fine figlio, lo collego alla transizione di fine regione parallela
                petri_utils.add_arc_from_to(p, t_end, net)

            petri_utils.add_arc_from_to(t_end, p_end, net)  # Collegamento transizione finale - place finale di regione

            return [parent_place, p_end, counter]

        if tree.data == 'loop':
            self.loop_regions.append("L" + str(counter))
            start = "start_L" + str(counter)
            end = "end_L" + str(counter)
            back = "back_L" + str(counter)
            t_start = petri_utils.add_transition(net, start, start)
            t_end = petri_utils.add_transition(net, end, end)
            t_back = petri_utils.add_transition(net, back, back)
            p_start = petri_utils.add_place(net, start)
            p_end = petri_utils.add_place(net, end)
            counter += 1

            petri_utils.add_arc_from_to(parent_place, t_start, net)
            petri_utils.add_arc_from_to(t_start, p_start, net)

            subnet = self.createSubnet(net, tree.children[0], p_start, counter) #I loop hanno sempre e solo un figlio (per come è stato pensato)
            p_end_subnet = subnet[1]  # La fine della sottorete è l'inizio della sottorete successiva (o comunque l'end del sequenziale che ritornerò)
            counter = subnet[2]  # Aggiorno il counter

            petri_utils.add_arc_from_to(p_end_subnet, t_back, net)
            petri_utils.add_arc_from_to(t_back, p_start, net)
            petri_utils.add_arc_from_to(p_end_subnet, t_end, net)
            petri_utils.add_arc_from_to(t_end, p_end, net)

            return [parent_place, p_end, counter]


    def createClause(self, tree, open_clauses, end_clauses, node_identity, node_children, region_counter):
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
                dict --> key: node, value: list [x,+,->,<>] where we have 1 if the region is that type
            node_children
                dict --> key: node, value: list of nodes
            region_counter
                Keeps the counter for the regions of the Petri Net
        """

        if tree.data == "task":
            task = tree.children[0].value  # Prendo il nome della task in per esempio: [Tree('task', [Token('NAME', 'T1')])
            start = "start_" + task
            end = "end_" + task
            open_clauses[task] = [{start}]  # Tuple
            end_clauses[task] = [{end}]

            # Lo start è l'inizio della task, mentre l'end è la fine della task
            return [open_clauses, end_clauses, task, node_identity, node_children, region_counter]

        # Inizializzo la nuova regione
        region = "R" + str(region_counter)

        # Aggiorno il region_counter come nella funzione del createSubnet e chiamo ricorsivamente il createClause
        region_counter += 1
        children = []  # Tengo traccia dei figli, in questo modo facilmente creo open_clauses ed end_clauses
        for child in tree.children:
            open_clauses, end_clauses, name, node_identity, node_children, region_counter = self.createClause(child,open_clauses,end_clauses,node_identity,node_children,region_counter)
            children.append(name)

        node_children[region] = children  # Per ogni regione, ho la lista solamente dei figli

        if tree.data == "sequential":
            node_identity[region] = [0, 0, 1, 0]  # Identificazione del tipo di regione

            open_clauses[region] = open_clauses[children[0]]  # Prendo solo il primo figlio, che aprirà la regione sequenziale
            end_clauses[region] = end_clauses[children[-1]]  # Prendo solo l'ultimo figlio, che chiuderà la regione sequenziale

        elif tree.data == "xor":
            node_identity[region] = [1, 0, 0, 0]  # Identificazione del tipo di regione

            # Per ogni figlio, le sue clauses open e di end valgono per lo xor
            open_clauses_current = []
            end_clauses_current = []
            for child in children:
                open_clauses_current += open_clauses[child]
                end_clauses_current += end_clauses[child]
            open_clauses[region] = open_clauses_current
            end_clauses[region] = end_clauses_current

        elif tree.data == "parallel":
            node_identity[region] = [0, 1, 0, 0]  # Identificazione del tipo di regione

            # Per ogni figlio, le sue clauses open valgono per il parallelo
            open_clauses_current = []
            for child in children:
                open_clauses_current += open_clauses[child]
            open_clauses[region] = open_clauses_current

            # Mentre devo fare le combinazioni tra le varie end clauses dei figli
            all_children_end_clauses = [end_clauses[child] for child in children]

            # L'asterisco * scompatta la lista in argomenti separati per product
            end_clauses[region] = [
                set().union(*combination)
                for combination in itertools.product(*all_children_end_clauses)
            ]

        elif tree.data == "loop":
            node_identity[region] = [0, 0, 0, 1]  # Identificazione del tipo di regione

            # Loop ha solo un figlio per definizione
            open_clauses[region] = open_clauses[children[0]]
            end_clauses[region] = [{f"end_L{region[1:]}"}]

        return open_clauses, end_clauses, region, node_identity, node_children, region_counter

    def visitePetriNetClauses(self,clauses):
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
        regions = []  # Aggiungo R0 di base
        tasks = []

        for label in clauses.keys():
            if label[0] == "R":  # Regione
                if label not in regions:
                    regions.append(label)
            else:
                if label not in tasks:
                    tasks.append(label)

        return regions, tasks

    def filterClauses(self,clauses):
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
        clauses_filtered = {k: v for k, v in clauses.items() if not k.startswith("T")}
        return clauses_filtered

