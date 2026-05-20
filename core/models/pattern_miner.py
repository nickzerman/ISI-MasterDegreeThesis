from anytree import NodeMixin

class NodoPattern(NodeMixin):
    def __init__(self, name, parent=None):
        super(NodoPattern, self).__init__()
        self.name = name
        self.parent = parent
        self.counter = 1

    def increaseCounter(self):
        self.counter += 1

class TracePatternMiner:
    def __init__(self):
        # Inizializziamo la radice dell'albero e il dizionario dei nodi
        self.root = NodoPattern(name="ROOT")
        self.nodes = {}

    def fit_trace(self, trace, max_depth=10):
        current_node = self.root

        cutted_trace = trace[:max_depth] # Tagliamo la traccia se necessario (prendiamo gli ultimi max_depth elementi)

        node_path = ""
        for element in cutted_trace:
            node_path += f"-{element}" # Chiave per il nodo nel dict nodes

            son = next((f for f in current_node.children if f.name == element), None) # Prendo il figlio

            if son: # Se il figlio esiste
                son.increaseCounter() # Incremento il figlio di 1
                current_node = son # Passo al figlio per vedere suo figlio e cosi via
            else: # Altrimenti se non esiste
                new_node = NodoPattern(name=element, parent=current_node) # Creo un nuovo nodo
                current_node = new_node # Lo metto come corrente (perchè lo è effettivamente)
                self.nodes[new_node] = node_path # Lo salvo nel dizionario dei nodi

    def select_patterns(self, num_traces, k):
        partitions = {
            "RESIDUALS" : {
                "numTraces" : num_traces,
                "nodes" : list(self.nodes.keys())
            }
        }

        print(partitions)

        step = 0
        while step < k-1: # Se k=2, una partizione di base la ho già, devo trovare solamente il modo di splittarlo diciamo
            check_partitions = {key: values for key, values in partitions.items() if len(values["nodes"]) > 0}

            if not check_partitions: # Se ho partizioni vuote mi fermo
                break

            biggerpartition = max(check_partitions, key=lambda x: check_partitions[x]["numTraces"])
            partition = partitions[biggerpartition]

            target = partition["numTraces"] / 2 # Trovare quello che lo divide a metà (o comunque il più vicino ad essa)
            best_node = min(
                partition["nodes"],
                key=lambda x: abs(x.counter - target),
            )

            if not best_node:
                break  # Non ci sono più nodi disponibili per spaccare

            best_node_children = list(best_node.descendants) # Prendo tutti i nodi (lui compreso)
            nodes_left = [node for node in partition["nodes"] if node not in best_node_children and node != best_node] # Li tolgo dal gruppo di prima
            counter_left = max(0, partition["numTraces"] - best_node.counter) # E aggiorno il counter del gruppo di prima

            if counter_left==0:
                del partitions[biggerpartition]
                step -= 1
            else:
                partitions[biggerpartition]["numTraces"] = counter_left
                partitions[biggerpartition]["nodes"] = nodes_left

            new_pattern = self.nodes[best_node]
            if new_pattern in partitions:
                new_pattern = f'{best_node.name}_{step}'

            partitions[new_pattern] = {
                "numTraces" : best_node.counter,
                "nodes" : best_node_children
            }

            step += 1


        for key,value in partitions.items():
            print(f"Partizione: {key} | Tracce coperte: {value['numTraces']}")

        return partitions

    def create_normalized_time_map(self, partitions):
        times_map = {}

        times = [2**i for i in range(len(partitions)+1)] # Esponenziale (con il +1 evito di assegnare 0.0 che è l'inizio della traccia)
        min_time = min(times)
        max_time = max(times)
        den = max_time - min_time if max_time > min_time else 1.0
        times = [round((value-min_time)/den, 4) for value in times]

        for i, key in enumerate (partitions.keys()):
            times_map[key] = times[i+1]

        return times_map

    def assign_time(self, trace, times_map, window=10):
        last_elements = trace[-window:]
        reversed_trace = list(reversed(last_elements))

        while len(reversed_trace) > 0:
            path = "-" + "-".join(reversed_trace)

            if path in times_map:
                return times_map[path]

            reversed_trace.pop()

        return times_map["RESIDUALS"]