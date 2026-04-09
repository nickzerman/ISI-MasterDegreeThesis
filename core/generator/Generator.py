from pm4py.objects.petri_net.semantics import enabled_transitions, execute
import random
from core import PetriNetP

def cleanTrace(trace):
    """
        Remove end_Ln from trace

    Parameters
    ----------
        trace in input

    Returns:
    ----------
        cleaned trace
    """
    return [step for step in trace if not step.startswith("end_L")]


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
    end = tuple(["end_" + c for c in check if c!="L"]) # si escludono gli end loop
    loop = tuple("end_L") + tuple(["back_L"])

    while current_marking != final_marking:  # Fino a quando il marking non corrisponde al finale, quindi fino a quando non ho finito di generare la traccia
        transitions = enabled_transitions(net,current_marking)  # Lista delle possibili transizioni possibili al current marking

        loops_end = [item for item in transitions if item.name.startswith(loop)]
        if loops_end:
            choice = random.choice(loops_end)
            trace.append(choice.name)
            current_marking = execute(choice, net, current_marking)
        else:
            # Prima di tutto chiudo le regioni aperte (che posso chiudere ovviamente)
            items_end = [item for item in transitions if item.name.startswith(end)]
            if items_end:
                choice = random.choice(items_end)
                trace.append(choice.name)
                current_marking = execute(choice, net, current_marking)
            else:
                # Poi apro tutte le regioni apribili MA il place da cui l'arco di input proviene deve avere solo un arco uscente, altrimenti NON POSSO dare precedenza (tipo tra regione e task darei sempre precedenza ad una regione in uno xor e NON va bene)
                items_start = [item for item in transitions if item.name.startswith(start)]
                items_start_updated = []
                for item in items_start:  # Check spiegato in precedenza
                    in_arc = list(item.in_arcs)
                    source = in_arc[0].source
                    if len(list(source.out_arcs)) == 1:
                        items_start_updated.append(item)

                if items_start_updated:  # Se c'è qualche regione che parte che posso consumare senza problemi, scelgo randomicamente da queste
                    choice = random.choice(items_start_updated)
                    trace.append(choice.name)
                    current_marking = execute(choice, net, current_marking)
                else:  # Altrimenti tiro a caso su cosa fare tra tutte le transizioni rimanenti
                    choice = random.choice(list(transitions))
                    trace.append(choice.name)
                    current_marking = execute(choice, net, current_marking)

    # Pulisco la traccia
    deletion = start + end + tuple(["back_L"]) # gli end_L mi servono negli encoding (in teoria???)
    trace_cleaned = []
    for t in trace:
        if not (t.startswith(deletion)):
            trace_cleaned.append(t)

    return trace_cleaned

class Generator:
    def __init__(self, num_traces: int, net: PetriNetP):
        self.num_traces = num_traces
        self.net = net

        self.generatedTraces = []
        for i in range(num_traces):
            self.generatedTraces.append(
                generateTrace(self.net.net, self.net.initial_marking, self.net.final_marking, ["P", "X", "L"]))

