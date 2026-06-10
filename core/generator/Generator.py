from pm4py.objects.petri_net.semantics import enabled_transitions, execute
import random
from core import PetriNetP

def removeBackLoop(trace):
    """
        Remove back_Ln from trace

        Parameters
        ----------
            trace in input

        Returns:
        ----------
            cleaned trace
        """
    return [step for step in trace if not step.startswith("back_L")]

def removeEndLoop(trace):
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


def generateTrace(net, initial_marking, final_marking, check, clean):
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
        clean
            clean flag --> if yes trace will be cleand with check else none

        Returns
        ----------
        trace
            Trace Generated
    """
    current_marking = initial_marking
    trace = []
    start = tuple(["start_" + c for c in check])
    end = tuple(["end_" + c for c in check if c!="L"]) # si escludono gli end loop
    loop = tuple(["end_L"]) + tuple(["back_L"])

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

    if clean:
        deletion = start + end + tuple(["start_X"]) # back_L ed end_L mi servono per fare il decision tree (end_L anche successivamente per fare la codifica in bit)
        trace_cleaned = []
        for t in trace:
            if not (t.startswith(deletion)):
                trace_cleaned.append(t)

        return trace_cleaned

    return trace

def generateTraceCond(net, initial_marking, final_marking, check, classifier_dict_loop, classifier_dict_xor, limit=100):
    """
        Generate a trace of Petri Net with decision tree on LOOP region.

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
        classifier_dict_loop
            dict of list value ("loop_region" : {"model": clf1, "encoding": enc1, "pad": 10} ...)
        classifier_dict_xor
            dict of list value ("xor_region" : {"model": clf1, "encoding": enc1, "pad": 10} ...)
        Returns
        ----------
        trace
            Trace Generated
    """
    current_marking = initial_marking
    trace = []
    start = tuple(["start_" + c for c in check])
    end = tuple(["end_" + c for c in check if c!="L"]) # si escludono gli end loop
    loop = tuple(["end_L"]) + tuple(["back_L"])

    previous_step = ""

    while current_marking != final_marking:  # Fino a quando il marking non corrisponde al finale, quindi fino a quando non ho finito di generare la traccia
        if len(trace) > limit:
            return None

        transitions = enabled_transitions(net,current_marking)  # Lista delle possibili transizioni possibili al current marking

        loops_end = [item for item in transitions if item.name.startswith(loop)] # Se sono alla fine di un loop
        if loops_end:
            loop_choice = random.choice(loops_end)
            loop_region = loop_choice.name.split("_")[1]

            if loop_region not in classifier_dict_loop:
                choice = loop_choice
            else:
                clf, encoding_clf, pad_clf = classifier_dict_loop[loop_region]

                reversed_trace = trace[::-1][:pad_clf] # Solo gli ultimi pad_clf elementi (in teoria dovrebbe andare bene)
                padded_trace = reversed_trace + ["PAD"] * (pad_clf - len(reversed_trace))

                # Usiamo .get(step, 0) per gestire attività sconosciute o PAD --> NON lo metto perchè NON dovrebbe servire in realtà
                encoded_trace = [encoding_clf[step] for step in padded_trace]

                pred_class = clf.predict([encoded_trace])[0]

                if pred_class == 1:
                    choice = next((t for t in loops_end if t.name.startswith("end_" + loop_region)), None)
                else:
                    choice = next((t for t in loops_end if t.name.startswith("back_" + loop_region)), None)

            trace.append(choice.name)
            current_marking = execute(choice, net, current_marking)
        elif previous_step.startswith("start_X"):
            xor_region = previous_step.split("_")[1]

            if xor_region not in classifier_dict_xor:
                choice = random.choice(list(transitions))
            else:
                clf, encoding_clf, pad_clf, class_to_branch = classifier_dict_xor[xor_region]

                if clf.get_depth() == 0: # Se non ho split (evito di chiudere la porta in faccia ad una parte di processo)
                    proba = clf.predict_proba([[0] * pad_clf])[0] # Prendo le probabilità
                    pred_class = random.choices(clf.classes_, weights=proba, k=1)[0] # E tiro a caso con quelle probabilità
                    branch = class_to_branch[pred_class]
                    choice = next((t for t in transitions if t.name.startswith(branch)), None)
                else: # Altrimenti guardo la predizione del classificato
                    reversed_trace = trace[:-1][::-1][:pad_clf]  # Solo gli ultimi pad_clf elementi (in teoria dovrebbe andare bene, taglio anche l'ultimo step perchè coincide con lo start dello xor)
                    padded_trace = reversed_trace + ["PAD"] * (pad_clf - len(reversed_trace))

                    # Usiamo .get(step, 0) per gestire attività sconosciute o PAD --> NON lo metto perchè NON dovrebbe servire in realtà
                    encoded_trace = [encoding_clf[step] for step in padded_trace]

                    pred_class = clf.predict([encoded_trace])[0]
                    branch = class_to_branch[pred_class]
                    choice = next((t for t in transitions if t.name.startswith(branch)), None)

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
        previous_step = choice.name

    # Pulisco la traccia
    deletion = start + end + tuple(["back_L"]) # end_L mi serve per la codifica successiva
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

    def generateTrace(self, clean):
        self.generatedTraces = []
        for i in range(self.num_traces):
            self.generatedTraces.append(
                generateTrace(self.net.net, self.net.initial_marking, self.net.final_marking, ["P", "X", "L"], clean))
        return self.generatedTraces

    def generateTraceCond(self, classifier_dict_loop, classifier_dict_xor):
        self.generatedTraces = []

        # Uso un while per assicurarmi di avere esattamente 'num_traces' tracce valide
        while len(self.generatedTraces) < self.num_traces:
            newTrace = generateTraceCond(self.net.net, self.net.initial_marking, self.net.final_marking,
                                         ["P", "X", "L"], classifier_dict_loop, classifier_dict_xor)

            if newTrace is not None:
                self.generatedTraces.append(newTrace)

        return self.generatedTraces

