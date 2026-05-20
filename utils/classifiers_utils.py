def create_loop_data(loop, traces):
    """
    Create the dataset for loop region using traces

    Parameters
    ----------
    loop: loop label/name
    traces

    Returns
    ----------
    x_processed: dataset input
    y: dataset output
    dict_loop_step_encoding: how the task are encoded in this loop's label
    max_len: we need it to padding other traces

    """
    x = []
    y = []
    for trace in traces: # Per ogni traccia, per ogni back o end aggiungo alla lista delle tracce per questo loop
        for i, element in enumerate(trace):
            if element.startswith("back_" + loop):
                x.append(trace[:i])
                y.append(0)
            elif element.startswith("end_" + loop):
                x.append(trace[:i])
                y.append(1)

    if not x:
        return None

    max_len = max(len(trace) for trace in x) # Prendo la massima lunghezza per permettere il padding

    all_unique_tasks = set(element for trace in x for element in trace)
    all_unique_tasks.add("PAD")  # Aggiungiamo il carattere speciale

    dict_loop_step_encoding = {task: i for i, task in enumerate(all_unique_tasks)} # Dizionario dell'encoding

    x_processed = []
    for trace in x: # Aggiungo il pad e codifico
        reversed_trace = trace[::-1]
        padded_trace = reversed_trace + ["PAD"] * (max_len - len(reversed_trace))
        encoded_trace = [dict_loop_step_encoding[step] for step in padded_trace]
        x_processed.append(encoded_trace)

    return x_processed, y, dict_loop_step_encoding, max_len