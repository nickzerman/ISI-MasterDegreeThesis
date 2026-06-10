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

def create_xor_data(xor, traces):
    """
    Create the dataset for XOR region using traces (multiclass).

    For each occurrence of start_X{n} in a trace, the prefix before it is x
    and the immediately following transition (first step of the chosen branch) determines y.

    Parameters
    ----------
    xor: xor label/name (e.g., "X0")
    traces: list of traces with XOR markers (generated with clean=False)

    Returns
    ----------
    x_processed: dataset input
    y: dataset output (class index of the branch taken)
    dict_xor_step_encoding: how steps are encoded
    max_len: padding length
    class_to_branch: dict {class_index: first_transition_name} for inference
    """
    x = []
    y = []
    xor_start = "start_" + xor  # e.g., "start_X0"

    for trace in traces:
        for i, element in enumerate(trace):
            if element == xor_start:
                x.append(trace[:i])
                y.append(trace[i+1])

    if not x:
        return None

    unique_branches = sorted(set(y)) # Quante uscite ho dallo xor
    branch_to_class = {branch: idx for idx, branch in enumerate(unique_branches)}
    class_to_branch = {idx: branch for branch, idx in branch_to_class.items()}

    y = [branch_to_class[branch] for branch in y] # Trasformo il branch in classificazione (0,1,2...)

    max_len = max(len(trace) for trace in x)
    max_len = max(max_len, 1)

    all_unique_tasks = set(element for trace in x for element in trace)
    all_unique_tasks.add("PAD")

    dict_xor_step_encoding = {task: i for i, task in enumerate(all_unique_tasks)}

    x_processed = []
    for trace in x:
        reversed_trace = trace[::-1]
        padded_trace = reversed_trace + ["PAD"] * (max_len - len(reversed_trace))
        encoded_trace = [dict_xor_step_encoding[step] for step in padded_trace]
        x_processed.append(encoded_trace)

    return x_processed, y, dict_xor_step_encoding, max_len, class_to_branch

def create_task_data(task, traces_balanced_decoded, trace_times_list, dict_task_step_encoding):
    """
    Create the dataset for every task

    Parameters
    ----------
    task: task label
    traces_balanced_decoded: list of traces balanced decoded
    trace_times_list: list of traces times balanced decoded
    dict_task_step_encoding: how the task are encoded

    Returns
    ----------
    x_processed: dataset input
    y: dataset output
    max_len: we need it to padding other traces

    """
    # Creating dataset for regression tree
    x = []
    y = []
    for trace, times_trace in zip(traces_balanced_decoded, trace_times_list):
        for i, element in enumerate(trace):
            if element == task: # Se corrisponde alla task aggiungo al dataset
                x.append(trace[:i])
                y.append(times_trace[i])

    if not x: # Se dataset vuoto esco
        return None

    max_len = max(len(trace) for trace in x)

    if max_len == 0:  # Se tutte le tracce sono = 0 (sono sempre la prima task) esco
        return None

    x_processed = []
    for trace in x: # Aggiungo il pad e codifico
        reversed_trace = trace[::-1]
        padded_trace = reversed_trace + ["PAD"] * (max_len - len(reversed_trace))
        encoded_trace = [dict_task_step_encoding[step] for step in padded_trace]
        x_processed.append(encoded_trace)

    return x_processed, y, max_len