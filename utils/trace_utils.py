import numpy as np
import random
import os
from joblib import Parallel, delayed

def hamming_distance(t1, t2):
    return sum(a != b for a, b in zip(t1, t2))

def clean_trace(trace):
    check = ["P", "X", "L"]

    start = tuple(["start_" + c for c in check])
    end = tuple(["end_" + c for c in check if c!="L"]) # si escludono gli end loop

    # Pulisco la traccia
    deletion = start + end
    trace_cleaned = []
    for t in trace:
        if not (t.startswith(deletion)):
            trace_cleaned.append(t)

    return trace_cleaned

def edit_distance_weighted_levenshtein(s1, s2, cost_ins, cost_del, cost_fn):
    """
    Calcola la distanza di Levenshtein pesata tra due tracce

    Parameters
    ----------
    s1
        transformer's generated trace
    s2
        aligned trace
    cost_ins
        column insertion cost
    cost_del
        column deletion cost
    cost_fn
        function for column substitution cost (total cost = # of different bits between columns - hamming distance)

    Returns
    ----------
        correction cost
    """

    # Le lunghezze possono essere differenti
    n = len(s1)
    m = len(s2)

    dp = np.zeros((n + 1, m + 1)) # Matrice per algoritmo prog. dinamica

    # Casi base: trasformare una stringa in una vuota (solo eliminazioni o inserimenti)
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + cost_del
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + cost_ins

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] # Se le attività sono uguali, il costo non aumenta
            else: # Altrimenti scegli il minimo tra: cancellazione (da s1), inserimento (su s1) o sostituzione (su s1)
                dp[i][j] = min(
                    dp[i-1][j] + cost_del,    # Cancellazione (valore cella sopra + costo cancellazione)
                    dp[i][j-1] + cost_ins,    # Inserimento (valore cella sinistra + costo inserimento)
                    dp[i-1][j-1] + cost_fn(s1[i-1],s2[j-1])   # Substitution (valore cella diagonale + costo sostituzione)
                )

    return dp[n][m]

def getall_traces(num_regions, df_traces):
    """
    Returns a list of all traces

    Parameters
    ----------
    num_regions: num_regions + num_tasks
    df_traces

    Returns
    ----------
    all_traces: all traces separated

    """
    all_traces = []
    trace_encoded = []
    zero_vector = tuple([0] * num_regions)

    for element in df_traces.T.values: # Ogni elemento è una colonna di 0/1
        element = tuple(int(x) for x in element)
        trace_encoded.append(element)
        if element == zero_vector:
            all_traces.append(tuple(trace_encoded))
            trace_encoded = []

    return all_traces

def compute_row(i, trace_i, traces_tail, num_regions):
    """
    Returns a list of all traces

    Parameters
    ----------
    i: rif trace's idx
    trace_i: idx's trace
    traces_tail: traces after trace i
    num_regions: num_regions + num_tasks

    Returns
    ----------
    i: index
    costs: distance between trace i and traces_tail (taken singularly)

    """
    costs = []
    for trace_j in traces_tail:
        cost = edit_distance_weighted_levenshtein(trace_i, trace_j, num_regions, num_regions, hamming_distance)
        costs.append(cost)
    return i, costs

def create_distance_matrix(n_traces, traces_encoded, num_regions, n_workers=1):
    """
    Returns the distance matrix between traces (parallelized over rows).

    Parameters
    ----------
    n_traces
    traces_encoded
    num_regions: num_regions + num_tasks
    n_workers: number of parallel workers (-1 = all CPUs)

    Returns
    ----------
    distance_matrix
    """
    distance_matrix = np.zeros((n_traces, n_traces))

    # Results è una lista di (i, costs[]) per ogni riga i
    results = Parallel(n_jobs=n_workers, verbose=1)(
        delayed(compute_row)(i, traces_encoded[i], traces_encoded[i + 1:], num_regions) # produce un oggetto che rappresenta la serie di chiamate alla funzione compute_row
        for i in range(n_traces)
    )

    for i, costs in results:
        for k, cost in enumerate(costs):
            j = i + 1 + k
            distance_matrix[i][j] = cost
            distance_matrix[j][i] = cost

    return distance_matrix

def get_balance_traces_by_cluster(traces_encoded, cluster_labels, num_trace_per_cluster):
    """
    Returns a list of columns that create traces --> the number of traces that has been taken from each cluster in this list is balanced

    Parameters
    ----------
    traces_encoded
    cluster_labels
    num_trace_per_cluster: how many traces take from each cluster

    Returns
    ----------
    balanced_traces: traces like initial dataframe of traces
    balanced_columns: columns like initial dataframe of traces

    """
    variants_per_cluster = {}
    for trace, cluster_id in zip(traces_encoded, cluster_labels):
        if cluster_id not in variants_per_cluster:
            variants_per_cluster[cluster_id] = []
        variants_per_cluster[cluster_id].append(trace)

    # Genero casualmente le tracce equiparando il numero di tracce per cluster
    balanced_traces = []  # Contiene le tracce per intero (NON divise in colonne)
    for cluster_id, variants_list in variants_per_cluster.items():
        for _ in range(num_trace_per_cluster):
            choiced_trace = random.choice(variants_list)  # Pesco a caso tra le varianti
            balanced_traces.append(choiced_trace)

    random.shuffle(balanced_traces)  # Mescolo le tracce (cosi da rendere la cosa molto più casuale)

    balanced_columns = []  # Rispetta la logica del dataframe iniziale --> la traccia è espressa in più colonne (ogni traccia può avere num_colonne diverse)

    for traccia in balanced_traces:
        for step in traccia:
            balanced_columns.append(step)

    return balanced_traces, balanced_columns

def assign_time(trace, times_map, window=10):
    """

    Parameters
    ----------
    trace
    times_map
    window: how much we look back in the trace

    Returns
    -------
    time

    """
    last_elements = trace[-window:]
    reversed_trace = list(reversed(last_elements))

    while len(reversed_trace) > 0:
        path = "-" + "-".join(reversed_trace)

        if path in times_map:
            return times_map[path]

        reversed_trace.pop()

    return times_map["RESIDUALS"]