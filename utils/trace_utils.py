import numpy as np
import pandas as pd
import random

def hamming_distance(t1, t2):
    return sum(a != b for a, b in zip(t1, t2))

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

def create_distance_matrix(n_traces, traces_encoded, num_regions):
    """
    Returns the distance matrix between traces

    Parameters
    ----------
    n_traces
    traces_encoded
    num_regions: num_regions + num_tasks

    Returns
    ----------
    distance_matrix: all traces separated

    """
    distance_matrix = np.zeros((n_traces, n_traces))
    for i in range(n_traces):
        for j in range(i + 1, n_traces):
            cost = edit_distance_weighted_levenshtein(traces_encoded[i], traces_encoded[j], num_regions,num_regions, hamming_distance)  # Utilizziamo la distanza di hamming al momento
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
    balanced_traces: columns like initial dataframe of traces

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

    return balanced_columns