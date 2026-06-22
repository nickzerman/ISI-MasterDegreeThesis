"""
Versione script di data/prepare_data.ipynb — accetta parametri da riga di comando.

Uso:
    # Albero passato come pickle (usato dal pipeline_runner)
    python data/prepare_data.py --tree-file /tmp/tree.pkl --n-traces 30000

    # Albero passato come stringa SESE
    python data/prepare_data.py --sese-string "seq(xor(T1,T2),loop(T3))" --n-traces 30000

    # Senza clustering
    python data/prepare_data.py --tree-file /tmp/tree.pkl --no-cluster

    # Clustering con epsilon manuale
    python data/prepare_data.py --tree-file /tmp/tree.pkl --eps 30

    # Clustering con auto-epsilon (trova il più grande eps che dà ≥ N cluster)
    python data/prepare_data.py --tree-file /tmp/tree.pkl --min-clusters 10

Parametri albero:
    --tree-file           File .pkl con lark.Tree già costruito (usato dal pipeline_runner)
    --sese-string         Stringa SESE da parsare con PARSER

Parametri generazione:
    --n-traces            Tracce generate dal Generator            (default: 30000)
    --coverage            Frazione regioni XOR/Loop classificate   (default: 0.5)
    --no-loop             Disabilita classificatori loop
    --no-xor              Disabilita classificatori XOR
    --no-interval         Genera tracce senza intervallo (start e end consecutive)

Parametri clustering:
    --no-cluster          Salta clustering e bilanciamento (usa tutte le tracce)
    --eps                 Epsilon manuale per DBSCAN        (default: 25)
    --min-clusters        Auto-epsilon: cerca il più grande eps con ≥ N cluster
                          (alternativo a --eps)

Parametri dataset:
    --num-trace-per-cluster  Tracce per cluster nel bilanciamento  (default: 1000)
    --k-partition            Pattern per il TracePatternMiner      (default: 3)
    --max-depth              Profondità massima pattern miner       (default: 5)
    --window                 Finestra contestuale per assign_time  (default: 5)
    --output                 Percorso file .pt di output
"""

import argparse
import pickle
import random
import numpy as np
import torch
import pandas as pd
from collections import Counter
from sklearn import tree as sktree
from sklearn.cluster import DBSCAN
from pm4py import save_vis_petri_net
from joblib import Parallel, delayed

from config import DATA_DIR, SEED
from core import *
from utils import *


DBSCAN_THRESHOLD = 1000  # sotto → DBSCAN esatto, sopra → CLARA K-Medoids
CLARA_M          = 1000  # dimensione campione CLARA
CLARA_N_WORKERS  = 12    # core paralleli per le distanze Levenshtein


def _compute_nearest(trace, medoid_traces, num_features):
    return int(np.argmin([
        edit_distance_weighted_levenshtein(trace, m, num_features, num_features, hamming_distance)
        for m in medoid_traces
    ]))


def _pam_on_sample(dist_matrix: np.ndarray, k: int) -> list:
    """K-Medoids (PAM) su matrice precomputata M×M. Ritorna indici dei medoidi."""
    M = len(dist_matrix)

    # BUILD: primo medoide = quello con distanza totale minima
    medoids     = [int(np.argmin(dist_matrix.sum(axis=1)))]
    non_medoids = [i for i in range(M) if i != medoids[0]]
    for _ in range(k - 1):
        min_dists = dist_matrix[:, medoids].min(axis=1)
        best_gain, best_o = -np.inf, None
        for o in non_medoids:
            gain = float(np.maximum(0.0, min_dists - dist_matrix[:, o]).sum())
            if gain > best_gain:
                best_gain, best_o = gain, o
        medoids.append(best_o)
        non_medoids.remove(best_o)

    # SWAP: finché c'è miglioramento
    improved = True
    while improved:
        improved     = False
        current_cost = dist_matrix[:, medoids].min(axis=1).sum()
        for mi in range(k):
            for o in [x for x in range(M) if x not in medoids]:
                trial    = medoids[:]
                trial[mi] = o
                new_cost = dist_matrix[:, trial].min(axis=1).sum()
                if new_cost < current_cost - 1e-9:
                    medoids, current_cost, improved = trial, new_cost, True
                    break
            if improved:
                break

    return medoids


def clara_kmedoids(traces_encoded, n_unique, num_features, k,
                   M=CLARA_M, n_workers=CLARA_N_WORKERS):
    """CLARA: campiona M tracce, K-Medoids (PAM) sul campione, assegna tutte le tracce."""
    M_actual      = min(M, n_unique)
    sample_idx    = np.random.choice(n_unique, size=M_actual, replace=False)
    sample_traces = [traces_encoded[i] for i in sample_idx]

    dist_sample   = create_distance_matrix(M_actual, sample_traces, num_features, n_workers=n_workers)
    medoid_idx    = _pam_on_sample(dist_sample, k)
    medoid_traces = [sample_traces[i] for i in medoid_idx]

    labels = np.array(
        Parallel(n_jobs=n_workers)(
            delayed(_compute_nearest)(t, medoid_traces, num_features)
            for t in traces_encoded
        )
    )
    print(f"CLARA: {k} cluster, M={M_actual}, n_unique={n_unique}")
    return labels


def parse_args():
    p = argparse.ArgumentParser()

    # Albero — priorità: tree-file > sese-string
    p.add_argument("--tree-file",             type=str,   default=None,
                   help="Percorso a file .pkl contenente un oggetto lark.Tree già costruito.")
    p.add_argument("--sese-string",           type=str,   default=None,
                   help="Stringa SESE da parsare con PARSER.")

    # Generazione tracce
    p.add_argument("--coverage",              type=float, default=0.5)
    p.add_argument("--n-traces",              type=int,   default=30000)
    p.add_argument("--no-loop",               action="store_true")
    p.add_argument("--no-xor",                action="store_true")
    p.add_argument("--no-interval",           action="store_true",
                   help="Genera tracce senza intervallo (start e end task consecutive).")

    # Clustering
    p.add_argument("--no-cluster",            action="store_true",
                   help="Salta distanza+clustering: usa tutte le tracce senza bilanciamento.")
    p.add_argument("--eps",                   type=float, default=None,
                   help="Epsilon per DBSCAN. Default=25 se --min-clusters non è passato.")
    p.add_argument("--min-clusters",          type=int,   default=None,
                   help="Auto-epsilon: trova il più grande eps che dà ancora ≥ N cluster.")

    # Dataset
    p.add_argument("--num-trace-per-cluster", type=int,   default=1000)
    p.add_argument("--k-partition",           type=int,   default=3)
    p.add_argument("--max-depth",             type=int,   default=5)
    p.add_argument("--window",                type=int,   default=5)
    p.add_argument("--output",                type=str,   default=str(DATA_DIR / "prepared_data.pt"))

    return p.parse_args()


def find_epsilon(distance_matrix: np.ndarray, trace_weights, min_clusters: int) -> float:
    """
    Trova il più grande epsilon tale che DBSCAN (min_samples=1) restituisca
    almeno `min_clusters` cluster.  Usa ricerca binaria sui valori distanza unici.
    """
    # Candidati: tutti i valori distanza unici, ordinati crescenti
    candidates = np.unique(distance_matrix)
    candidates = candidates[candidates > 0]   # eps=0 → ogni punto è un cluster, non utile

    if len(candidates) == 0:
        return 1.0

    lo, hi = 0, len(candidates) - 1
    best_eps = candidates[0]   # fallback: eps minimo (più cluster possibile)

    while lo <= hi:
        mid = (lo + hi) // 2
        eps = candidates[mid]
        labels = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(
            distance_matrix, sample_weight=trace_weights
        ).labels_
        n_clusters = len(set(labels))

        if n_clusters >= min_clusters:
            best_eps = eps   # questo eps va bene, proviamo a salire (eps più grande)
            lo = mid + 1
        else:
            hi = mid - 1     # troppo pochi cluster, dobbiamo abbassare eps

    return float(best_eps)


def main(args):
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(f"device={device}")

    LOOP = not args.no_loop
    XOR  = not args.no_xor

    # --- Albero di processo ---
    if args.tree_file:
        with open(args.tree_file, "rb") as f:
            tree = pickle.load(f)
    elif args.sese_string:
        tree = PARSER.parse(args.sese_string)
    else:
        raise ValueError("Passa --tree-file o --sese-string.")

    #print(tree)
    complexity_nested, complexity_parallel = compute_process_complexity(tree)
    #print(f"complexity_nested={complexity_nested}  complexity_parallel={complexity_parallel}")

    # --- Rete di Petri e generazione tracce iniziali ---
    net = PetriNetP(tree)
    generator = Generator(args.n_traces, net)
    no_interval = args.no_interval
    generator.generateTrace(False, no_interval=no_interval)

    save_vis_petri_net(net.net, net.initial_marking, net.final_marking, "bpmn.png", format="png")

    # --- Classificatori XOR ---
    classifier_dict_xor = {}
    if XOR:
        xors_to_classify = random.sample(net.xor_regions, round(args.coverage * len(net.xor_regions)))
        for xor in xors_to_classify:
            result = create_xor_data(xor, generator.generatedTraces, net.net)
            if result is None:
                continue
            x, y, dict_loop_step_encoding, max_len, class_to_branch = result
            dt = sktree.DecisionTreeClassifier(max_depth=5, class_weight="balanced")
            dt.fit(x, y)
            classifier_dict_xor[xor] = (dt, dict_loop_step_encoding, max_len, class_to_branch)
        print(f"XOR classificati: {list(classifier_dict_xor.keys())} / {net.xor_regions}")

    # --- Classificatori Loop ---
    classifier_dict_loop = {}
    if LOOP:
        loops_to_classify = random.sample(net.loop_regions, round(args.coverage * len(net.loop_regions)))
        for loop in loops_to_classify:
            result = create_loop_data(loop, generator.generatedTraces)
            if result is None:
                continue
            x, y, dict_loop_step_encoding, max_len = result
            dt = sktree.DecisionTreeClassifier(max_depth=5, class_weight="balanced")
            dt.fit(x, y)
            classifier_dict_loop[loop] = (dt, dict_loop_step_encoding, max_len)
        print(f"Loop classificati: {list(classifier_dict_loop.keys())} / {net.loop_regions}")

    # --- Rigenerazione condizionale ---
    generator.generateTraceCond(classifier_dict_loop, classifier_dict_xor, no_interval=no_interval)

    # --- Encoding ---
    df_region_identity = pd.DataFrame.from_dict(net.node_identity, orient="index").sort_index()
    df_region_identity.columns = ["X", "+", "->", "<>"]
    #print(df_region_identity)

    df_region_children = pd.Series(net.node_children).explode()
    df_region_children = pd.crosstab(df_region_children.index, df_region_children)
    df_region_children = df_region_children.reindex(
        index=net.regions, columns=net.regions + net.tasks, fill_value=0
    ).astype(int)
    df_region_children.index.name   = None
    df_region_children.columns.name = None
    #print(df_region_children)

    traceEncoded_regions, traceEncoded_tasks = get_encoding(
        generator.generatedTraces, net.regions, net.tasks, net.open_clauses, net.end_clauses
    )
    num_regions = len([i for i in traceEncoded_regions.index if str(i).startswith("R")])
    num_tasks   = len([i for i in traceEncoded_tasks.index   if str(i).startswith("T")])
    df_traces_complete = pd.concat([traceEncoded_regions, traceEncoded_tasks], axis=0)
    #print(f"num_regions={num_regions}, num_tasks={num_tasks}, shape={df_traces_complete.shape}")

    # --- Raccolta tracce uniche ---
    all_traces     = getall_traces(num_regions + num_tasks, df_traces_complete)
    trace_counts   = Counter(all_traces)
    traces_encoded = list(trace_counts.keys())
    trace_weights  = list(trace_counts.values())
    n_unique       = len(traces_encoded)
    print(f"Tracce uniche: {n_unique}")

    # --- Clustering (opzionale) ---
    if args.no_cluster:
        balanced_traces    = all_traces
        df_traces_balanced = df_traces_complete.copy()

    else:
        num_features = num_regions + num_tasks
        if n_unique <= DBSCAN_THRESHOLD:
            # DBSCAN esatto su matrice Levenshtein completa
            distance_matrix = create_distance_matrix(
                n_unique, traces_encoded, num_features, n_workers=CLARA_N_WORKERS
            )
            if args.min_clusters is not None:
                eps = find_epsilon(distance_matrix, trace_weights, args.min_clusters)
            else:
                eps = args.eps if args.eps is not None else 25.0
            dbscan = DBSCAN(eps=eps, min_samples=1, metric="precomputed")
            dbscan.fit(distance_matrix, sample_weight=trace_weights)
            cluster_labels = dbscan.labels_
            print(f"DBSCAN: {len(set(cluster_labels))} cluster (n_unique={n_unique})")
        else:
            # CLARA K-Medoids: campione M=1000, assegnazione parallela
            k = args.min_clusters if args.min_clusters is not None else 5
            cluster_labels = clara_kmedoids(
                traces_encoded, n_unique, num_features, k=k
            )

        balanced_traces, balanced_columns = get_balance_traces_by_cluster(
            traces_encoded, cluster_labels, args.num_trace_per_cluster
        )
        df_traces_balanced = pd.DataFrame(balanced_columns).T
        df_traces_balanced.index = df_traces_complete.index
        #print(df_traces_balanced)

    # --- Vocabolari ---
    df_regions = df_traces_balanced.head(num_regions).copy()
    df_tasks   = df_traces_balanced.tail(num_tasks).copy()
    df_traces_balanced_T = df_traces_balanced.T.copy()

    unique_cols_complete = df_traces_balanced_T.drop_duplicates()
    unique_tup_complete  = [tuple(x) for x in unique_cols_complete.values]
    bit_to_id_complete   = {v: i for i, v in enumerate(unique_tup_complete)}
    id_to_bit_complete   = {i: v for i, v in enumerate(unique_tup_complete)}
    vocab_size_complete  = len(unique_cols_complete)
    encode_complete      = lambda a: [bit_to_id_complete[tuple(x)] for x in a]

    df_regions_T        = df_regions.T
    unique_cols_regions = df_regions_T.drop_duplicates()
    unique_tup_regions  = [tuple(x) for x in unique_cols_regions.values]
    bit_to_id_regions   = {v: i for i, v in enumerate(unique_tup_regions)}
    id_to_bit_regions   = {i: v for i, v in enumerate(unique_tup_regions)}
    vocab_size_regions  = len(unique_cols_regions)
    encode_regions      = lambda a: [bit_to_id_regions[tuple(x)] for x in a]

    df_tasks_T        = df_tasks.T
    unique_cols_tasks = df_tasks_T.drop_duplicates()
    unique_tup_tasks  = [tuple(x) for x in unique_cols_tasks.values]
    bit_to_id_tasks   = {v: i for i, v in enumerate(unique_tup_tasks)}
    id_to_bit_tasks   = {i: v for i, v in enumerate(unique_tup_tasks)}
    vocab_size_tasks  = len(unique_cols_tasks)
    encode_tasks      = lambda a: [bit_to_id_tasks[tuple(x)] for x in a]

    #print(f"vocab complete={vocab_size_complete}, regions={vocab_size_regions}, tasks={vocab_size_tasks}")

    # --- Pattern miner temporale ---
    traces_balanced_decoded = get_decoding(balanced_traces, net.regions, net.tasks)
    possible_tasks = [f"{prefix}_{t}" for t in net.tasks for prefix in ("start", "end")]

    tree_times_task_dict = {}
    for task in possible_tasks:
        miner = TracePatternMiner()
        miner.root.name = task
        num_traces_for_task = 0
        for trace in traces_balanced_decoded:
            for idx in [i for i, step in enumerate(trace) if step == task]:  # TUTTE le occorrenze (non solo la prima: serve per i loop)
                miner.fit_trace(list(reversed(trace[:idx])), args.max_depth)
                num_traces_for_task += 1

        if num_traces_for_task == 0 or len(miner.nodes) == 0:
            tree_times_task_dict[task] = {
                "partitions": {"RESIDUALS": {"numTraces": num_traces_for_task, "nodes": []}},
                "times_map":  {"RESIDUALS": 0.0},
            }
            continue

        partitions = miner.select_patterns(num_traces_for_task, args.k_partition)
        times_map  = miner.create_normalized_time_map(partitions)
        tree_times_task_dict[task] = {"partitions": partitions, "times_map": times_map}
        #print(task, times_map)

    # --- Assegnazione tempi ---
    times = []
    trace_times_list = []
    for trace in traces_balanced_decoded:
        this_trace_times = []
        for i, element in enumerate(trace):
            if i == 0:
                this_trace_times.append(0.0)
                times.append(0.0)
                continue
            times_map      = tree_times_task_dict[element]["times_map"]
            generated_time = assign_time(trace[:i], times_map, args.window)
            this_trace_times.append(generated_time)
            times.append(generated_time)
        #print(this_trace_times)
        trace_times_list.append(this_trace_times)
    #print(f"Time steps totali: {len(times)}")

    # --- Classificatori di regressione temporale ---
    all_unique_steps = set(e for trace in traces_balanced_decoded for e in trace)
    all_unique_steps.add("PAD")
    dict_task_step_encoding = {task: i for i, task in enumerate(all_unique_steps)}

    classifier_dict_tasks = {}
    for task in possible_tasks:
        result = create_task_data(task, traces_balanced_decoded, trace_times_list, dict_task_step_encoding)
        if result is None:
            continue
        x, y, max_len = result
        rt = sktree.DecisionTreeRegressor(max_depth=5)
        rt.fit(x, y)
        classifier_dict_tasks[task] = (rt, max_len)
        #print(f"{task} | R²={rt.score(x, y):.4f}")

    # --- Tensori ---
    data_complete = torch.tensor(encode_complete(df_traces_balanced_T.values), dtype=torch.long)
    data_regions  = torch.tensor(encode_regions(df_regions_T.values),          dtype=torch.long)
    data_tasks    = torch.tensor(encode_tasks(df_tasks_T.values),              dtype=torch.long)
    data_times    = torch.tensor(times,                                         dtype=torch.float)

    n = int(0.8 * len(df_traces_balanced_T))
    #print(f"Train={n}, Val={len(df_traces_balanced_T)-n}")
    '''print(f"data_complete={data_complete.shape}, data_regions={data_regions.shape}, "
          f"data_tasks={data_tasks.shape}, data_times={data_times.shape}")'''

    # --- Lunghezza tracce al 95° percentile (usata da Optuna per lo spazio di ricerca adattivo) ---
    # Il 95° percentile evita che outlier rari gonfino il tier e l'architettura.
    max_trace_len = int(np.percentile([len(t) for t in traces_balanced_decoded], 95)) if traces_balanced_decoded else 0

    # --- Salvataggio ---
    torch.save({
        "data_complete":           data_complete,
        "data_regions":            data_regions,
        "data_tasks":              data_tasks,
        "data_times":              data_times,
        "n":                       n,
        "vocab_size_complete":     vocab_size_complete,
        "vocab_size_regions":      vocab_size_regions,
        "vocab_size_tasks":        vocab_size_tasks,
        "num_regions":             num_regions,
        "num_tasks":               num_tasks,
        "max_trace_len":           max_trace_len,
        "bit_to_id_complete":      bit_to_id_complete,
        "id_to_bit_complete":      id_to_bit_complete,
        "bit_to_id_regions":       bit_to_id_regions,
        "id_to_bit_regions":       id_to_bit_regions,
        "bit_to_id_tasks":         bit_to_id_tasks,
        "id_to_bit_tasks":         id_to_bit_tasks,
        "net":                     net,
        "regions":                 net.regions,
        "tasks":                   net.tasks,
        "classifier_dict_tasks":   classifier_dict_tasks,
        "dict_task_step_encoding": dict_task_step_encoding,
    }, args.output)
    #print(f"Salvato in {args.output}")

if __name__ == "__main__":
    main(parse_args())
