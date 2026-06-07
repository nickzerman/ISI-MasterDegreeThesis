"""
Funzioni di generazione e valutazione dei modelli Transformer.
Estratte da main.ipynb per riuso nel benchmark e nei notebook variant.
"""
import torch
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.log.obj import Trace, Event, EventLog

from .trace_utils import edit_distance_weighted_levenshtein, hamming_distance
from .generator_utils import get_decoding, get_encoding


# ---------------------------------------------------------------------------
# Generazione
# ---------------------------------------------------------------------------

def generate_complete(model_task, model_time, encode_fn, decode_fn,
                      num_features, block_size_task, block_size_time,
                      max_new_tokens, device):
    """
    Genera una sequenza con il modello completo (regione+task) + TimeTransformer.

    Parameters
    ----------
    model_task      TaskTransformer allenato su token completi (regione+task)
    model_time      TimeTransformer (separated_regions=False)
    encode_fn       encode_complete: list[bit_vector] → list[int]
    decode_fn       decode_complete: list[int] → list[tuple]
    num_features    num_regions + num_tasks
    block_size_task block_size del TaskTransformer
    block_size_time block_size del TimeTransformer
    max_new_tokens  numero di token da generare
    device

    Returns
    -------
    decoded_steps   list[tuple]  ogni elemento è un bit-vector come tupla
    times           list[float]  tempo predetto per ogni step
    """
    model_task.eval()
    model_time.eval()

    context = torch.tensor(
        encode_fn([[0] * num_features]), dtype=torch.long, device=device
    ).unsqueeze(0)
    context_time = torch.tensor([0], dtype=torch.float32, device=device).unsqueeze(0)

    first_time = model_time.predict_next_time(
        idx_tasks=context, idx_times=context_time, block_size=block_size_time
    )
    context_time = torch.cat((context_time, first_time), dim=1)

    generated_ids, generated_times = [], []
    for _ in range(max_new_tokens):
        next_id = model_task.predict_next_task(idx_task=context, block_size=block_size_task)
        context = torch.cat((context, next_id), dim=1)
        next_time = model_time.predict_next_time(
            idx_tasks=context, idx_times=context_time, block_size=block_size_time
        )
        context_time = torch.cat((context_time, next_time), dim=1)
        generated_ids.append(next_id.item())
        generated_times.append(next_time.item())

    model_task.train()
    model_time.train()

    return decode_fn(generated_ids), generated_times


def generate_separated(model_task, model_region, model_time,
                       encode_tasks_fn, encode_regions_fn,
                       decode_tasks_fn, decode_regions_fn,
                       num_tasks, num_regions,
                       block_size_task, block_size_region, block_size_time,
                       max_new_tokens, device):
    """
    Genera con la pipeline separata: TaskTransformer → RegionTransformer → TimeTransformer v2.

    Returns
    -------
    task_ids        list[int]
    region_ids      list[int]
    times           list[float]
    """
    model_task.eval()
    model_region.eval()
    model_time.eval()

    ctx_task = torch.tensor(
        encode_tasks_fn([[0] * num_tasks]), dtype=torch.long, device=device
    ).unsqueeze(0)
    ctx_region = torch.tensor(
        encode_regions_fn([[0] * num_regions]), dtype=torch.long, device=device
    ).unsqueeze(0)
    ctx_time = torch.tensor([0], dtype=torch.float32, device=device).unsqueeze(0)

    task_ids, region_ids, times = [], [], []
    for _ in range(max_new_tokens):
        next_task = model_task.predict_next_task(idx_task=ctx_task, block_size=block_size_task)
        ctx_task = torch.cat((ctx_task, next_task), dim=1)
        ctx_task_aligned = ctx_task[:, 1:]

        next_region = model_region.predict_next_region(
            idx_region=ctx_region, idx_task=ctx_task_aligned, block_size=block_size_region
        )
        ctx_region = torch.cat((ctx_region, next_region), dim=1)
        ctx_region_aligned = ctx_region[:, 1:]

        next_time = model_time.predict_next_time(
            idx_tasks=ctx_task_aligned, idx_regions=ctx_region_aligned,
            idx_times=ctx_time, block_size=block_size_time
        )
        ctx_time = torch.cat((ctx_time, next_time), dim=1)

        task_ids.append(next_task.item())
        region_ids.append(next_region.item())
        times.append(next_time.item())

    model_task.train()
    model_region.train()
    model_time.train()

    return task_ids, region_ids, times


def decode_separated_to_bits(task_ids, region_ids, times,
                              decode_tasks_fn, decode_regions_fn):
    """
    Converte l'output della generazione separata in bit-vector + tempi,
    stesso formato dell'output di generate_complete.

    Returns
    -------
    bit_steps   list[tuple]  ogni elemento è r_bits + t_bits concatenati
    times       list[float]  invariato
    """
    bit_steps = []
    for t_id, r_id in zip(task_ids, region_ids):
        t_bits = tuple(int(b) for b in decode_tasks_fn([t_id])[0])
        r_bits = tuple(int(b) for b in decode_regions_fn([r_id])[0])
        bit_steps.append(r_bits + t_bits)
    return bit_steps, times


# ---------------------------------------------------------------------------
# Splitting in tracce
# ---------------------------------------------------------------------------

def split_into_traces(decoded_steps, num_features):
    """
    Divide una sequenza piatta di bit-vector in tracce separate.
    La separazione avviene quando si incontra il vettore zero (PAD).

    Parameters
    ----------
    decoded_steps   list[tuple|list]  output di decode_fn o decode_separated_to_bits
    num_features    num_regions + num_tasks

    Returns
    -------
    traces          list[list[list[int]]]  lista di tracce, ognuna lista di step
    """
    pad = [0] * num_features
    traces, current = [], []
    for step in decoded_steps:
        bit_list = [int(b) for b in step]
        current.append(bit_list)
        if bit_list == pad:
            traces.append(current)
            current = []
    if current:
        traces.append(current)
    if traces:
        traces.pop(0)  # rimuove la prima traccia (artefatto dell'inizializzazione con PAD)
    return traces


# ---------------------------------------------------------------------------
# Allineamento al modello di processo
# ---------------------------------------------------------------------------

_SILENT_PREFIXES = (
    'start_P', 'start_X', 'start_L',
    'end_P', 'end_X',
    'back_L', 'end_L',
)


def align_traces(traces_generated, net, regions, tasks):
    """
    Esegue il conformance checking (alignment) delle tracce generate
    rispetto alla Petri Net, usando cost function personalizzata.

    Parameters
    ----------
    traces_generated    list[list[list[int]]]  output di split_into_traces
    net                 PetriNetP
    regions             list[str]
    tasks               list[str]

    Returns
    -------
    aligned_traces      output grezzo di pm4py alignments.apply
    traces_decoded      list[list[str]]  tracce decodificate passate all'allineamento
    """
    traces_decoded = get_decoding(traces_generated, regions, tasks)

    event_log = EventLog()
    for trace in traces_decoded:
        t = Trace()
        for activity in trace:
            t.append(Event({'concept:name': activity}))
        event_log.append(t)

    model_cost, sync_cost = {}, {}
    for trans in net.net.transitions:
        is_silent = (
            trans.label is None
            or (trans.label is not None and trans.label.startswith(_SILENT_PREFIXES))
        )
        if is_silent:
            model_cost[trans] = 0
            sync_cost[trans] = 10000
        else:
            model_cost[trans] = 10000
            sync_cost[trans] = 0

    params = {
        alignments.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost,
        alignments.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost,
    }
    aligned = alignments.apply(
        event_log, net.net, net.initial_marking, net.final_marking, parameters=params
    )
    return aligned, traces_decoded


def clean_aligned_traces(aligned_traces):
    """
    Rimuove le transizioni silenziose e i move-on-log ('>>') dall'allineamento.

    Returns
    -------
    cleaned     list[list[str]]  una lista di step per ogni traccia
    """
    cleaned = []
    for a_trace in aligned_traces:
        trace_cleaned = []
        for _, net_step in a_trace['alignment']:
            if net_step is None or net_step == '>>':
                continue
            if net_step.startswith(_SILENT_PREFIXES):
                continue
            trace_cleaned.append(net_step)
        cleaned.append(trace_cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Valutazione (edit distance)
# ---------------------------------------------------------------------------

def encode_cleaned_traces(aligned_traces_cleaned, net):
    """
    Codifica le tracce allineate e pulite in bit-vector.

    Returns
    -------
    aligned_encoded     list[list[list[int]]]  stessa struttura di split_into_traces
    """
    df_regions, df_tasks = get_encoding(
        aligned_traces_cleaned, net.regions, net.tasks,
        net.open_clauses, net.end_clauses
    )
    import pandas as pd
    df = pd.concat([df_regions, df_tasks], axis=0)

    num_features = len(net.regions) + len(net.tasks)
    pad = [0] * num_features

    encoded_traces, current = [], []
    for element in df.T.values:
        bit = element.tolist()
        current.append(bit)
        if bit == pad:
            encoded_traces.append(current)
            current = []
    return encoded_traces


def compute_edit_distances(traces_generated, aligned_encoded):
    """
    Calcola la weighted Levenshtein distance tra ogni traccia generata
    e la corrispondente traccia allineata.

    Parameters
    ----------
    traces_generated    list[list[list[int]]]  output di split_into_traces
    aligned_encoded     list[list[list[int]]]  output di encode_cleaned_traces

    Returns
    -------
    costs   list[float]
    """
    num_features = len(traces_generated[0][0]) if traces_generated else 0
    costs = []
    for gen, aln in zip(traces_generated, aligned_encoded):
        cost = edit_distance_weighted_levenshtein(
            gen, aln, num_features, num_features, hamming_distance
        )
        costs.append(cost)
    return costs
