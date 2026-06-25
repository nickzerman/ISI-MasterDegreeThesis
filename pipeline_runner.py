"""
Orchestratore della pipeline sperimentale — 171 processi.

Per ogni processo in PROCESSES esegue in sequenza:
  1. data/prepare_data.py  (script, output in tempo reale)
  2. experiments/run_all_optuna.ipynb
  3. variants/variant*.ipynb  (tutti in parallelo)

I notebook leggono sempre data/prepared_data.pt, che viene riscritto ad ogni processo.

Uso:
    python pipeline_runner.py
    python pipeline_runner.py --stop-on-error

Se all_variants_results.json esiste in results/, riprende automaticamente dal processo
dove si era fermato (le entry parziali dell'ultimo processo vengono rimosse).
"""

import sys
import os
import html
import time
import json
import pickle
import signal
import tempfile
import threading
import subprocess
import requests
import nbformat
import papermill as pm
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from lark import Tree, Token

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR    = BASE_DIR / "data"

VRAM_MIN_FREE_MB   = 4000  # MB liberi minimi prima di ritentare un notebook OOM
VRAM_POLL_INTERVAL = 60    # secondi tra un check VRAM e l'altro
VRAM_WAIT_TIMEOUT  = 1200  # timeout massimo attesa VRAM (20 minuti)
VRAM_RETRY_MAX     = 2     # max retry per notebook che crashano con OOM

NOTEBOOK_TIMEOUT   = 1800  # timeout (s) per i notebook variante (training di UN modello)
OPTUNA_TIMEOUT     = 7200  # run_all_optuna fa 6 varianti x N trial: tetto di sicurezza 2 ore.

CHECKPOINT_FILE = RESULTS_DIR / "all_variants_results.json"

BOT_TOKEN = "8910437774:AAHqyzkmTRtet_2ktDeJ-oJPbEbfPBYHPv8"
CHAT_ID   = "654952374"

OPTUNA_PROGRESS_FILE = Path(tempfile.gettempdir()) / "optuna_pipeline_progress.json"

# ---------------------------------------------------------------------------
# 171 processi.
# ---------------------------------------------------------------------------
def get_processes():
    from lark import Tree, Token

    XOR_SEQ_5 = Tree('xor', [
        Tree('sequential', [
            Tree('xor', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')])]),
            Tree('xor', [Tree('task', [Token('NAME', 'T6')]), Tree('task', [Token('NAME', 'T7')])])
        ]),
        Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')]),
        Tree('task', [Token('NAME', 'T10')]), Tree('task', [Token('NAME', 'T11')]),
    ])

    XOR_SEQ_10 = Tree('xor', [
        Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')]),
        Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]),
        Tree('sequential', [
            Tree('task', [Token('NAME', 'T5')]),
            Tree('xor', [
                Tree('task', [Token('NAME', 'T6')]),
                Tree('sequential', [
                    Tree('xor', [Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')]), Tree('task', [Token('NAME', 'T10')])]),
                    Tree('xor', [Tree('task', [Token('NAME', 'T11')]), Tree('task', [Token('NAME', 'T12')])])
                ])
            ])
        ]),
        Tree('task', [Token('NAME', 'T13')]), Tree('task', [Token('NAME', 'T14')]),
        Tree('task', [Token('NAME', 'T15')]), Tree('task', [Token('NAME', 'T16')]),
        Tree('task', [Token('NAME', 'T17')]), Tree('task', [Token('NAME', 'T18')]),
        Tree('task', [Token('NAME', 'T19')]), Tree('task', [Token('NAME', 'T20')]),
        Tree('task', [Token('NAME', 'T21')]), Tree('task', [Token('NAME', 'T22')]),
        Tree('sequential', [Tree('task', [Token('NAME', 'T23')]), Tree('task', [Token('NAME', 'T24')])]),
        Tree('task', [Token('NAME', 'T25')]), Tree('task', [Token('NAME', 'T26')]),
    ])

    PAR_SEQ_5 = Tree('sequential', [
        Tree('task', [Token('NAME', 'T1')]),
        Tree('parallel', [Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')]), Tree('task', [Token('NAME', 'T6')])]),
        Tree('task', [Token('NAME', 'T7')]),
        Tree('parallel', [Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')])])
    ])

    PAR_SEQ_11 = Tree('parallel', [
        Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')]),
        Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]),
        Tree('task', [Token('NAME', 'T5')]),
        Tree('sequential', [
            Tree('parallel', [Tree('task', [Token('NAME', 'T6')]), Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')])]),
            Tree('task', [Token('NAME', 'T9')])
        ]),
        Tree('task', [Token('NAME', 'T10')]),
        Tree('sequential', [Tree('task', [Token('NAME', 'T11')]), Tree('task', [Token('NAME', 'T12')])]),
        Tree('task', [Token('NAME', 'T13')]),
    ])

    XOR_PAR_SEQ_4 = Tree('parallel', [
        Tree('sequential', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')])]),
        Tree('xor', [Tree('task', [Token('NAME', 'T6')]), Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')])])
    ])

    XOR_PAR_SEQ_9 = Tree('xor', [
        Tree('task', [Token('NAME', 'T1')]),
        Tree('parallel', [
            Tree('sequential', [
                Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]),
                Tree('xor', [Tree('task', [Token('NAME', 'T5')]), Tree('task', [Token('NAME', 'T6')])])
            ]),
            Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')])
        ]),
        Tree('sequential', [
            Tree('xor', [Tree('task', [Token('NAME', 'T9')]), Tree('task', [Token('NAME', 'T10')])]),
            Tree('task', [Token('NAME', 'T11')])
        ])
    ])

    XOR_PAR_SEQ_16 = Tree('xor', [
        Tree('sequential', [
            Tree('task', [Token('NAME', 'T1')]),
            Tree('parallel', [
                Tree('xor', [
                    Tree('task', [Token('NAME', 'T2')]),
                    Tree('parallel', [Tree('xor', [Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')])]), Tree('task', [Token('NAME', 'T5')])]),
                    Tree('task', [Token('NAME', 'T6')])
                ]),
                Tree('sequential', [
                    Tree('parallel', [Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')])]),
                    Tree('xor', [Tree('task', [Token('NAME', 'T9')]), Tree('task', [Token('NAME', 'T10')]), Tree('task', [Token('NAME', 'T11')])])
                ])
            ])
        ]),
        Tree('task', [Token('NAME', 'T12')]), Tree('task', [Token('NAME', 'T13')]),
    ])

    LOOP_SEQ_3 = Tree('loop', [
        Tree('sequential', [
            Tree('loop', [Tree('task', [Token('NAME', 'T1')])]),
            Tree('task', [Token('NAME', 'T2')]),
            Tree('loop', [Tree('task', [Token('NAME', 'T3')])]),
            Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T4')])]), Tree('task', [Token('NAME', 'T5')])])])
        ])
    ])

    LOOP_SEQ_5 = Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T2')])]), Tree('task', [Token('NAME', 'T3')])])])])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T4')])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T5')])]), Tree('loop', [Tree('task', [Token('NAME', 'T6')])]), Tree('loop', [Tree('task', [Token('NAME', 'T7')])])])])])])])]), Tree('loop', [Tree('sequential', [Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')])])])])])

    LOOP_XOR_5 = Tree('loop', [
        Tree('xor', [
            Tree('loop', [Tree('task', [Token('NAME', 'T1')])]),
            Tree('loop', [Tree('xor', [Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')])])])
        ])
    ])

    LOOP_XOR_10 = Tree('xor', [
        Tree('loop', [Tree('xor', [
            Tree('loop', [Tree('xor', [Tree('loop', [Tree('xor', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')])])]), Tree('task', [Token('NAME', 'T3')]), Tree('loop', [Tree('task', [Token('NAME', 'T4')])]), Tree('loop', [Tree('xor', [Tree('loop', [Tree('task', [Token('NAME', 'T5')])]), Tree('loop', [Tree('task', [Token('NAME', 'T6')])]), Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')]), Tree('task', [Token('NAME', 'T10')])])])])]),
            Tree('task', [Token('NAME', 'T11')]), Tree('task', [Token('NAME', 'T12')])
        ])]),
    ])

    LOOP_PAR_9 = Tree('loop', [
        Tree('parallel', [
            Tree('loop', [Tree('parallel', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('loop', [Tree('task', [Token('NAME', 'T2')])])])]),
            Tree('task', [Token('NAME', 'T3')])
        ])
    ])

    LOOP_PAR_15 = Tree('loop', [Tree('parallel', [Tree('loop', [Tree('parallel', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('loop', [Tree('parallel', [Tree('task', [Token('NAME', 'T2')]), Tree('task', [Token('NAME', 'T3')])])])])]), Tree('loop', [Tree('parallel', [Tree('loop', [Tree('task', [Token('NAME', 'T4')])]), Tree('task', [Token('NAME', 'T5')])])])])])

    LOOP_XOR_PAR_SEQ_6 = Tree('parallel', [
        Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('task', [Token('NAME', 'T2')])])]),
        Tree('xor', [
            Tree('parallel', [Tree('task', [Token('NAME', 'T3')]), Tree('sequential', [Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')])])]),
            Tree('sequential', [Tree('task', [Token('NAME', 'T6')]), Tree('task', [Token('NAME', 'T7')])])
        ])
    ])

    LOOP_XOR_PAR_SEQ_12 = Tree('parallel', [
        Tree('xor', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')])]),
        Tree('xor', [
            Tree('sequential', [
                Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]),
                Tree('xor', [Tree('task', [Token('NAME', 'T5')]), Tree('parallel', [Tree('task', [Token('NAME', 'T6')]), Tree('task', [Token('NAME', 'T7')])])])
            ]),
            Tree('loop', [Tree('xor', [
                Tree('loop', [Tree('parallel', [Tree('task', [Token('NAME', 'T8')]), Tree('task', [Token('NAME', 'T9')])])]),
                Tree('loop', [Tree('task', [Token('NAME', 'T10')])])
            ])])
        ])
    ])

    LOOP_XOR_PAR_SEQ_25 = Tree('loop', [
        Tree('parallel', [
            Tree('xor', [
                Tree('loop', [Tree('sequential', [Tree('parallel', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')])]), Tree('task', [Token('NAME', 'T3')])])]),
                Tree('parallel', [
                    Tree('xor', [Tree('task', [Token('NAME', 'T4')]), Tree('loop', [Tree('task', [Token('NAME', 'T5')])]), Tree('task', [Token('NAME', 'T6')])]),
                    Tree('loop', [Tree('xor', [Tree('task', [Token('NAME', 'T7')]), Tree('task', [Token('NAME', 'T8')])])])
                ])
            ]),
            Tree('task', [Token('NAME', 'T9')]), Tree('task', [Token('NAME', 'T10')]),
            Tree('xor', [Tree('task', [Token('NAME', 'T11')]), Tree('task', [Token('NAME', 'T12')])])
        ])
    ])

    def _base(n_traces, no_cluster=True, ntpc=None, no_xor=False, no_loop=False, coverage=None, no_interval=False):
        args = ["--n-traces", str(n_traces)]
        if no_xor:
            args += ["--no-xor"]
        if no_loop:
            args += ["--no-loop"]
        if coverage is not None:
            args += ["--coverage", str(coverage)]
        if no_cluster:
            args += ["--no-cluster"]
        else:
            args += ["--min-clusters", "5", "--num-trace-per-cluster", str(ntpc)]
        args += ["--k-partition", "3", "--max-depth", "5", "--window", "5"]
        return args

    def _ntpc(n): # num trace per cluster
        return {3000: 600, 5000: 1000, 10000: 2000, 15000: 3000, 20000: 4000, 30000: 6000, 50000: 10000}[n]

    processes = []

    # === 1) XOR + SEQ (5) ===
    processes += [
        {"name": "xs_c5_3k",              "tree": XOR_SEQ_5,  "args": _base(3000,  no_xor=True, no_loop=True)},
        {"name": "xs_c10_3k_nocluster",   "tree": XOR_SEQ_10, "args": _base(3000,  no_xor=True, no_loop=True)},
        {"name": "xs_c10_3k_cluster",     "tree": XOR_SEQ_10, "args": _base(3000,  no_xor=True, no_loop=True, no_cluster=False, ntpc=_ntpc(3000))},
        {"name": "xs_c10_15k_nocluster",  "tree": XOR_SEQ_10, "args": _base(15000, no_xor=True, no_loop=True)},
        {"name": "xs_c10_15k_cluster",    "tree": XOR_SEQ_10, "args": _base(15000, no_xor=True, no_loop=True, no_cluster=False, ntpc=_ntpc(15000))},
    ]

    # === 2) PAR + SEQ (9) ===
    processes += [
        {"name": "ps_c5_5k_interval", "tree": PAR_SEQ_5, "args": _base(5000, no_xor=True, no_loop=True)},
        {"name": "ps_c5_5k_nointerval", "tree": PAR_SEQ_5, "args": _base(5000, no_xor=True, no_loop=True), "no_interval": True},
    ]
    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                name = f"ps_c11_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}"
                entry = {"name": name, "tree": PAR_SEQ_11, "args": _base(n, no_xor=True, no_loop=True, no_cluster=clust, ntpc=nc)}
                if ni:
                    entry["no_interval"] = True
                processes.append(entry)

    # === 3) XOR + PAR + SEQ (29) ===
    processes += [
        {"name": "xps_c4_5k", "tree": XOR_PAR_SEQ_4, "args": _base(5000, no_xor=True, no_loop=True)},
    ]
    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            name = f"xps_c9_{n//1000}k_{'nocluster' if clust else 'cluster'}"
            processes.append({"name": name, "tree": XOR_PAR_SEQ_9, "args": _base(n, no_xor=True, no_loop=True, no_cluster=clust, ntpc=nc)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_loop=True, coverage=0.5)), ("cov1", dict(no_loop=True, coverage=1.0))]:
                    name = f"xps_c16_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}_{cov_name}"
                    entry = {"name": name, "tree": XOR_PAR_SEQ_16, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)}
                    if ni:
                        entry["no_interval"] = True
                    processes.append(entry)

    # === 4) LOOP + SEQ (15) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"ls_c3_5k_{cov_name}", "tree": LOOP_SEQ_3, "args": _base(5000, **kw)})

    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None)]:  # solo nocluster: tracce loop+seq lunghe e tutte uniche -> clustering Levenshtein O(L^2) infattibile e poco utile
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"ls_c5_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_SEQ_5, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    # === 5) LOOP + XOR (15) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lx_c5_5k_{cov_name}", "tree": LOOP_XOR_5, "args": _base(5000, **kw)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"lx_c10_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_XOR_10, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    # === 6) LOOP + PAR (27) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lp_c9_5k_{cov_name}", "tree": LOOP_PAR_9, "args": _base(5000, **kw)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (30000, _ntpc(30000))]:
        for clust, nc in [(True, None)]:  # solo nocluster: par-in-loop -> 100% tracce uniche (anche con nointerval) -> clustering caro e senza bilanciamento utile
            for intv, ni in [("interval", False), ("nointerval", True)]:
                for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                    name = f"lp_c15_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}_{cov_name}"
                    entry = {"name": name, "tree": LOOP_PAR_15, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)}
                    if ni:
                        entry["no_interval"] = True
                    processes.append(entry)

    # === 7) LOOP + XOR + PAR + SEQ (71) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lxps_c6_5k_{cov_name}", "tree": LOOP_XOR_PAR_SEQ_6, "args": _base(5000, **kw)})

    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"lxps_c12_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_XOR_PAR_SEQ_12, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    cov_combos = [
        ("ll0_xl0",   dict(no_xor=True, no_loop=True)),
        ("ll05_xl0",  dict(no_xor=True, coverage=0.5)),
        ("ll1_xl0",   dict(no_xor=True, coverage=1.0)),
        ("ll0_xl05",  dict(no_loop=True, coverage=0.5)),
        ("ll0_xl1",   dict(no_loop=True, coverage=1.0)),
        ("ll05_xl05", dict(coverage=0.5)),
        ("ll1_xl1",   dict(coverage=1.0)),
    ]
    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                for cov_name, kw in cov_combos:
                    name = f"lxps_c25_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}_{cov_name}"
                    entry = {"name": name, "tree": LOOP_XOR_PAR_SEQ_25, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)}
                    if ni:
                        entry["no_interval"] = True
                    processes.append(entry)

    return processes

PROCESSES = get_processes()

PREPARE_DATA_ARGS = [
    "--n-traces",              "30000",
    "--coverage",              "0.5",
    "--num-trace-per-cluster", "1000",
    "--k-partition",           "3",
    "--window",                "5",
]

OPTUNA_STEP = BASE_DIR / "experiments" / "run_all_optuna.ipynb"

VARIANT_STEPS = [
    BASE_DIR / "variants" / "variant1_taskregion_time.ipynb",
    BASE_DIR / "variants" / "variant2_taskregion_time.ipynb",
    BASE_DIR / "variants" / "variant3_task_region_time.ipynb",
    BASE_DIR / "variants" / "variant4_unified.ipynb",
    BASE_DIR / "variants" / "variant5_task_unified.ipynb",
    BASE_DIR / "variants" / "variant6_taskregion_unified.ipynb",
]


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

TG_MAX = 4000


def _tg_send(text: str, silent: bool = False) -> int | None:
    """Invia un messaggio e restituisce il message_id (None se errore)."""
    chunks = [text[i:i+TG_MAX] for i in range(0, len(text), TG_MAX)]
    last_id = None
    for chunk in chunks:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                data={
                    "chat_id":             CHAT_ID,
                    "text":                chunk,
                    "parse_mode":          "HTML",
                    "disable_notification": "true" if silent else "false",
                },
                timeout=10,
            )
            data = r.json()
            if data.get("ok"):
                last_id = data["result"]["message_id"]
        except Exception as e:
            print(f"[TG SEND ERROR] {e}")
    return last_id


def _tg_edit(message_id: int, text: str):
    """Edita un messaggio esistente."""
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText",
            data={
                "chat_id":    CHAT_ID,
                "message_id": message_id,
                "text":       text[:TG_MAX],
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as e:
        print(f"[TG EDIT ERROR] {e}")


def _tg_delete(message_id: int | None):
    """Cancella un messaggio."""
    if message_id is None:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/deleteMessage",
            data={"chat_id": CHAT_ID, "message_id": message_id},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG DELETE ERROR] {e}")


def _tg(text: str, silent: bool = False):
    """Fire-and-forget: invia senza restituire l'id."""
    _tg_send(text, silent=silent)


# ---------------------------------------------------------------------------
# Logging su file + stdout
# ---------------------------------------------------------------------------

_report_path: Path | None = None
_log_lock = threading.Lock()


def _log(msg: str, telegram: bool = False, silent_tg: bool = False):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if _report_path:
            with open(_report_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    if telegram:
        _tg(msg, silent=silent_tg)


# ---------------------------------------------------------------------------
# Esecuzione script .py — nessun messaggio Telegram (gestito dal chiamante)
# ---------------------------------------------------------------------------

def run_script(name: str, script_path: Path, extra_args: list) -> tuple[bool, str]:
    cmd = [sys.executable, "-u", str(script_path)] + [str(a) for a in extra_args]
    _log(f"▶ [{name}] Avvio script: {script_path.name}")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    output_lines: list[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
            env=env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            _log(f"  {line}")
            output_lines.append(line)
        proc.wait()

        elapsed     = time.time() - t0
        full_output = "\n".join(output_lines)

        if proc.returncode == 0:
            _log(f"✅ [{name}] Completato: {script_path.name} ({elapsed:.0f}s)")
        else:
            _log(f"❌ [{name}] ERRORE (exit {proc.returncode}): {script_path.name} ({elapsed:.0f}s)")
        return proc.returncode == 0, full_output

    except Exception as e:
        err = str(e)
        _log(f"❌ [{name}] ECCEZIONE: {script_path.name}\n{err}")
        return False, err


# ---------------------------------------------------------------------------
# Esecuzione notebook .ipynb — nessun messaggio Telegram (gestito dal chiamante)
# ---------------------------------------------------------------------------

def _extract_all_outputs(nb_path: Path) -> str:
    try:
        nb = nbformat.read(nb_path, as_version=4)
    except Exception:
        return ""

    chunks = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            otype = out.get("output_type", "")
            if otype == "stream":
                text = "".join(out.get("text", []))
            elif otype in ("execute_result", "display_data"):
                text = "".join(out.get("data", {}).get("text/plain", []))
            elif otype == "error":
                # Include il traceback completo — necessario per rilevare CUDA OOM
                text = "\n".join(out.get("traceback", [])) + "\n" + out.get("ename", "") + ": " + out.get("evalue", "")
            else:
                continue
            text = text.strip()
            if text:
                chunks.append(text)

    return "\n---\n".join(chunks)


def _popen_isolated(cmd: list, env: dict) -> tuple[subprocess.Popen, callable]:
    """Spawna un subprocess in un nuovo process group (cross-platform).
    Restituisce (proc, kill_fn) dove kill_fn() uccide l'intero albero."""
    if sys.platform == "win32":
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        kill_fn = lambda: subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            preexec_fn=os.setsid,
        )
        # Salva PGID subito: rimane valido anche dopo che il subprocess esce e viene reaped
        pgid = os.getpgid(proc.pid)

        def kill_fn():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    return proc, kill_fn


def run_notebook(name: str, nb_path: Path, output_dir: Path, timeout: int | None = NOTEBOOK_TIMEOUT) -> tuple[bool, str]:
    """Esegue il notebook in un subprocess isolato con nuovo process group.
    Al termine (successo, errore o timeout) garantisce la terminazione dell'intero
    albero di processi figlio — incluso il kernel Jupyter — su Linux e Windows."""
    output_path = output_dir / nb_path.name
    _log(f"▶ [{name}] Avvio notebook: {nb_path.name}")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    pm_script = (
        "import papermill as pm\n"
        f"pm.execute_notebook(\n"
        f"    {repr(str(nb_path))},\n"
        f"    {repr(str(output_path))},\n"
        f"    kernel_name='python3',\n"
        f"    progress_bar=False,\n"
        f"    request_save_on_cell_execute=True,\n"
        f"    cwd={repr(str(BASE_DIR))},\n"
        f")\n"
    )

    tmp_script = None
    kill_fn = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=BASE_DIR, prefix="_pm_nb_"
        ) as f:
            f.write(pm_script)
            tmp_script = f.name

        proc, kill_fn = _popen_isolated([sys.executable, tmp_script], env)

        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _log(f"⚠️ [{name}] TIMEOUT {nb_path.name} — terminazione forzata")
            kill_fn()
            stdout, _ = proc.communicate()
            elapsed = time.time() - t0
            _log(f"❌ [{name}] TIMEOUT: {nb_path.name} ({elapsed:.0f}s)")
            return False, "timeout"

        elapsed  = time.time() - t0
        full_out = _extract_all_outputs(output_path)

        if proc.returncode == 0:
            _log(f"✅ [{name}] Completato: {nb_path.name} ({elapsed:.0f}s)")
            return True, full_out
        else:
            err_msg = (stdout or "")[-800:]
            _log(f"❌ [{name}] ERRORE (exit {proc.returncode}): {nb_path.name} ({elapsed:.0f}s)\n{err_msg}")
            # Combina output notebook + stderr subprocess: l'OOM check trova la stringa in uno dei due
            combined = "\n---\n".join(filter(None, [full_out, err_msg]))
            return False, combined

    except Exception as e:
        err_msg = str(e)
        _log(f"❌ [{name}] ECCEZIONE: {nb_path.name}\n{err_msg}")
        return False, err_msg

    finally:
        # Sempre: termina l'albero di processi per eliminare kernel orfani
        if kill_fn is not None:
            kill_fn()
        if tmp_script:
            Path(tmp_script).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# VRAM wait + retry
# ---------------------------------------------------------------------------

def wait_for_vram(
    min_free_mb: int = VRAM_MIN_FREE_MB,
    poll_interval: int = VRAM_POLL_INTERVAL,
    timeout: int = VRAM_WAIT_TIMEOUT,
) -> bool:
    """Aspetta finché nvidia-smi segnala almeno min_free_mb di VRAM libera."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            free_mb = int(result.stdout.strip().split("\n")[0])
            if free_mb >= min_free_mb:
                return True
        except Exception:
            return True  # nvidia-smi non disponibile: tenta comunque
        time.sleep(poll_interval)
    return False


def run_notebook_with_retry(name: str, nb_path: Path, output_dir: Path) -> tuple[bool, str]:
    """Esegue un notebook con retry automatico su CUDA OOM."""
    for attempt in range(VRAM_RETRY_MAX + 1):
        ok, out = run_notebook(name, nb_path, output_dir)
        if ok:
            return True, out
        is_oom = "CUDA out of memory" in out or "OutOfMemoryError" in out
        if is_oom and attempt < VRAM_RETRY_MAX:
            _log(
                f"⚠️ [{name}] OOM su {nb_path.name} (tentativo {attempt + 1}/{VRAM_RETRY_MAX}) — attendo VRAM...",
                telegram=True,
            )
            if wait_for_vram():
                _log(f"🔄 [{name}] VRAM disponibile, riprovo {nb_path.name}...")
                continue
        return False, out
    return False, out


# ---------------------------------------------------------------------------
# Watcher thread per il progresso optuna
# ---------------------------------------------------------------------------

def _watch_optuna(pname: str, msg_id: int, stop_event: threading.Event):
    last = 0
    while not stop_event.is_set():
        try:
            if OPTUNA_PROGRESS_FILE.exists():
                data = json.loads(OPTUNA_PROGRESS_FILE.read_text())
                v = data.get("variant", 0)
                if v != last:
                    last = v
                    _tg_edit(msg_id, f"⏳ <b>[{pname}] Optuna</b>: variante {v}/6...")
        except Exception:
            pass
        stop_event.wait(3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stop_on_error: bool = False):
    global _report_path

    # ── Resume automatico: se il checkpoint esiste riprende da dove si è fermato ─
    start_from = 0
    if CHECKPOINT_FILE.exists():
        try:
            ckpt: list = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
            n_variants     = len(VARIANT_STEPS)
            complete_procs = len(ckpt) // n_variants
            partial        = len(ckpt) % n_variants
            if partial:
                ckpt = ckpt[:complete_procs * n_variants]
                CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2, ensure_ascii=False), encoding="utf-8")
            start_from = complete_procs
            if start_from > 0:
                _log(f"▶️  RESUME: {complete_procs} processi nel checkpoint, riparto dal processo {start_from + 1}.")
        except Exception as exc:
            _log(f"⚠️  Impossibile leggere il checkpoint ({exc}). Avvio da zero.")
            start_from = 0

    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir      = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _report_path = run_dir / "report.txt"
    total_procs  = len(PROCESSES)
    start_ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _log(f"{'='*60}\nPIPELINE AVVIATA — {start_ts}\n"
         f"Processi: {total_procs}  |  Run dir: {run_dir}\n{'='*60}")

    # Dashboard: unico messaggio che viene editato per tutta la durata
    dashboard_id = _tg_send(
        f"📊 <b>Pipeline avviata</b>\n"
        f"📅 {start_ts}\n"
        f"📦 {total_procs} processi totali\n\n"
        f"🔵 Avvio..."
    )

    all_results: list[dict] = []

    for proc_idx, proc in enumerate(PROCESSES, 1):
        if proc_idx <= start_from:
            _log(f"⏭️  Salto {proc['name']} (già completato nel checkpoint).")
            continue

        pname      = proc["name"]
        proc_dir   = run_dir / pname
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_results: list[dict] = []
        proc_start = datetime.now().strftime("%H:%M:%S")

        _log(f"\n{'─'*60}\n🔵 PROCESSO: {pname}  [{proc_start}]\n{'─'*60}")

        # Aggiorna dashboard
        if dashboard_id:
            _tg_edit(
                dashboard_id,
                f"📊 <b>Pipeline in corso</b>\n"
                f"📅 {start_ts}\n"
                f"✅ {proc_idx - 1}/{total_procs} completati\n\n"
                f"🔵 Processo {proc_idx}/{total_procs}\n"
                f"📋 {pname}",
            )

        # Messaggio permanente: intestazione processo
        _tg_send(f"🔵 <b>Processo {proc_idx}/{total_procs}</b>: <code>{pname}</code>\n🕐 {proc_start}")

        # Serializza il lark.Tree
        tree_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(proc["tree"], tree_file)
        tree_file.close()

        # ── 1. prepare_data ──────────────────────────────────────────────
        prepare_args = ["--tree-file", tree_file.name] + proc.get("args", PREPARE_DATA_ARGS)
        if proc.get("no_interval", False):
            prepare_args.append("--no-interval")

        tmp_prepare = _tg_send(f"⏳ <b>[{pname}]</b> prepare_data in corso...", silent=True)
        ok, out = run_script(pname, BASE_DIR / "data" / "prepare_data.py", prepare_args)
        if ok:
            _tg_edit(tmp_prepare, f"✅ <b>[{pname}] prepare_data</b>\n\n<pre>{html.escape(out[-800:])}</pre>" if out else f"✅ <b>[{pname}] prepare_data</b>")
        else:
            _tg_edit(tmp_prepare, f"❌ <b>[{pname}] prepare_data ERRORE</b>\n\n<pre>{html.escape(out[-800:])}</pre>")
        Path(tree_file.name).unlink(missing_ok=True)
        proc_results.append({"step": "prepare_data.py", "success": ok})

        if not ok and stop_on_error:
            _log(f"🛑 [{pname}] stop_on_error, salto al prossimo processo.", telegram=True)
            all_results.append({"process": pname, "steps": proc_results})
            continue

        # ── Prepara stato condiviso varianti ─────────────────────────────
        existing_variants = [nb for nb in VARIANT_STEPS if nb.exists()]
        for nb in VARIANT_STEPS:
            if not nb.exists():
                _log(f"⚠️ [{pname}] {nb.name} non trovato, saltato.", telegram=True)

        # Rimuove i params del run precedente per evitare trigger prematuri
        for i in range(1, len(VARIANT_STEPS) + 1):
            (DATA_DIR / f"v{i}_best_params.pt").unlink(missing_ok=True)

        var_tmp_ids: dict[str, int | None]          = {}
        variant_results: dict[str, tuple[bool, str]] = {}
        variant_futures: dict[str, Future]           = {}
        variant_executor = ThreadPoolExecutor(max_workers=max(len(existing_variants), 1))

        def _launch_variant(nb: Path) -> None:
            tmp_id = _tg_send(f"⏳ <b>[{pname}]</b> {nb.name} in corso...", silent=True)
            var_tmp_ids[nb.name] = tmp_id
            ok, out = run_notebook_with_retry(pname, nb, proc_dir)
            _tg_delete(var_tmp_ids.get(nb.name))
            variant_results[nb.name] = (ok, out)

        def _stagger_watcher(done_event: threading.Event) -> None:
            """Lancia ogni variante appena il suo v{i}_best_params.pt è disponibile.
            Mentre optuna è attivo limita a 2 varianti contemporanee; dopo, nessun limite."""
            launched: set[str] = set()
            while True:
                optuna_running = not done_event.is_set()
                for i, nb in enumerate(VARIANT_STEPS, 1):
                    if nb not in existing_variants:
                        continue
                    if nb.name not in launched and (DATA_DIR / f"v{i}_best_params.pt").exists():
                        if optuna_running:
                            running = sum(1 for f in variant_futures.values() if not f.done())
                            if running >= 2:
                                break  # aspetta il prossimo ciclo
                        launched.add(nb.name)
                        variant_futures[nb.name] = variant_executor.submit(_launch_variant, nb)
                if done_event.is_set():
                    # sweep finale senza limiti: lancia le rimanenti (tipicamente V6)
                    for i, nb in enumerate(VARIANT_STEPS, 1):
                        if nb not in existing_variants:
                            continue
                        if nb.name not in launched:
                            launched.add(nb.name)
                            variant_futures[nb.name] = variant_executor.submit(_launch_variant, nb)
                    break
                done_event.wait(5)

        # ── 2. Optuna ────────────────────────────────────────────────────
        if OPTUNA_STEP.exists():
            tmp_optuna = _tg_send(f"⏳ <b>[{pname}] Optuna</b>: avvio...", silent=True)
            OPTUNA_PROGRESS_FILE.write_text('{"variant": 0, "total": 6}')

            stop_ev  = threading.Event()
            watch_t  = threading.Thread(
                target=_watch_optuna, args=(pname, tmp_optuna, stop_ev), daemon=True
            )
            watch_t.start()

            stagger_t = threading.Thread(target=_stagger_watcher, args=(stop_ev,), daemon=True)
            stagger_t.start()

            ok, _ = run_notebook(pname, OPTUNA_STEP, proc_dir, timeout=OPTUNA_TIMEOUT)

            stop_ev.set()
            watch_t.join(timeout=5)
            stagger_t.join(timeout=30)
            _tg_delete(tmp_optuna)
            OPTUNA_PROGRESS_FILE.unlink(missing_ok=True)

            if not ok:
                _tg_send(f"❌ <b>[{pname}] Optuna ERRORE</b>")

            proc_results.append({"step": OPTUNA_STEP.name, "success": ok})
            if not ok and stop_on_error:
                _log(f"🛑 [{pname}] stop_on_error, salto al prossimo processo.", telegram=True)
                for f in list(variant_futures.values()):
                    f.result()
                variant_executor.shutdown(wait=False)
                all_results.append({"process": pname, "steps": proc_results})
                continue
        else:
            _log(f"⚠️ [{pname}] {OPTUNA_STEP.name} non trovato, saltato.", telegram=True)
            proc_results.append({"step": OPTUNA_STEP.name, "success": None})
            # Senza optuna lancia tutte le varianti immediatamente
            for nb in existing_variants:
                variant_futures[nb.name] = variant_executor.submit(_launch_variant, nb)

        # ── 3. Attende le varianti e raccoglie i risultati ────────────────
        for nb in existing_variants:
            if nb.name in variant_futures:
                variant_futures[nb.name].result()
        variant_executor.shutdown(wait=True)

        # Messaggi permanenti: risultati in ordine 1→6
        for nb in VARIANT_STEPS:
            ok, out = variant_results.get(nb.name, (None, ""))
            if ok is True:
                _tg_send(
                    f"✅ <b>[{pname}] {nb.name}</b>\n\n<pre>{html.escape(out)}</pre>"
                    if out else
                    f"✅ <b>[{pname}] {nb.name}</b>"
                )
            elif ok is False:
                _tg_send(f"❌ <b>[{pname}] ERRORE</b>: {nb.name}\n\n<pre>{html.escape(out[-800:])}</pre>")
            proc_results.append({"step": nb.name, "success": ok})

        all_results.append({"process": pname, "steps": proc_results})

    # ── Riepilogo finale ──────────────────────────────────────────────────
    total_steps = sum(len(p["steps"]) for p in all_results)
    total_ok    = sum(1 for p in all_results for s in p["steps"] if s["success"] is True)
    total_fail  = sum(1 for p in all_results for s in p["steps"] if s["success"] is False)
    proc_ok     = sum(1 for p in all_results if all(s["success"] is not False for s in p["steps"]))

    summary_lines = [
        f"\n{'='*60}", "RISULTATO FINALE", "="*60,
        f"  Processi OK: {proc_ok}/{len(all_results)}",
        f"  Step totali: {total_ok} OK / {total_fail} falliti / {total_steps} totali",
        "", "  Dettaglio:",
    ]
    for p in all_results:
        has_fail = any(s["success"] is False for s in p["steps"])
        summary_lines.append(f"  {'❌' if has_fail else '✅'}  {p['process']}")
    summary_lines += [f"  Report: {_report_path}", "="*60]
    _log("\n".join(summary_lines))

    # Aggiorna dashboard con risultato finale
    if dashboard_id:
        _tg_edit(
            dashboard_id,
            f"🏁 <b>Pipeline completata</b>\n"
            f"📅 {start_ts}\n"
            f"📦 {total_procs} processi\n\n"
            f"✅ {proc_ok} OK  ❌ {len(all_results) - proc_ok} falliti",
        )

    # Messaggio riepilogo finale (nuovo, rimane in chat)
    _tg_send(
        f"🏁 <b>Pipeline completata</b>\n"
        f"📊 Processi OK: {proc_ok}/{len(all_results)}\n"
        f"Step: ✅ {total_ok}  ❌ {total_fail}\n\n"
        + "\n".join(
            f"{'❌' if any(s['success'] is False for s in p['steps']) else '✅'}  {p['process']}"
            for p in all_results
        )
        + f"\n\n📄 <code>{_report_path.name}</code>"
    )

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "results": all_results}, f, indent=2)

    return total_ok, total_fail


if __name__ == "__main__":
    stop = "--stop-on-error" in sys.argv
    main(stop_on_error=stop)
