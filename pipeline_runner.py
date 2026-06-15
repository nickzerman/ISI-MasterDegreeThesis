"""
Orchestratore della pipeline sperimentale — 24 processi.

Per ogni processo in PROCESSES esegue in sequenza:
  1. data/prepare_data.py  (script, output in tempo reale)
  2. experiments/run_all_optuna.ipynb
  3. variants/variant*.ipynb  (tutti)

I notebook leggono sempre data/prepared_data.pt, che viene riscritto ad ogni processo.

Uso:
    python pipeline_runner.py
    python pipeline_runner.py --stop-on-error
"""

import sys
import time
import json
import pickle
import tempfile
import threading
import subprocess
import requests
import nbformat
import papermill as pm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from lark import Tree, Token

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

BOT_TOKEN = ""
CHAT_ID   = ""

# ---------------------------------------------------------------------------
# 24 processi.
#
# Campi obbligatori:
#   "name"        → identificativo usato nei log e nelle cartelle output
#   "tree"        → oggetto lark.Tree già costruito (copia/incolla dal tuo codice)
#
# Campi opzionali:
#   "args"        → lista di argomenti per prepare_data.py che SOSTITUISCE
#                   PREPARE_DATA_ARGS per quel processo.
#                   Se non presente, usa PREPARE_DATA_ARGS.
#   "no_interval" → se True aggiunge --no-interval a prepare_data.py
#                   (tracce con start/end task consecutivi, senza intervallo).
#                   Default: False (tracce con intervallo, comportamento standard).
#
# Esempio:
#   {
#       "name": "process_03",
#       "tree": Tree('xor', [Tree('task', [Token('NAME', 'T1')]),
#                             Tree('task', [Token('NAME', 'T2')])]),
#       "args": ["--n-traces", "50000", "--no-cluster"],
#       "no_interval": True,
#   },
# ---------------------------------------------------------------------------
def get_processes():
    from lark import Tree, Token

    # --------------------------------------------------------
    # Alberi dei processi
    # --------------------------------------------------------

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

    LOOP_SEQ_8 = Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T2')])]), Tree('loop', [Tree('task', [Token('NAME', 'T3')])])])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T4')])]), Tree('loop', [Tree('task', [Token('NAME', 'T5')])])])])])])])]), Tree('loop', [Tree('task', [Token('NAME', 'T6')])])])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T7')])]), Tree('task', [Token('NAME', 'T8')])])])])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T9')])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T10')])]), Tree('loop', [Tree('sequential', [Tree('loop', [Tree('task', [Token('NAME', 'T11')])]), Tree('task', [Token('NAME', 'T12')])])])])]), Tree('loop', [Tree('task', [Token('NAME', 'T13')])]), Tree('loop', [Tree('task', [Token('NAME', 'T14')])]), Tree('task', [Token('NAME', 'T15')])])])])]), Tree('loop', [Tree('sequential', [Tree('task', [Token('NAME', 'T16')]), Tree('loop', [Tree('task', [Token('NAME', 'T17')])])])])])]), Tree('loop', [Tree('sequential', [Tree('task', [Token('NAME', 'T18')]), Tree('loop', [Tree('task', [Token('NAME', 'T19')])]), Tree('task', [Token('NAME', 'T20')])])])])])])])

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
        Tree('task', [Token('NAME', 'T11')]), Tree('task', [Token('NAME', 'T12')]),
    ])

    LOOP_PAR_9 = Tree('loop', [
        Tree('parallel', [
            Tree('loop', [Tree('parallel', [Tree('loop', [Tree('task', [Token('NAME', 'T1')])]), Tree('loop', [Tree('task', [Token('NAME', 'T2')])])])]),
            Tree('task', [Token('NAME', 'T3')])
        ])
    ])

    LOOP_PAR_21 = Tree('loop', [
        Tree('parallel', [
            Tree('loop', [Tree('parallel', [
                Tree('loop', [Tree('parallel', [Tree('task', [Token('NAME', 'T1')]), Tree('task', [Token('NAME', 'T2')])])]),
                Tree('loop', [Tree('parallel', [Tree('task', [Token('NAME', 'T3')]), Tree('task', [Token('NAME', 'T4')]), Tree('task', [Token('NAME', 'T5')]), Tree('task', [Token('NAME', 'T6')])])])
            ])]),
            Tree('loop', [Tree('task', [Token('NAME', 'T7')])])
        ])
    ])

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

    # --------------------------------------------------------
    # Helper per costruire gli args
    # --------------------------------------------------------

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

    def _ntpc(n):
        return {5000: 1000, 15000: 3000, 50000: 10000}[n]

    # --------------------------------------------------------
    # Costruzione PROCESSES
    # --------------------------------------------------------

    processes = []

    # === 1) XOR + SEQ (5 configurazioni) ===
    processes += [
        {"name": "xs_c5_5k",              "tree": XOR_SEQ_5,  "args": _base(5000,  no_xor=True, no_loop=True)},
        {"name": "xs_c10_5k_nocluster",   "tree": XOR_SEQ_10, "args": _base(5000,  no_xor=True, no_loop=True)},
        {"name": "xs_c10_5k_cluster",     "tree": XOR_SEQ_10, "args": _base(5000,  no_xor=True, no_loop=True, no_cluster=False, ntpc=_ntpc(5000))},
        {"name": "xs_c10_15k_nocluster",  "tree": XOR_SEQ_10, "args": _base(15000, no_xor=True, no_loop=True)},
        {"name": "xs_c10_15k_cluster",    "tree": XOR_SEQ_10, "args": _base(15000, no_xor=True, no_loop=True, no_cluster=False, ntpc=_ntpc(15000))},
    ]

    # === 2) PAR + SEQ (9 configurazioni) ===
    processes += [
        {"name": "ps_c5_5k", "tree": PAR_SEQ_5, "args": _base(5000, no_xor=True, no_loop=True)},
    ]
    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                name = f"ps_c11_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}"
                entry = {"name": name, "tree": PAR_SEQ_11, "args": _base(n, no_xor=True, no_loop=True, no_cluster=clust, ntpc=nc)}
                if ni:
                    entry["no_interval"] = True
                processes.append(entry)

    # === 3) XOR + PAR + SEQ (29 configurazioni) ===
    processes += [
        {"name": "xps_c4_5k", "tree": XOR_PAR_SEQ_4, "args": _base(5000, no_xor=True, no_loop=True)},
    ]
    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            name = f"xps_c9_{n//1000}k_{'nocluster' if clust else 'cluster'}"
            processes.append({"name": name, "tree": XOR_PAR_SEQ_9, "args": _base(n, no_xor=True, no_loop=True, no_cluster=clust, ntpc=nc)})

    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_loop=True, coverage=0.5)), ("cov1", dict(no_loop=True, coverage=1.0))]:
                    name = f"xps_c16_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}_{cov_name}"
                    entry = {"name": name, "tree": XOR_PAR_SEQ_16, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)}
                    if ni:
                        entry["no_interval"] = True
                    processes.append(entry)

    # === 4) LOOP + SEQ (15 configurazioni) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"ls_c3_5k_{cov_name}", "tree": LOOP_SEQ_3, "args": _base(5000, **kw)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"ls_c8_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_SEQ_8, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    # === 5) LOOP + XOR (15 configurazioni) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lx_c5_5k_{cov_name}", "tree": LOOP_XOR_5, "args": _base(5000, **kw)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"lx_c10_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_XOR_10, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    # === 6) LOOP + PAR (27 configurazioni) ===
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lp_c9_5k_{cov_name}", "tree": LOOP_PAR_9, "args": _base(5000, **kw)})

    for n, ntpc_val in [(15000, _ntpc(15000)), (50000, _ntpc(50000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for intv, ni in [("interval", False), ("nointerval", True)]:
                for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                    name = f"lp_c21_{n//1000}k_{'nocluster' if clust else 'cluster'}_{intv}_{cov_name}"
                    entry = {"name": name, "tree": LOOP_PAR_21, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)}
                    if ni:
                        entry["no_interval"] = True
                    processes.append(entry)

    # === 7) LOOP + XOR + PAR + SEQ (71 configurazioni) ===
    # LXPS-1: c=6, 5K, no cluster, solo con intervalli, coverage solo loop
    for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
        processes.append({"name": f"lxps_c6_5k_{cov_name}", "tree": LOOP_XOR_PAR_SEQ_6, "args": _base(5000, **kw)})

    # LXPS-2: c=12, 5K/15K × cluster × solo con intervalli × coverage solo loop
    for n, ntpc_val in [(5000, _ntpc(5000)), (15000, _ntpc(15000))]:
        for clust, nc in [(True, None), (False, ntpc_val)]:
            for cov_name, kw in [("cov0", dict(no_xor=True, no_loop=True)), ("cov05", dict(no_xor=True, coverage=0.5)), ("cov1", dict(no_xor=True, coverage=1.0))]:
                name = f"lxps_c12_{n//1000}k_{'nocluster' if clust else 'cluster'}_{cov_name}"
                processes.append({"name": name, "tree": LOOP_XOR_PAR_SEQ_12, "args": _base(n, no_cluster=clust, ntpc=nc, **kw)})

    # LXPS-3: c=25, 15K/50K × cluster × intervalli × 7 combinazioni coverage
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

# Argomenti di default per prepare_data.py.
# Usati per i processi che non hanno il campo "args".
PREPARE_DATA_ARGS = [
    "--n-traces",              "30000",
    "--coverage",              "0.5",
    "--num-trace-per-cluster", "1000",
    "--k-partition",           "3",
    "--window",                "5",
]

# Gira in sequenza prima delle varianti (produce i best params).
OPTUNA_STEP = BASE_DIR / "experiments" / "run_all_optuna.ipynb"

# Girano tutti in parallelo dopo optuna.
VARIANT_STEPS = [
    BASE_DIR / "variants" / "variant1_taskregion_time.ipynb",
    BASE_DIR / "variants" / "variant2_taskregion_time.ipynb",
    BASE_DIR / "variants" / "variant3_task_region_time.ipynb",
    BASE_DIR / "variants" / "variant4_unified.ipynb",
    BASE_DIR / "variants" / "variant5_task_unified.ipynb",
    BASE_DIR / "variants" / "variant6_taskregion_unified.ipynb",
]


# ---------------------------------------------------------------------------
# Telegram — supporta messaggi lunghi (split automatico ogni 4000 char)
# ---------------------------------------------------------------------------

TG_MAX = 4000

def _tg(text: str, silent: bool = False):
    chunks = [text[i:i+TG_MAX] for i in range(0, len(text), TG_MAX)]
    for chunk in chunks:
        try:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_notification": silent,
                },
                timeout=10,
            )
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}")


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
# Esecuzione script .py — stdout in tempo reale, tutto su Telegram a fine run
# ---------------------------------------------------------------------------

def run_script(name: str, script_path: Path, extra_args: list) -> tuple[bool, str]:
    cmd = [sys.executable, "-u", str(script_path)] + [str(a) for a in extra_args]

    _log(f"▶ [{name}] Avvio script: {script_path.name}", telegram=True)
    t0 = time.time()

    output_lines: list[str] = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
        )
        for line in proc.stdout:
            line = line.rstrip()
            _log(f"  {line}")
            output_lines.append(line)
        proc.wait()

        elapsed = time.time() - t0
        full_output = "\n".join(output_lines)

        if proc.returncode == 0:
            _log(f"✅ [{name}] Completato: {script_path.name} ({elapsed:.0f}s)")
            _tg(
                f"✅ <b>[{name}] Completato</b>: {script_path.name}\n"
                f"⏱ {elapsed:.0f}s\n\n"
                f"<pre>{full_output}</pre>"
            )
            return True, full_output
        else:
            _log(f"❌ [{name}] ERRORE (exit {proc.returncode}): {script_path.name}")
            _tg(
                f"❌ <b>[{name}] ERRORE</b>: {script_path.name}\n"
                f"exit={proc.returncode}  ⏱ {elapsed:.0f}s\n\n"
                f"<pre>{full_output[-1500:]}</pre>"
            )
            return False, full_output

    except Exception as e:
        err = str(e)
        _log(f"❌ [{name}] ECCEZIONE: {script_path.name}\n{err}")
        _tg(f"❌ <b>[{name}] ECCEZIONE</b>: {script_path.name}\n<pre>{err}</pre>")
        return False, err


# ---------------------------------------------------------------------------
# Esecuzione notebook .ipynb — tutto l'output su Telegram a fine run
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
            else:
                continue
            text = text.strip()
            if text:
                chunks.append(text)

    return "\n---\n".join(chunks)


def run_notebook(name: str, nb_path: Path, output_dir: Path, send_telegram: bool = True) -> tuple[bool, str]:
    output_path = output_dir / nb_path.name

    _log(f"▶ [{name}] Avvio notebook: {nb_path.name}", telegram=True)
    t0 = time.time()

    try:
        pm.execute_notebook(
            str(nb_path),
            str(output_path),
            kernel_name="python3",
            progress_bar=False,
            request_save_on_cell_execute=True,
            cwd=str(BASE_DIR),
        )
        elapsed  = time.time() - t0
        full_out = _extract_all_outputs(output_path)

        _log(f"✅ [{name}] Completato: {nb_path.name} ({elapsed:.0f}s)")
        if send_telegram:
            _tg(
                f"✅ <b>[{name}] Completato</b>: {nb_path.name}\n"
                f"⏱ {elapsed:.0f}s\n\n"
                f"<pre>{full_out}</pre>"
                if full_out else
                f"✅ <b>[{name}] Completato</b>: {nb_path.name}\n⏱ {elapsed:.0f}s"
            )
        return True, full_out

    except pm.PapermillExecutionError as e:
        elapsed = time.time() - t0
        err_msg = str(e)[:800]
        _log(f"❌ [{name}] ERRORE: {nb_path.name} ({elapsed:.0f}s)\n{err_msg}")
        if send_telegram:
            _tg(f"❌ <b>[{name}] ERRORE</b>: {nb_path.name}\n⏱ {elapsed:.0f}s\n\n<pre>{err_msg}</pre>")
        return False, err_msg

    except Exception as e:
        err_msg = str(e)
        _log(f"❌ [{name}] ECCEZIONE: {nb_path.name}\n{err_msg}")
        if send_telegram:
            _tg(f"❌ <b>[{name}] ECCEZIONE</b>: {nb_path.name}\n<pre>{err_msg}</pre>")
        return False, err_msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stop_on_error: bool = False):
    global _report_path

    run_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir      = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _report_path = run_dir / "report.txt"

    _log(f"{'='*60}\nPIPELINE AVVIATA — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
         f"Processi: {len(PROCESSES)}  |  Run dir: {run_dir}\n{'='*60}")
    _tg(
        f"🚀 <b>Pipeline avviata</b>\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📊 {len(PROCESSES)} processi\n"
        f"📁 {run_dir.name}"
    )

    all_results: list[dict] = []

    for proc in PROCESSES:
        pname      = proc["name"]
        proc_dir   = run_dir / pname
        proc_dir.mkdir(parents=True, exist_ok=True)
        proc_results: list[dict] = []
        proc_start = datetime.now().strftime("%H:%M:%S")

        _log(f"\n{'─'*60}\n🔵 PROCESSO: {pname}  [{proc_start}]\n{'─'*60}")
        _tg(
            f"🔵 <b>Processo: {pname}</b>\n"
            f"🕐 Avvio: {proc_start}"
        )

        # Serializza il lark.Tree su file temporaneo per passarlo a prepare_data.py
        tree_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        pickle.dump(proc["tree"], tree_file)
        tree_file.close()

        # 1. prepare_data.py
        prepare_args = ["--tree-file", tree_file.name] + proc.get("args", PREPARE_DATA_ARGS)
        if proc.get("no_interval", False):
            prepare_args.append("--no-interval")
        ok, out = run_script(pname, BASE_DIR / "data" / "prepare_data.py", prepare_args)
        Path(tree_file.name).unlink(missing_ok=True)
        proc_results.append({"step": "prepare_data.py", "success": ok})

        if not ok and stop_on_error:
            _log(f"🛑 [{pname}] stop_on_error, salto al prossimo processo.", telegram=True)
            all_results.append({"process": pname, "steps": proc_results})
            continue

        # 2. Optuna (sequenziale — produce i best params per le varianti)
        if OPTUNA_STEP.exists():
            ok, _ = run_notebook(pname, OPTUNA_STEP, proc_dir)
            proc_results.append({"step": OPTUNA_STEP.name, "success": ok})
            if not ok and stop_on_error:
                _log(f"🛑 [{pname}] stop_on_error, salto al prossimo processo.", telegram=True)
                all_results.append({"process": pname, "steps": proc_results})
                continue
        else:
            _log(f"⚠️ [{pname}] {OPTUNA_STEP.name} non trovato, saltato.", telegram=True)
            proc_results.append({"step": OPTUNA_STEP.name, "success": None})

        # 3. Varianti in parallelo — output mandato su Telegram in ordine 1→6
        _tg(f"⚡ <b>[{pname}]</b> Avvio {len(VARIANT_STEPS)} varianti in parallelo...")
        variant_results: dict[str, tuple[bool, str]] = {}
        with ThreadPoolExecutor(max_workers=len(VARIANT_STEPS)) as ex:
            futures = {
                ex.submit(run_notebook, pname, nb, proc_dir, send_telegram=False): nb
                for nb in VARIANT_STEPS if nb.exists()
            }
            for nb in VARIANT_STEPS:
                if not nb.exists():
                    _log(f"⚠️ [{pname}] {nb.name} non trovato, saltato.", telegram=True)
                    variant_results[nb.name] = (None, "")
            for future in as_completed(futures):
                nb = futures[future]
                ok, out = future.result()
                variant_results[nb.name] = (ok, out)

        # Manda i risultati in ordine 1→6
        for nb in VARIANT_STEPS:
            ok, out = variant_results.get(nb.name, (None, ""))
            elapsed_tag = ""
            if ok is True:
                _tg(
                    f"✅ <b>[{pname}] Completato</b>: {nb.name}\n\n"
                    f"<pre>{out}</pre>" if out else
                    f"✅ <b>[{pname}] Completato</b>: {nb.name}"
                )
            elif ok is False:
                _tg(f"❌ <b>[{pname}] ERRORE</b>: {nb.name}\n\n<pre>{out[-800:]}</pre>")
            proc_results.append({"step": nb.name, "success": ok})

        # Riepilogo del processo
        ok_n   = sum(1 for r in proc_results if r["success"] is True)
        fail_n = sum(1 for r in proc_results if r["success"] is False)
        _tg(
            f"📋 <b>{pname} — riepilogo</b>\n"
            + "\n".join(
                f"{'✅' if r['success'] else ('❌' if r['success'] is False else '⏭️')}  {r['step']}"
                for r in proc_results
            )
            + f"\n✅ {ok_n}  ❌ {fail_n}"
        )
        all_results.append({"process": pname, "steps": proc_results})

    # --- Riepilogo globale ---
    total_steps  = sum(len(p["steps"]) for p in all_results)
    total_ok     = sum(1 for p in all_results for s in p["steps"] if s["success"] is True)
    total_fail   = sum(1 for p in all_results for s in p["steps"] if s["success"] is False)
    proc_ok      = sum(1 for p in all_results if all(s["success"] is not False for s in p["steps"]))

    summary_lines = [f"\n{'='*60}", "RISULTATO FINALE", "="*60,
                     f"  Processi completati senza errori: {proc_ok}/{len(all_results)}",
                     f"  Step totali: {total_ok} OK / {total_fail} falliti / {total_steps} totali",
                     "", "  Dettaglio:"]
    for p in all_results:
        has_fail = any(s["success"] is False for s in p["steps"])
        summary_lines.append(f"  {'❌' if has_fail else '✅'}  {p['process']}")
    summary_lines += [f"  Report: {_report_path}", "="*60]
    _log("\n".join(summary_lines))

    _tg(
        f"🏁 <b>Pipeline completata</b>\n"
        f"📊 Processi OK: {proc_ok}/{len(all_results)}\n"
        f"Step: ✅ {total_ok}  ❌ {total_fail}\n\n"
        + "\n".join(
            f"{'❌' if any(s['success'] is False for s in p['steps']) else '✅'}  {p['process']}"
            for p in all_results
        )
        + f"\n\n📄 <code>{_report_path.name}</code>"
    )

    # JSON riepilogo
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "results": all_results}, f, indent=2)

    return total_ok, total_fail


if __name__ == "__main__":
    stop = "--stop-on-error" in sys.argv
    main(stop_on_error=stop)
