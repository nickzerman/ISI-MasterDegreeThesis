"""
Orchestratore della pipeline sperimentale — SOLO processo Sepsi.

Identico a pipeline_runner.py nell'orchestrazione (prepare → optuna → varianti in
parallelo, Telegram, checkpoint/resume), ma l'unico processo è la Sepsi.

A differenza della pipeline principale, lo step di preparazione NON è il generico
data/prepare_data.py (sintetico) ma data/sepsis_prepare_data.ipynb, eseguito via
papermill con parametri iniettati. Il notebook fa, ad OGNI processo:
  • discovery del processo dal log XES con Inductive Miner (→ SESE → tree n-ario);
  • ALIGNMENT delle ~1050 tracce reali del log sulla rete di Petri;
  • classificatori XOR/Loop allenati sulle tracce REALI allineate;
  • distribuzioni dei tempi ricavate dai timestamp REALI (log1p) + regressori;
  • generazione delle tracce sintetiche condizionate e salvataggio.

Griglia (24 processi):
  • Tracce generate:  15k, 50k, 100k   (N_TRACES_GENERATED)
  • Intervalli:       SEMPRE disattivati (NO_INTERVAL=True)
  • Clustering:       nocluster (CLUSTER=False) e cluster (CLUSTER=True)
  • Coverage:         4 angoli binari (loop_coverage, xor_coverage):
                        (0,0)  LOOP=False, XOR=False
                        (0,1)  LOOP=False, XOR=True,  COVERAGE=1.0  (solo XOR)
                        (1,0)  LOOP=True,  XOR=False, COVERAGE=1.0  (solo Loop)
                        (1,1)  LOOP=True,  XOR=True,  COVERAGE=1.0  (Loop + XOR)

Per ogni processo esegue in sequenza:
  1. data/sepsis_prepare_data.ipynb  (papermill, parametri iniettati)
  2. experiments/run_all_optuna.ipynb
  3. variants/variant*.ipynb  (tutti in parallelo)

I notebook leggono sempre data/prepared_data.pt, che viene riscritto ad ogni processo.

Uso:
    python pipeline_runner_sepsis.py
    python pipeline_runner_sepsis.py --stop-on-error

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
# Results dir DEDICATA alla Sepsi: i notebook variante scrivono all_variants_results.json
# in RESULTS_DIR (config.py la legge dalla env var RESULTS_DIR). Puntando l'intero run a
# results_sepsis/ il checkpoint resta isolato da quello della pipeline principale.
RESULTS_DIR = BASE_DIR / "results_sepsis"
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
# Griglia processi Sepsi: 3 tracce x 4 angoli di coverage = 12 processi.
#
# Lo step di preparazione esegue data/sepsis_prepare_data.ipynb via papermill,
# che fa discovery (Inductive Miner) + ALIGNMENT delle tracce reali sulla rete +
# classificatori XOR/Loop allenati sulle tracce reali allineate + distribuzioni
# dei tempi dai timestamp reali. Coverage / LOOP / XOR / N_TRACES sono iniettati.
# ---------------------------------------------------------------------------
def get_processes():
    # 4 angoli binari (loop_coverage, xor_coverage). COVERAGE è 1.0 quando la
    # regione è attiva; LOOP/XOR=False la disattivano (equivale a coverage 0).
    coverage_corners = [
        ("ll0_xl0", dict(LOOP=False, XOR=False, COVERAGE=1.0)),  # (0,0)
        ("ll0_xl1", dict(LOOP=False, XOR=True,  COVERAGE=1.0)),  # (0,1) solo XOR
        ("ll1_xl0", dict(LOOP=True,  XOR=False, COVERAGE=1.0)),  # (1,0) solo Loop
        ("ll1_xl1", dict(LOOP=True,  XOR=True,  COVERAGE=1.0)),  # (1,1) Loop + XOR
    ]

    processes = []
    for n in [15000, 50000, 100000]:
        for cluster, ctag in [(False, "nocluster"), (True, "cluster")]:
            for cov_name, kw in coverage_corners:
                name = f"sepsis_{n // 1000}k_{ctag}_nointerval_{cov_name}"
                params = {
                    "N_TRACES_GENERATED": n,
                    "NO_INTERVAL":        True,   # intervalli sempre disattivati
                    "CLUSTER":            cluster,
                    **kw,
                }
                processes.append({"name": name, "params": params})
    return processes

PROCESSES = get_processes()

# Step 1 della pipeline Sepsi: il notebook di preparazione (alignment + tempi reali).
SEPSIS_PREP_STEP = DATA_DIR / "sepsis_prepare_data.ipynb"
# Output che varianti e optuna leggono (DATA_DIR/prepared_data.pt).
PREPARED_DATA_OUT = DATA_DIR / "prepared_data.pt"
# Generazione 100k tracce + alignment di 1050 tracce reali: tetto di sicurezza ampio.
PREP_TIMEOUT = 10800  # 3 ore

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
    env["RESULTS_DIR"] = str(RESULTS_DIR)  # i notebook variante scrivono il checkpoint qui
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


def run_notebook(name: str, nb_path: Path, output_dir: Path, timeout: int | None = NOTEBOOK_TIMEOUT,
                 parameters: dict | None = None) -> tuple[bool, str]:
    """Esegue il notebook in un subprocess isolato con nuovo process group.
    Al termine (successo, errore o timeout) garantisce la terminazione dell'intero
    albero di processi figlio — incluso il kernel Jupyter — su Linux e Windows.
    Se `parameters` è passato, viene iniettato da papermill nella cella 'parameters'."""
    output_path = output_dir / nb_path.name
    _log(f"▶ [{name}] Avvio notebook: {nb_path.name}")
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)
    env["RESULTS_DIR"] = str(RESULTS_DIR)  # i notebook variante scrivono il checkpoint qui

    pm_script = (
        "import papermill as pm\n"
        f"pm.execute_notebook(\n"
        f"    {repr(str(nb_path))},\n"
        f"    {repr(str(output_path))},\n"
        f"    kernel_name='python3',\n"
        f"    progress_bar=False,\n"
        f"    request_save_on_cell_execute=True,\n"
        f"    cwd={repr(str(BASE_DIR))},\n"
        f"    parameters={repr(parameters or {})},\n"
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

        # ── 1. sepsis_prepare_data (notebook via papermill) ───────────────
        # Discovery + alignment tracce reali + classificatori XOR/Loop su tracce
        # reali + tempi dai timestamp reali. Output -> DATA_DIR/prepared_data.pt
        # (lo stesso file che leggono optuna e le varianti).
        prep_params = {**proc["params"], "OUTPUT_PATH": str(PREPARED_DATA_OUT)}

        tmp_prepare = _tg_send(
            f"⏳ <b>[{pname}]</b> sepsis_prepare_data in corso...\n<code>{prep_params}</code>",
            silent=True,
        )
        ok, out = run_notebook(pname, SEPSIS_PREP_STEP, proc_dir,
                               timeout=PREP_TIMEOUT, parameters=prep_params)
        if ok:
            _tg_edit(tmp_prepare, f"✅ <b>[{pname}] sepsis_prepare_data</b>\n\n<pre>{html.escape(out[-800:])}</pre>" if out else f"✅ <b>[{pname}] sepsis_prepare_data</b>")
        else:
            _tg_edit(tmp_prepare, f"❌ <b>[{pname}] sepsis_prepare_data ERRORE</b>\n\n<pre>{html.escape(out[-800:])}</pre>")
        proc_results.append({"step": "sepsis_prepare_data.ipynb", "success": ok})

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
