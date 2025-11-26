import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization

# 1. Creazione di un DataFrame con Nomi Standard
# Usiamo nomi semplici che poi mapperemo ai nomi attesi da PM4Py
data = [
    ['C1', 'Start', '2023-10-01 09:00:00'],
    ['C1', 'A', '2023-10-01 09:10:00'],
    ['C2', 'Start', '2023-10-02 10:00:00'],
    ['C2', 'B', '2023-10-02 10:15:00'],
    ['C2', 'End', '2023-10-02 10:30:00'],
]
# Nomi delle colonne NON PM4Py (che useremo per la ridenominazione)
column_names_simple = ['Case ID', 'Activity', 'Timestamp']
df = pd.DataFrame(data, columns=column_names_simple)


# 2. Ridenominazione Esplicita delle Colonne (Manuale)
# Questo è il passaggio chiave che bypassa le funzioni mancanti di utilità.
# PM4Py si aspetta attributi specifici per Case ID, Attività e Timestamp.
column_mapping = {
    'Case ID': 'case:concept:name',    # ID del caso/processo
    'Activity': 'concept:name',        # Nome dell'attività
    'Timestamp': 'time:timestamp'      # Momento in cui l'attività è avvenuta
}
df = df.rename(columns=column_mapping)


# 3. Formattazione dei Tipi di Dati e Ordinamento
# Assicurati che il Timestamp sia un oggetto datetime (essenziale)
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
df = df.sort_values('time:timestamp')


# 4. Conversione in Event Log PM4Py
# Ora il convertitore può lavorare correttamente perché le colonne hanno i nomi attesi
log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)

# 4. Discovery del DFG (Directly-Follows Graph)
dfg = dfg_discovery.apply(log)

# 5. Creazione e Salvataggio del DFG
# Crea l'oggetto di visualizzazione
gviz = dfg_visualization.apply(
    dfg,
    log=log,
    variant=dfg_visualization.Variants.FREQUENCY
)

# SALVA L'IMMAGINE IN UN FILE
# Verrà creato un file 'dfg_test.png' nella stessa cartella dello script.
dfg_visualization.save(gviz, "dfg_test.png")

# Puoi anche usare un formato vettoriale come SVG per una qualità migliore:
# dfg_visualization.save(gviz, "dfg_test.svg")

print("Esecuzione completata.")
print("*** Verifica se è stato creato il file 'dfg_test.png' nella cartella dello script. ***")
