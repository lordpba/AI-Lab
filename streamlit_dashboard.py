import streamlit as st
import pandas as pd
import json
import os
import subprocess

# Percorsi dei file nella cartella AI-Lab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
LOG_FILE = os.path.join(BASE_DIR, "email_log.xlsx")
MONITOR_SCRIPT = os.path.join(BASE_DIR, "email_monitor.py")

# Helper Functions
def load_settings():
    with open(SETTINGS_FILE, "r") as file:
        return json.load(file)

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as file:
        json.dump(settings, file, indent=4)

def start_monitoring():
    """Avvia il processo di monitoraggio."""
    return subprocess.Popen(["python", MONITOR_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def stop_monitoring(process):
    """Termina il processo di monitoraggio."""
    if process:
        process.terminate()
        st.success("Monitoraggio terminato!")

# Streamlit Dashboard
st.title("Email Assistant Dashboard")

# Carica impostazioni
settings = load_settings()

# Sidebar settings
st.sidebar.header("Configurazione")
monitoring_active = st.sidebar.toggle("Monitoraggio Attivo", settings["monitoring_active"])
check_interval = st.sidebar.slider("Intervallo di Controllo (s)", 10, 300, settings["check_interval"])
model_choice = st.sidebar.radio("Seleziona il Modello", ["Groq", "OpenAI"], index=0 if settings["model"] == "Groq" else 1)

# Aggiorna impostazioni
settings["monitoring_active"] = monitoring_active
settings["check_interval"] = check_interval
settings["model"] = model_choice
save_settings(settings)

# Controllo del processo di monitoraggio
st.sidebar.subheader("Controllo Script Monitoraggio")
if "monitor_process" not in st.session_state:
    st.session_state["monitor_process"] = None

if monitoring_active:
    if not st.session_state["monitor_process"]:
        st.session_state["monitor_process"] = start_monitoring()
        st.success("Monitoraggio avviato!")
    else:
        st.info("Monitoraggio già attivo.")
else:
    if st.session_state["monitor_process"]:
        stop_monitoring(st.session_state["monitor_process"])
        st.session_state["monitor_process"] = None
    else:
        st.info("Monitoraggio già fermo.")

# Visualizza log delle email
st.header("Log Email Processate")
try:
    df = pd.read_excel(LOG_FILE)
    st.dataframe(df)
except FileNotFoundError:
    st.info("Nessun log trovato.")
