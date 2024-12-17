import streamlit as st
import pandas as pd
import json

# Load settings
def load_settings():
    with open("settings.json", "r") as file:
        return json.load(file)

def save_settings(settings):
    with open("settings.json", "w") as file:
        json.dump(settings, file, indent=4)

# UI
st.title("Email Assistant Dashboard")

# Load settings
settings = load_settings()

# Sidebar settings
st.sidebar.header("Configurazione")
monitoring = st.sidebar.toggle("Monitoraggio Attivo", settings["monitoring_active"])
check_interval = st.sidebar.slider("Intervallo di Controllo (s)", 10, 300, settings["check_interval"])
model_choice = st.sidebar.radio("Seleziona il Modello", ["Groq", "OpenAI"], index=0 if settings["model"] == "Groq" else 1)

# Update settings
settings["monitoring_active"] = monitoring
settings["check_interval"] = check_interval
settings["model"] = model_choice
save_settings(settings)

# Display email logs
st.header("Log Email Processate")
try:
    df = pd.read_excel("emails_log.xlsx")
    st.dataframe(df)
except FileNotFoundError:
    st.info("Nessun log trovato.")
