import streamlit as st
import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import pandas as pd
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from threading import Thread

# Load environment variables
load_dotenv()
imap_host = os.getenv('IMAP')
smtp_host = os.getenv('SMTP')
email_account = os.getenv('EMAIL')
email_password = os.getenv('EMAIL_PASSWORD')

# Define LLM models
groq = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Helper Functions
def check_emails():
    """Check for unread emails."""
    try:
        mail = imaplib.IMAP4_SSL(imap_host)
        mail.login(email_account, email_password)
        mail.select('inbox')
        status, messages = mail.search(None, 'UNSEEN')
        if status == 'OK':
            email_data = []
            for num in messages[0].split():
                status, data = mail.fetch(num, '(RFC822)')
                if status == 'OK':
                    email_message = email.message_from_bytes(data[0][1])
                    subject = email_message.get("Subject", "No Subject")
                    sender = email_message.get("From", "Unknown Sender")
                    body = get_email_body(email_message)
                    email_data.append({"From": sender, "Subject": subject, "Body": body})
            return email_data
    except Exception as e:
        st.error(f"Errore nel controllo email: {e}")
    return []

def get_email_body(email_message):
    """Extract email body."""
    body = ""
    if email_message.is_multipart():
        for part in email_message.walk():
            ctype = part.get_content_type()
            if ctype == 'text/plain' and 'attachment' not in str(part.get('Content-Disposition')):
                body = part.get_payload(decode=True)
                break
    else:
        body = email_message.get_payload(decode=True)
    return body.decode('utf-8', errors='ignore') if body else "No Body"

def process_email_with_llm(body, model):
    """Generate LLM response."""
    llm = groq if model == "Groq" else openai
    prompt = ChatPromptTemplate.from_template(
        "AI Lab306, rispondi in modo chiaro, professionale e motivazionale. {topic}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": body})

def send_response(to, response):
    """Send an email response."""
    try:
        msg = MIMEText(response, 'plain', 'utf-8')
        msg['Subject'] = 'Automatic Response from AiLab306'
        msg['From'] = email_account
        msg['To'] = to
        with smtplib.SMTP_SSL(smtp_host, 465) as smtp:
            smtp.login(email_account, email_password)
            smtp.send_message(msg)
        st.success("Risposta inviata con successo!")
    except Exception as e:
        st.error(f"Errore nell'invio della risposta: {e}")

# Streamlit UI
st.title("Email Assistant Dashboard")
st.sidebar.header("Configurazione")

# Options in Sidebar
model_choice = st.sidebar.radio("Seleziona il modello LLM", ["Groq", "OpenAI"])
check_interval = st.sidebar.slider("Intervallo di controllo (secondi)", 10, 300, 60)
monitor_emails = st.sidebar.toggle("Attiva monitoraggio email")

# Countdown Timer
if monitor_emails:
    countdown = st.empty()
    stop_flag = False

    def monitor_loop():
        while monitor_emails:
            countdown.write(f"Prossimo controllo tra {check_interval} secondi...")
            email_data = check_emails()
            if email_data:
                st.write("📧 Nuove email trovate:")
                for idx, mail in enumerate(email_data):
                    st.subheader(f"Email #{idx+1}")
                    st.text(f"Da: {mail['From']}")
                    st.text(f"Oggetto: {mail['Subject']}")
                    st.text_area("Contenuto", mail['Body'], height=200)
                    if st.button(f"Genera risposta per Email #{idx+1}", key=idx):
                        response = process_email_with_llm(mail['Body'], model_choice)
                        st.text_area("Risposta generata", response, height=200)
                        if st.button(f"Invia risposta per Email #{idx+1}", key=f"send_{idx}"):
                            send_response(mail['From'], response)
            time.sleep(check_interval)
    
    Thread(target=monitor_loop, daemon=True).start()

else:
    st.write("Monitoraggio email disattivato. Attivalo dalla sidebar.")

st.write("📊 **Log delle email processate**")
if os.path.exists('ai_lab_logs.xlsx'):
    log_df = pd.read_excel('ai_lab_logs.xlsx')
    st.dataframe(log_df)
else:
    st.info("Nessun log trovato. Inizia a processare le email!")
