import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import time
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import openpyxl
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain_openai import ChatOpenAI
import datetime
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables from .env file
load_dotenv()

# Configuration
imap_host = os.getenv('IMAP')
smtp_host = os.getenv('SMTP')
email_account = os.getenv('EMAIL') # Get email from environment variable
email_password = os.getenv('EMAIL_PASSWORD')  # Get password from environment variable

# Set up the LLM models
#ollama = ChatOllama(model="llama3", temperature=0.0, stop=["<|start_header_id|>", "<|end_header_id|>", "<eot_id>", "<|reserved_special_token"])
#gpt3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
groq = ChatGroq(temperature=0.0, model_name="llama3-8b-8192") # mixtral-8x7b-32768 - llama3-70b-8192 - gemma-7b-it - llama3-8b-8192
llm = groq



# controlla mail non lette e returna email_message
def check_for_new_emails(email_account):
    email_message = None
    mail = imaplib.IMAP4_SSL(imap_host)
    mail.login(email_account, email_password)
    mail.select('inbox')
    status, messages = mail.search(None, 'UNSEEN')
    if status == 'OK':
        for num in messages[0].split():
            status, data = mail.fetch(num, '(RFC822)')
            if status == 'OK':
                email_message = email.message_from_bytes(data[0][1])
                # Estrai il contenuto dell'email qui e passalo al LLM
                # process_email(email_message)
                # generate_response_using_chroma(email_mesclearsage, vectordb)
                # update_vector_store_and_reply(email_message)
    mail.close()
    mail.logout()
    return(email_message)

# dalla mail estrae il corpo e returna body
def get_email_body(email_message):
    body = ""
    if email_message is not None and email_message.is_multipart():
        for part in email_message.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = email_message.get_payload(decode=True)
        print("Corpo dell'email: ", body)

    return body.decode('utf-8', errors='ignore')

# processa il corpo dell'email e returna llm_response  
@traceable # Auto-trace this function  
def process_email(body):
    
    prompt = ChatPromptTemplate.from_template("AI Lab306, l'assistente AI del laboratorio di Interaction Design.\
                                               Conosci le specializzazioni del nostro team:\
                                               Antonio in Interaction Design,\
                                               Mario in NLP,\
                                               Giulia in matematica e AI,\
                                               Anna in UX Design,\
                                               Leonardo B. in web design,\
                                               Leonardo G. in programmazione,\
                                               Martina in marketing,\
                                               e Claudia in analisi dei dati.\
                                               Il tuo ruolo è offrire consigli basati su queste competenze,\
                                               analizzare dati, facilitare la comunicazione e coordinare il team.\
                                               Rispondi alle e-mail con un tono positivo e motivazionale,\
                                                scrivi sempre in lingua italiana\
                                               mantenendo un linguaggio chiaro e professionale. {topic}")

    # using LangChain Expressive Language chain syntax
    # learn more about the LCEL on
    # /docs/expression_language/why
    chain = prompt | llm | StrOutputParser()
    
    # for brevity, response is printed in terminal
    # You can use LangServe to deploy your application for
    # production
    #llm_response_from_chroma = generate_response_using_chroma(body)
    llm_response = (chain.invoke({"topic": body }))
    return(llm_response)

# Invia llm_response come risposta all'email
def send_email_response(llm_response, email_message):
    msg = MIMEText(llm_response)
    msg['Subject'] = 'Risposta Automatica da AiLab306'
    msg['From'] = email_account
    msg['To'] = email_message['From']
    
    # Aggiungi destinatari in CC se presenti
    if 'Cc' in email_message:
        msg['Cc'] = email_message['Cc']
    
    with smtplib.SMTP_SSL(smtp_host, 465) as smtp:  # Nota l'uso di SMTP_SSL e la porta 465
        smtp.login(email_account, email_password)
        # Includi sia i destinatari 'To' che 'Cc' quando invii il messaggio
        recipients = [email_message['From']]
        if 'Cc' in email_message:
            recipients += [email.strip() for email in email_message['Cc'].split(',') if email.strip() != email_account]
        smtp.send_message(msg, to_addrs=recipients)

# Funzione per salvare la domanda e la risposta in un file di log excel e 
def save_qa_to_log(question, answer):
    data = {'Domanda': [question], 'Risposta': [answer]}
    df = pd.DataFrame(data)
    try:
        existing_df = pd.read_excel('ai_lab_logs.xlsx')
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel('ai_lab_logs.xlsx', index=False)
    except FileNotFoundError:
        df.to_excel('ai_lab_logs.xlsx', index=False)
    except Exception as e:
        print("Errore durante il salvataggio nel file di log:", str(e))
    else:
        print("Salvataggio nel file di log completato con successo.")
    #save also in a text file
    with open('ai_lab_logs.txt', 'a') as f:
        f.write(f'Domanda: {question}\nRisposta: {answer}\n\n')
        #save_qa_to_vector_store(f)






# main loop
while True:
        print("Attesa per il prossimo controllo...")
        # Esegui il controllo delle email
        email_message = check_for_new_emails(email_account)
        # Estrai il corpo dell'email
        body = get_email_body(email_message)
        print("Corpo dell'email: ", body)
        # Processa il corpo dell'email
        llm_response = process_email(body)
        print("Risposta generata da LLM: ", llm_response)
        # Invia la risposta all'email
        send_email_response(llm_response, email_message)
        print("Risposta inviata con successo.")
        # Salva la domanda e la risposta nel file di log
        save_qa_to_log(body, llm_response)
        print("Domanda e risposta salvate nel file di log.")
        time.sleep(300)  # Pausa di 5 minuti
