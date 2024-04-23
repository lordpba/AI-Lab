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
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
#from langchain_openai import ChatOpenAI
import datetime
from dotenv import load_dotenv
import os
from langsmith import traceable

# Load environment variables from .env file
load_dotenv()

# Configuration
imap_host = 'imap.gmail.com'
smtp_host = 'smtp.gmail.com'
email_account = os.getenv('EMAIL') # Get email from environment variable
email_password = os.getenv('EMAIL_PASSWORD')  # Get password from environment variable
llm = ChatOllama(model="llama3", temperature=0.2)
#llmOpenAi = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
#llm = llmOpenAi # se non si ha Ollama, si può usare OpenAI
embedding = OllamaEmbeddings(model="llama3")
chroma_directory = './chroma/'
"""
# Create a new Chroma instance if it doesn't exist
if not os.path.exists(chroma_directory):
    vectordb = Chroma.from_documents(
        documents='ai_lab_logs.txt',
        embedding=embedding,
        persist_directory=chroma_directory
    )
else:
    vectordb = Chroma(chroma_directory)
"""

# Funzione per controllare la presenza di nuove email e manda alla funzione process_email
def check_for_new_emails():
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
                process_email(email_message)
                # generate_response_using_chroma(email_mesclearsage, vectordb)
                # update_vector_store_and_reply(email_message)
    mail.close()
    mail.logout()

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


# Funzione per salvare la domanda e la risposta nel vector store Chroma
def save_qa_to_vector_store(log_txt):
    #Crea un nuovo vettore e lo aggiunge al client Chroma
    collection = vectordb.get_or_create_collection("mail_collection")
    # Aggiungi il documento e il vettore al client Chroma
    collection.add(documents=[log_txt], embeddings=[embedding.embed_query(log_txt)])
    # Salva il vettore nel client Chroma
    collection.persist()
    return collection


def generate_response_using_chroma(email_message: str, collection):
    #estrare il corpo dell'email
    body = get_email_body(email_message)
    print("Corpo dell'email: ", body)

    # Esegui una query nel vector store Chroma
    query_embedding = embedding.embed_query(body)
    search_results = collection.similarity_search(query_embedding, k=4)

    # Genera una risposta utilizzando i risultati della query e Langchain
    prompt_rag = ChatPromptTemplate.from_template("AI Lab306, l'assistente AI del laboratorio di Interaction Design.\
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
                                               mantenendo un linguaggio chiaro e professionale. {topic}"),
    
    chain = prompt_rag | llm | StrOutputParser()

    rag_response = (chain.invoke({"topic": search_results[0]['page_content'] }))
    save_qa_to_log(body, rag_response)
    print(rag_response)
    send_email_response(rag_response, email_message)

# take the body of the mail, split into chuncks, embed it, load into Chroma, and query it
def update_vector_store_and_reply(email_message):
    #load the logs of the previuos emails
    loader = TextLoader(file_path='ai_lab_logs.txt')
    documents = loader.load()
    #split the text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split(documents)
    #embed the chunks
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #load into Chroma
    db = Chroma.from_documents(documents=docs, embedding=embedding_function, persist_directory='chroma/')
    # query it
    # Estrarre il corpo dell'email
    body = get_email_body(email_message)
    print("Corpo dell'email: ", body)
    query = body
    docs = db.similarity_search(query)
    rag_response = docs[0].page_content
    # print results
    print(rag_response)
    save_qa_to_log(body, rag_response)
    send_email_response(rag_response, email_message)
    return docs


@traceable # Auto-trace this function
def process_email(email_message):
    # Estrarre il corpo dell'email
    body = get_email_body(email_message)
    print("Corpo dell'email: ", body)

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
    save_qa_to_log(body, llm_response)
    print(llm_response)
    send_email_response(llm_response, email_message)

def get_email_body(email_message):
    body = ""
    if email_message.is_multipart():
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
    return body.decode('utf-8', errors='ignore')

def send_email_response(response, email_message):
    msg = MIMEText(response)
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
            recipients += email_message['Cc'].split(',')
        smtp.send_message(msg, to_addrs=recipients)

if __name__ == '__main__':
    while True:
        check_for_new_emails()
        print("Attesa per il prossimo controllo...")
        time.sleep(300)  # Pausa di 5 minuti
