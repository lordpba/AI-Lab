import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter





embedding = OllamaEmbeddings(model="llama3")
chroma_directory = './chroma/'

loader = WebBaseLoader("https://sites.google.com/unisi.it/interactiondesignlab/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)

# Create a new Chroma instance if it doesn't exist
if not os.path.exists(chroma_directory):
    vectordb = Chroma.from_documents(
        documents=['./archive/ai_lab_logs.txt'],
        embedding=embedding,
        persist_directory=chroma_directory
    )
else:
    vectordb = Chroma(chroma_directory)

# Funzione per generare una risposta utilizzando Chroma
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

# Funzione per salvare la domanda e la risposta nel vector store Chroma
def save_qa_to_vector_store(log_txt):
    #Crea un nuovo vettore e lo aggiunge al client Chroma
    collection = vectordb.get_or_create_collection("mail_collection")
    # Aggiungi il documento e il vettore al client Chroma
    collection.add(documents=[log_txt], embeddings=[embedding.embed_query(log_txt)])
    # Salva il vettore nel client Chroma
    collection.persist()
    return collection
