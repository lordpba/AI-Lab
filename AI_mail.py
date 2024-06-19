import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langsmith import traceable
#from langchain_openai import ChatOpenAI
#from langchain_ollama import ChatOllama

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

# Check for unread emails and return email_message
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
                # Extract the email content here and pass it to the LLM
                # process_email(email_message)
                # generate_response_using_chroma(email_message, vectordb)
                # update_vector_store_and_reply(email_message)
    mail.close()
    mail.logout()
    return(email_message)

# Extract the body from the email and return body
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
        print("Email body: ", body)

    return body.decode('utf-8', errors='ignore')

# Process the email body and return llm_response  
@traceable # Auto-trace this function  
def process_email(body):
    
    prompt = ChatPromptTemplate.from_template("AI Lab306, the AI assistant of the Interaction Design laboratory.\
                                               You know the specializations of our team:\
                                               Antonio in Interaction Design,\
                                               Mario in NLP,\
                                               Giulia in mathematics and AI,\
                                               Anna in UX Design,\
                                               Leonardo B. in web design,\
                                               Leonardo G. in programming,\
                                               Martina in marketing,\
                                               and Claudia in data analysis.\
                                               Your role is to offer advice based on these skills,\
                                               analyze data, facilitate communication and coordinate the team.\
                                               Respond to emails with a positive and motivational tone,\
                                                always write in Italian\
                                               maintaining a clear and professional language. {topic}")

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

# Send llm_response as a response to the email
def send_email_response(llm_response, email_message):
    msg = MIMEText(llm_response)
    msg['Subject'] = 'Automatic Response from AiLab306'
    msg['From'] = email_account
    msg['To'] = email_message['From']
    
    # Add recipients in CC if present
    if 'Cc' in email_message:
        msg['Cc'] = email_message['Cc']
    
    with smtplib.SMTP_SSL(smtp_host, 465) as smtp:  # Note the use of SMTP_SSL and port 465
        smtp.login(email_account, email_password)
        # Include both 'To' and 'Cc' recipients when sending the message
        recipients = [email_message['From']]
        if 'Cc' in email_message:
            recipients += [email.strip() for email in email_message['Cc'].split(',') if email.strip() != email_account]
        smtp.send_message(msg, to_addrs=recipients)

# Function to save the question and answer in an excel log file and 
def save_qa_to_log(question, answer):
    data = {'Question': [question], 'Answer': [answer]}
    df = pd.DataFrame(data)
    try:
        existing_df = pd.read_excel('ai_lab_logs.xlsx')
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_excel('ai_lab_logs.xlsx', index=False)
    except FileNotFoundError:
        df.to_excel('ai_lab_logs.xlsx', index=False)
    except Exception as e:
        print("Error while saving to log file:", str(e))
    else:
        print("Saving to log file completed successfully.")
    #save also in a text file
    with open('ai_lab_logs.txt', 'a') as f:
        f.write(f'Question: {question}\nAnswer: {answer}\n\n')
        #save_qa_to_vector_store(f)


# main loop
while True:
    print("Waiting for the next check...")
    # Perform the email check
    email_message = check_for_new_emails(email_account)
    # If there is a new email, process it
    if email_message is not None:
        # Extract the body of the email
        body = get_email_body(email_message)
        print("Email body: ", body)
        # Process the body of the email
        llm_response = process_email(body)
        print("Response generated by LLM: ", llm_response)
        # Send the response to the email
        send_email_response(llm_response, email_message)
        print("Response sent successfully.")
        # Save the question and answer in the log file
        # save_qa_to_log(body, llm_response)
        # print("Question and answer saved in the log file.")
    time.sleep(300)  # Pause for 5 minutes
