import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import time
import pandas as pd
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from langsmith import traceable
import logging
import threading

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables from .env file
load_dotenv()

# Configuration
imap_host = os.getenv('IMAP')
smtp_host = os.getenv('SMTP')
email_account = os.getenv('EMAIL')
email_password = os.getenv('EMAIL_PASSWORD')
check_interval = int(os.getenv('CHECK_INTERVAL', 30))  # Check interval in seconds (default: 30)

# LLM Setup
groq = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
llm = groq

@traceable
def process_email(body):
    """Generate LLM response based on the email body."""
    prompt = ChatPromptTemplate.from_template(
        "AI Lab306, the AI assistant of the Interaction Design laboratory. "
        "You know the specializations of our team: "
        "Antonio in Interaction Design, Mario in NLP, Giulia in mathematics and AI, "
        "Anna in UX Design, Leonardo B. in web design, Leonardo G. in programming, "
        "Martina in marketing, and Claudia in data analysis. "
        "Your role is to offer advice based on these skills, analyze data, facilitate communication, "
        "and coordinate the team. Respond to emails with a positive and motivational tone, "
        "always write in the same language of the question, maintaining a clear and professional language. {topic}"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"topic": body})


def check_for_new_emails():
    """Check for new unread emails and return the email message."""
    try:
        mail = imaplib.IMAP4_SSL(imap_host)
        mail.login(email_account, email_password)
        mail.select('inbox')
        status, messages = mail.search(None, 'UNSEEN')
        if status == 'OK':
            for num in messages[0].split():
                status, data = mail.fetch(num, '(RFC822)')
                if status == 'OK':
                    email_message = email.message_from_bytes(data[0][1])
                    mail.store(num, '+FLAGS', '\\Seen')  # Mark email as read
                    return email_message
    except Exception as e:
        logging.error("Error checking emails: %s", e)
    finally:
        mail.logout()
    return None


def get_email_body(email_message):
    """Extract the plain or HTML text body from the email."""
    body = ""
    try:
        if email_message and email_message.is_multipart():
            for part in email_message.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype in ['text/plain', 'text/html'] and 'attachment' not in cdispo:
                    body = part.get_payload(decode=True)
                    break
        else:
            body = email_message.get_payload(decode=True)
    except Exception as e:
        logging.error("Error extracting email body: %s", e)
    return body.decode('utf-8', errors='ignore') if body else ""


def send_email_response(response, email_message):
    """Send the LLM response as a reply email."""
    try:
        msg = MIMEText(response, 'plain', 'utf-8')
        msg['Subject'] = 'Automatic Response from AiLab306'
        msg['From'] = email_account
        msg['To'] = email_message['From']
        if 'Cc' in email_message:
            msg['Cc'] = email_message['Cc']
        recipients = [email_message['From']] + (email_message.get('Cc', '').split(','))
        with smtplib.SMTP_SSL(smtp_host, 465) as smtp:
            smtp.login(email_account, email_password)
            smtp.send_message(msg, to_addrs=recipients)
        logging.info("Response sent successfully to: %s", email_message['From'])
    except Exception as e:
        logging.error("Error sending email response: %s", e)


def save_qa_to_log(question, answer):
    """Save the email question and LLM answer to an Excel and text log file."""
    data = {'Question': [question], 'Answer': [answer]}
    try:
        df = pd.DataFrame(data)
        try:
            existing_df = pd.read_excel('ai_lab_logs.xlsx')
            updated_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            updated_df = df
        updated_df.to_excel('ai_lab_logs.xlsx', index=False)
        with open('ai_lab_logs.txt', 'a', encoding='utf-8') as f:
            f.write(f'Question: {question}\nAnswer: {answer}\n\n')
        logging.info("Question and answer saved to log files.")
    except Exception as e:
        logging.error("Error saving to log files: %s", e)


def process_incoming_emails():
    """Main function to process unread emails."""
    email_message = check_for_new_emails()
    if email_message:
        logging.info("New email received from: %s", email_message['From'])
        body = get_email_body(email_message)
        if body:
            logging.info("Processing email body.")
            response = process_email(body)
            logging.info("Generated LLM response.")
            send_email_response(response, email_message)
            save_qa_to_log(body, response)
        else:
            logging.warning("Email body is empty. Skipping.")
    else:
        logging.info("No new emails found.")


def email_check_loop():
    """Loop to continuously check for new emails."""
    while True:
        process_incoming_emails()
        time.sleep(check_interval)


if __name__ == "__main__":
    logging.info("Starting AiLab306 Email Assistant...")
    email_thread = threading.Thread(target=email_check_loop, daemon=True)
    email_thread.start()
    email_thread.join()
