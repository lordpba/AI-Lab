import time
import json
import imaplib
import email
from email.mime.text import MIMEText
import smtplib
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
imap_host = os.getenv('IMAP')
smtp_host = os.getenv('SMTP')
email_account = os.getenv('EMAIL')
email_password = os.getenv('EMAIL_PASSWORD')

# LLM models
groq = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
openai = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

def load_settings():
    """Load monitoring settings."""
    with open("settings.json", "r") as file:
        return json.load(file)

def save_log(data):
    """Save email logs to Excel."""
    df = pd.DataFrame(data)
    if os.path.exists("emails_log.xlsx"):
        existing = pd.read_excel("emails_log.xlsx")
        df = pd.concat([existing, df], ignore_index=True)
    df.to_excel("emails_log.xlsx", index=False)

def check_emails():
    """Check unread emails."""
    try:
        mail = imaplib.IMAP4_SSL(imap_host)
        mail.login(email_account, email_password)
        mail.select('inbox')
        status, messages = mail.search(None, 'UNSEEN')
        emails = []
        if status == 'OK':
            for num in messages[0].split():
                status, data = mail.fetch(num, '(RFC822)')
                if status == 'OK':
                    msg = email.message_from_bytes(data[0][1])
                    body = get_email_body(msg)
                    emails.append({
                        "From": msg["From"],
                        "Subject": msg["Subject"],
                        "Body": body
                    })
        return emails
    except Exception as e:
        print(f"Error: {e}")
    return []

def get_email_body(msg):
    """Extract email body."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    return msg.get_payload(decode=True).decode()

def process_email_with_llm(body, model):
    """Generate LLM response."""
    llm = groq if model == "Groq" else openai
    return llm.invoke(body)

if __name__ == "__main__":
    print("Email Monitor Started...")
    while True:
        settings = load_settings()
        if settings["monitoring_active"]:
            emails = check_emails()
            if emails:
                for mail in emails:
                    response = process_email_with_llm(mail["Body"], settings["model"])
                    mail["Response"] = response
                save_log(emails)
                print("Processed and logged emails.")
        else:
            print("Monitoring paused...")
        time.sleep(settings["check_interval"])
