# AI Mail

AI Mail is a Python application that uses AI to interact with emails.
It replies to the mail sent to the configured email address. 

It utilizes Ollama, if you want a local machine elaboration (GPU required).
Or it can uses Groq, an API (free actually, with some limits) tha uses a remote Language Processing Unit and some Open Source LLMs, it has a very fast inference speed.

To install Ollama, please refer to the official GitHub repository: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)

To use Groq, check: https://groq.com

We use this into our university lab, so you will references of our team into the prompts, please modify as your needs

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Navigate into the project directory:
```bash
cd <project-directory>
```

3. Create a `.env` file in the root of your project and add your data:
```env
EMAIL_PASSWORD = your-app-password (i.e. for gmail needs to generate an app pw)
EMAIL = the email that you want to automate

IMAP = mail imap
SMTP = mail smtp

LANGCHAIN_API_KEY= your Langchain api key
GROQ_API_KEY= your Groq api key
```
Replace `'your-app-password'` with your actual app password.

To create an App Password for your Gmail account, follow these steps:

- Go to your Google Account.
- Select Security.
- Under "Signing in to Google," select App Passwords. You may need to sign in again.
- If you don’t have this option, it might be because:
  - 2-Step Verification is not set up for your account.
  - 2-Step Verification is only set up for security keys.
  - Your account is through work, school, or other organization.
  - You turned on Advanced Protection.
- At the bottom, choose Select app and choose the app you using and then Select device and choose the device you’re using and then Generate.
- Follow the instructions to enter the App Password. The App Password is the 16-character code in the yellow bar on your device.
- Copy the App Password and paste it into your `.env` file.

## Usage

The main functionality of the application is contained in the `AI_mail.py` script. The script uses the `os` module and `python-dotenv` to securely access your email password, and the `ChatOllama` and `OllamaEmbeddings` classes for AI functionalities, or the `ChatGroq` if you want to use the remote LPU

The `check_for_new_emails` function is where the main logic of the application resides.

In deployment, a `listen_base` function to give the ability to transcribe audio to text and summarize it, usefull for sharing report of meetings, also we would like to implement a RAG system based on the lab website, `generate_response_rag`.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
