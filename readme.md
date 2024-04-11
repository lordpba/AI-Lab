# AI Mail

AI Mail is a Python application that uses AI to interact with emails.
It replies to the mail sent to the configured email address. It utilizes Ollama, as AI framework.
To install Ollama, please refer to the official GitHub repository: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
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

3. Create a `.env` file in the root of your project and add your email password:
```env
EMAIL_PASSWORD=your-app-password
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

The main functionality of the application is contained in the `AI_mail.py` script. The script uses the `os` module and `python-dotenv` to securely access your email password, and the `ChatOllama` and `OllamaEmbeddings` classes for AI functionalities.

The `check_for_new_emails` function is where the main logic of the application resides.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
