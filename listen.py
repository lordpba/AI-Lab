"""

import openai

# Load the audio file
audio_file = "/path/to/audio/file.wav"

# Set up OpenAI Whisper API credentials
openai.api_key = "YOUR_API_KEY"

# Transcribe the audio file
response = openai.WhisperTranscription.create(
    audio=audio_file,
    language="en-US",
    format="txt"
)

# Save the transcription to a text file
text_file = "/path/to/output/file.txt"
with open(text_file, "w") as file:
    file.write(response.transcriptions[0].text)

print("Transcription saved to", text_file)
"""

import whisper

model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(".\Intervista 06 maggio.m4a")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)

text_file = "trascription.txt"
with open(text_file, "w") as file:
    file.write(result["text"])

print("Transcription saved to", text_file)