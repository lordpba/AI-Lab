import whisper
import time

start_time = time.time()

model = whisper.load_model("medium")
result = model.transcribe("Intervista_06_maggio.m4a")
print(result["text"])

text_file = "trascription.txt"
with open(text_file, "w") as file:
    file.write(result['text'])

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")