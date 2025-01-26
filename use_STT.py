# Whisper model
import os
from groq import Groq

client = Groq()
filename = "output.wav"

def get_STT_response(filename):
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3",
        response_format="verbose_json",
        )
    print(transcription.text)
   
    return transcription.text
      