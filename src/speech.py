"""
Steps
1. Speech to Text (STT) 
2. Process Text 
3. Text to Speech (TTS)
"""

import openai
import os
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv

# Initialize Text to Speech Engine
tts_engine = pyttsx3.init()

# Get API Key
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Speech Recognition
recognizer = sr.Recognizer()

def stt_to_response_to_tts():
    with sr.Microphone as source:
        try:
            audio = recognizer.listen(source)
            user_input = recognizer.recognize_google(audio)

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_input}]
            )

            ai_response = response['choices'][0]['message']['content']

            tts_engine.say(ai_response)
            tts_engine.runAndWait()

        except Exception as ex:
            tts_engine.say("Sorry I couldn't understand that; may you please repeat yourself")


if __name__ == "__main__":
    stt_to_response_to_tts()