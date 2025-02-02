import speech_recognition as sr
import pyttsx3
import openai 
import os
from dotenv import load_dotenv

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()


load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

if not API_KEY:
    raise ValueError("ERROR: OPENAI_API_KEY not found in environment variables")

try:
    with sr.Microphone() as source:
        
        audio = recognizer.listen(source)

        if audio:
            print("We in business")

        user_input = recognizer.recognize_google(audio)

        response = openai.chat.completions.create(
                model="gpt-3.5 turbo",
                messages=[{"role": "user", "content": user_input}]
            )


        ai_response = response['choices'][0]['message']['content']
        tts_engine.say(ai_response)
        tts_engine.runAndWait()


except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")

except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
    
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")