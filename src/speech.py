import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import os
from dotenv import load_dotenv

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()


load_dotenv()
API_KEY = os.getenv('DEEPSEEK_KEY')

if not API_KEY:
    raise ValueError("ERROR: DEEPSEEK_KEY not found in environment variables")

try:
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

        if audio:
            print("Audio captured successfully")

        user_input = recognizer.recognize_google(audio)
        print(f"You said {user_input}")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
            {
                "role": "user",
                "content": user_input
            }
        ]
        )

        if response.choices:
            ai_message = response.choices[0].message.content
            if ai_message:
                ai_response = ai_message
            else:
                ai_response = "No content returned from the AI."
        else:
            ai_response = "No choices available in the response."


        print(f"AI Response: {ai_response}")
        tts_engine.say(ai_response)
        tts_engine.runAndWait()


except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio.")

except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
    
except Exception as ex:
    print(f"An unexpected error occurred: {ex}")