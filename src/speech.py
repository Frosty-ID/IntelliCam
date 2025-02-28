import speech_recognition as sr
import pyttsx3
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()


load_dotenv()
API_KEY = os.getenv('DEEPSEEK_KEY')

if not API_KEY:
    raise ValueError("ERROR: DEEPSEEK_KEY not found in environment variables")

client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY
    )


def listen_for_audio():
    try:
        with sr.Microphone() as source:
            print("Listening, say something")
            audio = recognizer.listen(source)
            user_input = recognizer.recognize_google(audio)
            print(f"You said {user_input}")
            return user_input
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as ex:
        print(f"An unexpected error occurred while listening: {ex}")
    


def generate_ai_response(user_input: str):
    try:
        response = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[{
                "role": "user", 
                "content": user_input
            }]
        )

        if response.choices:
            ai_response = response.choices[0].message.content
            return ai_response
        else:
            print("I'm sorry, I couldn't process your request.")
            sys.exit(1)
    except Exception as ex:
        print(f"Error: {ex}")
        sys.exit(1)

def speak_response(ai_response: str):
    try:
        tts_engine.say(ai_response)
        tts_engine.runAndWait()
    except Exception as ex:
        print(f"An unexpected error occurred while speaking: {ex}")
    

def main():
    user_input: str = listen_for_audio()
    if user_input:
        ai_response: str = generate_ai_response(user_input)
        speak_response(ai_response)
    else:
        print("No valid input recieved")

if __name__ == "__main__":
    main()