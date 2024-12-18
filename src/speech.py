import openai
import os
import speech_recognition as sr
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize Speech Recognition
recognizer = sr.Recognizer()