import speech_recognition as sr
import pyttsx3
import cv2
from transformers import pipeline



recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")


def listen_for_audio() -> str:
    """Returns a string of user_input"""
    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, 0.6)
                audio = recognizer.listen(source)
                user_input = recognizer.recognize_google(audio)
                return user_input
            
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio.")
            print("Try Again: ....")
            continue
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as ex:
            print(f"An unexpected error occurred while listening: {ex}")
    


def generate_ai_response(user_input: str) -> str:
    """Processes user input & returns response"""
    try:
        response = generator(user_input, truncation=True)
        ai_response = response[0]['generated_text']
        return ai_response
    except Exception as ex:
        print(f"An error occurred while generating a response: {ex}")
        return "Sorry, I couldn't generate a response."



def speak_response(ai_response: str) -> None:
    try:
        tts_engine.say(ai_response)
        tts_engine.runAndWait()
    except Exception as ex:
        print(ex)



def main() -> None:
    while True:
        if cv2.waitKey(0):
            break

        user_input: str = listen_for_audio()
        if user_input:
            ai_response: str = generate_ai_response(user_input)
            speak_response(ai_response)
        else:
            print("No valid input recieved")



if __name__ == "__main__":
    main()