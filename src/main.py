import cv2
import torch
import logging
import openai
import os
import sys
import time
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from ultralytics import YOLO

# Initialize All Models
tts_engine = pyttsx3.init()
recognizer = sr.Recognizer()
model = YOLO('model/yolov8n.pt')


load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')

if API_KEY:
    logging.info("Accessed OPENAI API")
else:
    raise ValueError("ERROR: OPENAI_API_KEY not found in environment variables")


def stt_to_response_to_tts():
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)


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
        logging.error(f"Error in speech-to-text or OpenAI API: {ex}")


# Configure Logging
logging.basicConfig(
    filename='monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if torch.cuda.is_available():
    model.to('cuda')
    logging.info("Model using CUDA")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    logging.error("Could not open webcam")
    sys.exit(1)

spoken_to_user = False
min_confidence = 0.4

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame, exiting...")
            break

        results = model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if confidence > min_confidence:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    fontscale = 2.0
                    thickness = 3
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness)
                
                cv2.imshow('Object Detection', frame)

                if class_name == 'person':
                    tts_engine.say("Hello there, if you need any help please feel free to ask?")
                    tts_engine.runAndWait()
                    time.sleep(5)
                    stt_to_response_to_tts()

        if cv2.waitKey(10) == ord('q'):
            logging.info("User: Quit Application")
            break

    except Exception as ex:
        logging.error(f"Unexpected error: {ex}")
        break

# Release the video capture object and close any open OpenCV windows
cap.release()
cv2.destroyAllWindows()