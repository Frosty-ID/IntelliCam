import cv2
import torch
import logging
import time
import pyttsx3
from ultralytics import YOLO
from speech import stt_to_response_to_tts

# Configure Logging

logging.basicConfig(
    filename='monitoring.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# Initialize Object Detection Model (Yolo) & Text-To-Speech Engine(pyttsx3)
model = YOLO('model/yolov8n.pt')
engine = pyttsx3.init()


if torch.cuda.is_available():
    model.to('cuda')
    logging.info("Model using CUDA") 

cap = cv2.VideoCapture(0)

person_detected = False
if not cap.isOpened():
    logging.error("Could not open webcam")
    raise RuntimeError("Could not open Video Stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  
            confidence = box.conf[0]  
            class_id = int(box.cls[0])  
            class_name = model.names[class_id] 

            if confidence > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                fontscale = 2.0
                thickness = 3
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness)
            
            if class_name == 'person':
                engine.say("Hello their, if you need any help please feel free to ask?")
                logging.info("Engine: Hello their, if you need any help please feel free to ask?")
                engine.runAndWait()
                stt_to_response_to_tts()
                time.sleep(15)
        

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(10) == ord('q'):
        logging.info("User: Quit Application")
        break


# Release the video capture object and close any open OpenCV windows
cap.release()
cv2.destroyAllWindows()