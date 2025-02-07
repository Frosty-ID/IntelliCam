import cv2
import torch
import sys
from ultralytics import YOLO

# Initialize YOLO
model = YOLO('model/yolov8n.pt')

if torch.cuda.is_available():
    model.to('cuda')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    sys.exit(1)

spoken_to_user = False
min_confidence = 0.4

while True:
    try:
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

                if confidence > min_confidence:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    fontscale = 2.0
                    thickness = 3
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness)
                
                cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as ex:
        break

# Release the video capture object and close any open OpenCV windows
cap.release()
cv2.destroyAllWindows()