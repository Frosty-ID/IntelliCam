import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the bounding box
            confidence = box.conf[0]  # Confidence score for the detection
            class_id = int(box.cls[0])  # Class ID of the detected object
            class_name = model.names[class_id]  # Get the class name using class ID

            if confidence > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                fontscale = 2.0
                thickness = 3
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()