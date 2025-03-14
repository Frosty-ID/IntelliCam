import cv2
import torch
import sys
from ultralytics import YOLO

model = YOLO('model/yolov8n.pt')

if torch.cuda.is_available():
    model.to('cuda')

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    sys.exit(1)

min_confidence = 0.4

while True:
    try:
        successful, captured_frame = cam.read()
        if not successful:
            break

        results = model(captured_frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if confidence > min_confidence:
                    cv2.rectangle(captured_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{class_name}: {confidence:.2f}"
                    fontscale = 2.0
                    thickness = 3
                    cv2.putText(captured_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), thickness)
                
                cv2.imshow('Object Detection', captured_frame)

        if cv2.waitKey(1) == ord('q'):
            break

    except Exception as ex:
        print(f"Error has occured: {ex}")


cam.release()
cv2.destroyAllWindows()