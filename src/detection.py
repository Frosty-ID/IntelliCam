import cv2
import torch
from ultralytics import YOLO



def load_model(model_path: str = 'model/yolov8n.pt') -> YOLO:
    """Loads the YOLO model with CUDA if available."""
    model = YOLO(model_path)
    if torch.cuda.is_available():
        model.to('cuda')
    return model



def initialize_camera(camera_index: int = 0) -> cv2.VideoCapture:
    """Initializes and returns the camera object."""
    cam = cv2.VideoCapture(camera_index)
    if not cam.isOpened():
        raise Exception("Could not open camera")
    return cam



def process_frame(frame, model, min_confidence=0.7):
    """Processes a single frame for object detection and draws bounding boxes."""
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if confidence > min_confidence:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, ( x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)

    return frame



def main() -> None:
    model = load_model()
    cam = initialize_camera()
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            frame = process_frame(frame, model)
            cv2.imshow('Object Detection', frame)

            # Break loop on any key press
            if cv2.waitKey(0):
                break

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        # Release resources
        cam.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main()