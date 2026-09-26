import cv2
from ultralytics import YOLO

# Load the nano model
model = YOLO("yolov8n.pt")

# Use a simple relative path since your terminal is already in the traffic folder
video_path = "traffic1.mp4"

cap = cv2.VideoCapture(video_path)
cv2.namedWindow("YOLOv8 Traffic Detection", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Could not open video file. Please check if it's named 'traffic.mp4'.")
else:
    print("Playing video... Press 'q' inside the video window to exit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = model(frame, conf=0.15, iou=0.45, imgsz= 1920, verbose=False)
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLOv8 Traffic Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()