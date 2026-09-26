import streamlit as st
import cv2
from ultralytics import YOLO
import os

st.title("YOLOv8 Live Traffic Detection Feed")

# Load model
model = YOLO("yolov8n.pt")

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "traffic1.mp4")

if not os.path.exists(video_path):
    st.error(f"❌ Could not find 'traffic1.mp4' in: {current_dir}")
else:
    if st.button("Start Live Video Detection"):
        st.text("🚀 Processing and streaming video...")
        st_frame = st.empty()
        
        # Use YOLO's built-in generator stream which bypasses OpenCV codec issues
        results_stream = model.predict(source=video_path, stream=True, imgsz=1920, conf=0.3, iou=0.6, classes=[2, 3, 5, 7], verbose=False)
        
        for result in results_stream:
            annotated_frame = result.plot()
            # Convert BGR color to RGB for Streamlit display
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
            
        st.success("✅ Video processing complete!")