import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import urllib.request

# HSV color ranges for color detection
HSV_COLOR_RANGES = {
    "red":    [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
    "blue":   [(90, 50, 50), (130, 255, 255)],
    "green":  [(35, 50, 50), (85, 255, 255)],
    "yellow": [(20, 100, 100), (35, 255, 255)],
    "white":  [(0, 0, 200), (180, 40, 255)],
    "black":  [(0, 0, 0), (180, 255, 50)],
}

def download_yolo_model(path="yolov8n.pt"):
    if not os.path.exists(path):
        st.warning("Downloading YOLOv8n model...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        urllib.request.urlretrieve(url, path)
        st.success("Model downloaded.")

def in_hsv_range(hsv_color, ranges):
    h, s, v = hsv_color
    for i in range(0, len(ranges), 2):
        lower, upper = ranges[i], ranges[i+1]
        # Handle hue wrap-around (for red)
        if lower[0] <= upper[0]:
            if (lower[0] <= h <= upper[0] and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                return True
        else:
            if ((h >= lower[0] or h <= upper[0]) and
                lower[1] <= s <= upper[1] and
                lower[2] <= v <= upper[2]):
                return True
    return False

def get_dominant_hsv(image):
    if image.size == 0:
        return [0, 0, 0]
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    resized = cv2.resize(hsv, (50, 50))
    pixels = resized.reshape(-1, 3)
    hist = cv2.calcHist([pixels], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))
    median_s = int(np.median(pixels[:, 1]))
    median_v = int(np.median(pixels[:, 2]))
    return [dominant_hue, median_s, median_v]

def process_frame(frame, model, target_color=None):
    results = model(frame)
    count = 0
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:  # Person class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            top = max(0, y1 + int(height * 0.3))
            bottom = min(frame.shape[0], y1 + int(height * 0.7))
            left = max(0, x1)
            right = min(frame.shape[1], x2)
            if top >= bottom or left >= right:
                continue
            shirt_roi = frame[top:bottom, left:right]
            if shirt_roi.size == 0:
                continue
            dominant_color = get_dominant_hsv(shirt_roi)
            if target_color:
                if target_color in HSV_COLOR_RANGES and in_hsv_range(dominant_color, HSV_COLOR_RANGES[target_color]):
                    count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                count += 1
                color_bgr = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_HSV2BGR)[0][0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr.tolist(), 2)
    return frame, count

def display_image(image, model, color_filter):
    frame, count = process_frame(image, model, color_filter)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             caption=f"{color_filter.capitalize() if color_filter else 'All'} shirts: {count}",
             channels="RGB")

def main():
    st.title("👕 Robust Shirt Color Detector")
    st.markdown("Detects shirts even in **blurred** or **shaded** images using AI and color logic.")
    download_yolo_model()
    model = YOLO("yolov8n.pt")
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])
    color_filter = st.selectbox("Target Shirt Color", ["Auto", "Red", "Blue", "Green", "Yellow", "White", "Black"])
    if uploaded_file:
        target_color = color_filter.lower() if color_filter != "Auto" else None
        suffix = uploaded_file.name.split('.')[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name
        try:
            if suffix in ["jpg", "jpeg", "png"]:
                img = cv2.imread(temp_path)
                if img is not None:
                    display_image(img, model, target_color)
                else:
                    st.error("❌ Failed to load image. Please try another file.")
            elif suffix == "mp4":
                cap = cv2.VideoCapture(temp_path)
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    result_frame, count = process_frame(frame, model, target_color)
                    stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                  channels="RGB",
                                  caption=f"{color_filter if color_filter != 'Auto' else 'All'} shirts: {count}")
                cap.release()
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

if __name__ == "__main__":
    main()
