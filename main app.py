import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import urllib.request

# HSV color ranges for shirt detection
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
        if lower[0] <= upper[0]:
            if lower[0] <= h <= upper[0] and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
                return True
        else:
            if (h >= lower[0] or h <= upper[0]) and lower[1] <= s <= upper[1] and lower[2] <= v <= upper[2]:
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

def strong_blur(image, ksize=101):
    k = max(3, ksize)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(image, (k, k), 0)

def process_frame(frame, model, target_color=None, blur_strength=101):
    results = model(frame)
    count = 0

    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            if width < 30 or height < 60:
                continue

            shirt_top = y1 + int(height * 0.4)
            shirt_bottom = y1 + int(height * 0.65)
            shirt_left = x1 + int(width * 0.3)
            shirt_right = x2 - int(width * 0.3)

            shirt_top = max(0, min(shirt_top, frame.shape[0]))
            shirt_bottom = max(0, min(shirt_bottom, frame.shape[0]))
            shirt_left = max(0, min(shirt_left, frame.shape[1]))
            shirt_right = max(0, min(shirt_right, frame.shape[1]))

            shirt_roi = frame[shirt_top:shirt_bottom, shirt_left:shirt_right]

            if shirt_roi.size == 0:
                continue

            dom_hsv = get_dominant_hsv(shirt_roi)
            is_match = False

            if target_color:
                is_match = (
                    target_color in HSV_COLOR_RANGES and
                    in_hsv_range(dom_hsv, HSV_COLOR_RANGES[target_color])
                )
            else:
                is_match = True

            if is_match:
                count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                blur_top = y1 + int(height * 0.3)
                blur_bottom = y1 + int(height * 0.75)
                blur_top = max(0, min(blur_top, frame.shape[0]))
                blur_bottom = max(0, min(blur_bottom, frame.shape[0]))
                blur_left = max(0, x1)
                blur_right = min(frame.shape[1], x2)

                torso_area = frame[blur_top:blur_bottom, blur_left:blur_right]
                if torso_area.size != 0:
                    blurred = strong_blur(torso_area, blur_strength)
                    frame[blur_top:blur_bottom, blur_left:blur_right] = blurred

    return frame, count

def display_image(image, model, color_filter, blur_strength):
    frame, count = process_frame(image, model, color_filter, blur_strength)
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             caption=f"{color_filter.capitalize() if color_filter else 'All'} shirts: {count}",
             channels="RGB")

def live_webcam(model, target_color, blur_strength):
    stframe = st.empty()
    run = st.checkbox("Start Webcam", key="start_webcam")
    cap = None
    if run:
        cap = cv2.VideoCapture(0)
        while st.session_state.get("start_webcam", False) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break
            processed_frame, count = process_frame(frame, model, target_color, blur_strength)
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                          caption=f"{target_color if target_color else 'All'} shirts: {count}",
                          channels="RGB")
        cap.release()
    else:
        st.warning("Webcam stopped or not started yet.")

def main():
    st.set_page_config(page_title="Shirt Color Detector", layout="wide")
    st.title("🧠 AI Shirt Color Detector with YOLOv8")
    st.markdown("Detects and counts people wearing shirts of a target color using **YOLOv8** and HSV logic.")

    download_yolo_model()
    model = YOLO("yolov8n.pt")

    st.sidebar.header("🎛️ Detection Settings")
    mode = st.sidebar.radio("Choose Mode", ["Image Upload (Multiple)", "Video Upload", "Live Webcam"])
    color_filter = st.sidebar.selectbox("Shirt Color Filter", ["Auto", "Red", "Blue", "Green", "Yellow", "White", "Black"])
    blur_strength = st.sidebar.slider("Blur Strength", min_value=11, max_value=151, step=10, value=101)
    target_color = color_filter.lower() if color_filter != "Auto" else None

    if mode == "Image Upload (Multiple)":
        uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            cols = st.columns(min(3, len(uploaded_files)))
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
                    tfile.write(uploaded_file.read())
                    temp_path = tfile.name
                img = cv2.imread(temp_path)
                if img is not None:
                    processed_img, count = process_frame(img, model, target_color, blur_strength)
                    with cols[i % len(cols)]:
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                                 caption=f"{uploaded_file.name} → {color_filter if target_color else 'All'} shirts: {count}",
                                 channels="RGB")
                os.unlink(temp_path)

    elif mode == "Video Upload":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_file:
            suffix = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tfile:
                tfile.write(uploaded_file.read())
                temp_path = tfile.name
            cap = cv2.VideoCapture(temp_path)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame, count = process_frame(frame, model, target_color, blur_strength)
                stframe.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                              channels="RGB",
                              caption=f"{color_filter if color_filter != 'Auto' else 'All'} shirts: {count}")
            cap.release()
            os.unlink(temp_path)

    elif mode == "Live Webcam":
        live_webcam(model, target_color, blur_strength)

if __name__ == "__main__":
    main()
