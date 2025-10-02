import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tempfile

# --- Config ---
MODEL1_PATH = 'yolov8n.pt'
MODEL2_PATH = r"D:\Uni\Project\Vehicle-Detection\runs\detect\vehicle_detector\weights\best.pt"
COCO_CLASSES = [1, 2, 3, 5]
DEFAULT_CONF = 0.4
LABEL_MAP = {'bus': 'heavy_truck', 'motorcycle': 'two_wheeled_vehicle', 'bicycle': 'two_wheeled_vehicle'}

# --- Load Models ---
@st.cache_resource
def load_models():
    return YOLO(MODEL1_PATH), YOLO(MODEL2_PATH)

# --- Utils ---
def iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter) if area1 + area2 - inter > 0 else 0

def draw_boxes(img, dets, conf_thresh):
    for d in dets:
        if d['conf'] < conf_thresh: continue
        x1, y1, x2, y2 = map(int, d['bbox'])
        label = LABEL_MAP.get(d['label'], d['label'])
        txt = f"{label} {d['conf']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-10), (x1+tw+10, y1), (255, 0, 0), -1)
        cv2.putText(img, txt, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def process_image(image, model1, model2, conf):
    r1, r2 = model1(image, conf=conf, classes=COCO_CLASSES, verbose=False)[0], model2(image, conf=conf, verbose=False)[0]
    dets, custom_boxes = [], []

    for box in r2.boxes:
        b = box.xyxy[0].tolist()
        dets.append({'bbox': b, 'conf': box.conf.item(), 'label': model2.names[int(box.cls)], 'src': 'custom'})
        custom_boxes.append(b)

    for box in r1.boxes:
        b = box.xyxy[0].tolist()
        if all(iou(b, cb) < 0.5 for cb in custom_boxes):
            dets.append({'bbox': b, 'conf': box.conf.item(), 'label': model1.names[int(box.cls)], 'src': 'coco'})

    return draw_boxes(image.copy(), dets, conf), dets

def process_video(path, model1, model2, conf):
    cap = cv2.VideoCapture(path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame, _ = process_image(frame, model1, model2, conf)
        out.write(frame)

    cap.release(), out.release()
    return output_path

# --- UI ---
st.set_page_config("Vehicle Detection", layout="wide", page_icon="🚗")
st.title("🚗 Vehicle Detector And Categorizer")
st.write("Upload an image or video to detect and classify vehicles.")

model1, model2 = load_models()
conf = st.slider("Confidence Threshold", 0.1, 1.0, DEFAULT_CONF, 0.05)
file = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])

if model1 and model2 and file:
    ext = file.name.split('.')[-1].lower()
    is_video = ext in ['mp4', 'avi', 'mov', 'mkv']

    if is_video:
        if (file.name != st.session_state.get('last_file') or conf != st.session_state.get('last_conf')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file.read())
                video_path = tmp.name

            with st.spinner("Processing video..."):
                out_path = process_video(video_path, model1, model2, conf)

            st.session_state['processed_video'] = out_path
            st.session_state['last_file'] = file.name
            st.session_state['last_conf'] = conf
        else:
            out_path = st.session_state['processed_video']

        with open(out_path, 'rb') as f:
            st.download_button("Download Processed Video", f.read(), file_name="processed_video.mp4")


    else:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        col1, col2 = st.columns(2)
        col1.image(img, channels="BGR", caption="Original Image", use_container_width=True)

        with st.spinner("Detecting vehicles..."):
            annotated, dets = process_image(img, model1, model2, conf)
        col2.image(annotated, channels="BGR", caption="Detected Vehicles", use_container_width=True)

        st.subheader("📊 Detection Details")
        st.dataframe(pd.DataFrame(
            [{"Final Class": LABEL_MAP.get(d['label'], d['label']), "Confidence": f"{d['conf']:.2%}"} for d in dets if d['conf'] >= conf]), use_container_width=True, hide_index=True)
#####################################################--PROGRAMMED BY ALI GHOLAMI--###########################################################################